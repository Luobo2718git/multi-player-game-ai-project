import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th

# 导入环境和对手AI
from games.bomb.bomb_env import BombEnv
from .agents import BombAI # 你的对手AI

# 定义一个自定义的特征提取器来处理 Dict 观察空间
# 这将把图像和标量特征拼接起来
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1) # features_dim 会被覆盖

        # 图像特征提取器 (CNN)
        # 假设 image_observation_space 是 (channels, height, width)
        n_channels = observation_space["board_features"].shape[0]
        board_height = observation_space["board_features"].shape[1]
        board_width = observation_space["board_features"].shape[2]

        # 简单的CNN架构
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # 计算CNN输出的维度
        with th.no_grad():
            dummy_input = th.as_tensor(np.zeros((1, n_channels, board_height, board_width)), dtype=th.float32)
            cnn_output_dim = self.cnn(dummy_input).shape[1]

        # 标量特征的维度
        scalar_feature_dim = observation_space["scalar_features"].shape[0]

        # 合并后的特征维度
        self._features_dim = cnn_output_dim + scalar_feature_dim

        self.linear = nn.Sequential(
            nn.Linear(self._features_dim, 256),
            nn.ReLU()
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        board_features = observations["board_features"].float() / BombGame.WILL_EXPLOSION # 归一化图像特征
        scalar_features = observations["scalar_features"].float() / 1000.0 # 归一化标量特征，假设最大值1000

        cnn_out = self.cnn(board_features)
        combined_features = th.cat((cnn_out, scalar_features), dim=1)
        return self.linear(combined_features)


def make_env(rank: int, seed: int = 0, player_id: int = 1, opponent_agent=None):
    """
    用于SubprocVecEnv创建环境的函数
    """
    def _init():
        env = BombEnv(board_size=15, player_id=player_id, opponent_agent=opponent_agent)
        # env.seed(seed + rank) # Gym 0.21 可能不支持 .seed()
        return env
    # 设置随机种子
    # np.random.seed(seed + rank) # 确保每个环境有不同的随机种子
    # random.seed(seed + rank)
    return _init

if __name__ == '__main__':
    LOG_DIR = "./ppo_bomb_logs/"
    os.makedirs(LOG_DIR, exist_ok=True)

    # 实例化你的对手AI
    opponent_ai = BombAI(name="OpponentBombAI", player_id=2) # RL智能体作为玩家1，AI作为玩家2

    # 创建多个并行环境进行训练
    num_envs = 8 # 可以根据你的CPU核心数调整
    # 使用 SubprocVecEnv 以利用多核CPU加速训练
    # 注意：传递对象（如 opponent_ai）给子进程需要其是可序列化的
    # BombAI应该是可序列化的
    vec_env = SubprocVecEnv([make_env(i, player_id=1, opponent_agent=opponent_ai) for i in range(num_envs)])
    # vec_env = DummyVecEnv([make_env(0, player_id=1, opponent_agent=opponent_ai)]) # 单环境调试用


    # 定义PPO模型的策略网络架构
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        net_arch=dict(pi=[64, 64], vf=[64, 64]) # 策略网络和价值网络结构
    )

    # 创建PPO模型
    model = PPO(
        "MultiInputPolicy", # 用于处理 Dict 观察空间
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=LOG_DIR,
        gamma=0.99, # 折扣因子
        n_steps=2048, # 收集多少步数据后进行一次学习更新
        ent_coef=0.01, # 熵正则化系数，鼓励探索
        learning_rate=0.0003,
        clip_range=0.2,
        batch_size=64,
        gae_lambda=0.95
    )

    # 设置回调函数：保存最佳模型和定期保存检查点
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, # 每100000步保存一次模型
        save_path="./models/",
        name_prefix="ppo_bomb_model"
    )

    # 创建一个独立的评估环境（不用于训练，只用于评估）
    eval_env = make_env(0, player_id=1, opponent_agent=opponent_ai)() # 单独的环境实例
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./results/",
        eval_freq=50000, # 每50000步评估一次
        deterministic=True, # 评估时使用确定性策略
        render=False # 评估时不渲染
    )

    print("开始训练PPO智能体...")
    total_timesteps = 5000000 # 总训练步数，可以调整
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
    print("训练完成！")

    # 保存最终模型
    model.save("ppo_bomb_final_model")

    # 测试训练好的模型 (可选)
    print("测试训练好的模型...")
    obs, info = eval_env.reset()
    for _ in range(100): # 运行100步
        action, _states = model.predict(obs, deterministic=True, action_masks=np.array([eval_env._get_action_mask(eval_env.player_id)]))
        # action_masks 应该是一个 batch 的掩码，这里只传一个
        # 对于 PPO，action_masks 的支持可能依赖于 SB3 的版本和特定实现
        # 如果不工作，你可能需要在 CustomPolicy 中自定义 ActionDistribution
        # 或者在环境 step 中处理无效动作的惩罚
        
        obs, reward, done, truncated, info = eval_env.step(action[0]) # action[0]因为 predict 返回的是批次
        eval_env.render() # 如果你实现了 render 方法
        if done or truncated:
            print(f"回合结束，获胜者: {info.get('winner', 'N/A')}, 玩家1分数: {info.get('player1_score', 'N/A')}, 玩家2分数: {info.get('player2_score', 'N/A')}")
            obs, info = eval_env.reset()
    eval_env.close()
