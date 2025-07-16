import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

# 导入环境和对手AI
from games.bomb.bomb_env import BombEnv
from agents import BombAI # 您的对手AI

def make_env(player_id: int = 1, opponent_agent=None):
    """
    用于创建环境的函数。
    """
    def _init():
        # 注意：这里我们只创建一个环境，因为是单局游戏
        env = BombEnv(board_size=15, player_id=player_id, opponent_agent=opponent_agent)
        # 在玩游戏时，通常不需要设置随机种子，除非您想复现特定对局
        env.reset() 
        return env
    return _init

if __name__ == '__main__':
    # 定义模型和日志的路径
    MODEL_PATH = "./best_model/best_model.zip" # 确保这是您最佳模型的正确路径

    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误：未找到模型文件在 {MODEL_PATH}。请确保您已经训练并保存了模型。")
        exit()

    print(f"正在加载模型：{MODEL_PATH}")
    try:
        # 加载训练好的PPO模型
        # 由于训练时使用了MultiInputPolicy，加载时也需要指定
        model = PPO.load(MODEL_PATH)
        print("模型加载成功！")
    except Exception as e:
        print(f"加载模型时发生错误：{e}")
        exit()

    # 实例化您的对手AI (假设RL智能体是玩家1，BombAI是玩家2)
    opponent_ai = BombAI(name="OpponentBombAI", player_id=2) 
    print("对手AI实例化成功。")

    # 创建一个单环境，用于RL智能体和AI对手之间的对局
    # 使用DummyVecEnv包装，即使是单个环境，也能保持与Stable Baselines3训练接口的一致性
    eval_env = DummyVecEnv([make_env(player_id=1, opponent_agent=opponent_ai)])
    print("游戏环境创建成功。")

    print("\n--- 开始RL智能体 vs. AI对手的对局 ---")
    
    # 重置环境以开始新对局
    obs, info = eval_env.reset()
    
    done = False
    truncated = False
    episode_step_count = 0
    max_test_steps = 1000 # 设置一个最大测试步数，防止无限循环

    while not (done or truncated) and episode_step_count < max_test_steps:
        # RL智能体 (玩家1) 预测动作
        # model.predict 返回的是一个批处理的动作，即使只有一个环境，也需要取第一个元素
        action_player1, _states = model.predict(obs, deterministic=True)
        
        # 获取对手AI (玩家2) 的动作
        # opponent_ai.get_action 期望的是原始的 BombGame 实例，而不是 BombEnv
        # 因此，我们需要从 eval_env.envs[0] 中获取实际的 BombGame 实例
        actual_game_instance = eval_env.envs[0].game # 获取 BombEnv 内部的 BombGame 实例
        opponent_observation_for_ai = actual_game_instance.get_state() # 为AI生成观察
        action_player2 = opponent_ai.get_action(opponent_observation_for_ai, actual_game_instance)

        # 执行环境步进
        # eval_env.step 期望的是批处理的动作，所以直接传入 action_player1
        # 对于 action_player2，如果您的 BombEnv.step 期望的是单个动作，可能需要调整
        # 假设 BombEnv.step 内部会处理两个玩家的动作
        # 注意：这里 reward, done, truncated, info 都是批处理的，需要取 [0]
        obs, reward, done, truncated, info = eval_env.step(action_player1) # action_player1 已经是批处理的

        episode_step_count += 1

        # 打印当前游戏信息 (可以根据需要调整打印频率)
        current_info = info[0] # 获取第一个环境的信息
        print(f"步数: {episode_step_count}, 玩家1分数: {current_info.get('player1_score', 'N/A')}, 玩家2分数: {current_info.get('player2_score', 'N/A')}, 玩家1存活: {current_info.get('alive1', 'N/A')}, 玩家2存活: {current_info.get('alive2', 'N/A')}")

        if done[0] or truncated[0]:
            print("\n--- 对局结束 ---")
            print(f"总步数: {episode_step_count}")
            print(f"获胜者: {current_info.get('winner', 'N/A')}")
            print(f"最终玩家1分数: {current_info.get('player1_score', 'N/A')}")
            print(f"最终玩家2分数: {current_info.get('player2_score', 'N/A')}")
            break # 退出循环

    if not (done[0] or truncated[0]):
        print("\n--- 对局达到最大步数，未决出胜负 ---")
        print(f"总步数: {episode_step_count}")
        current_info = info[0]
        print(f"最终玩家1分数: {current_info.get('player1_score', 'N/A')}")
        print(f"最终玩家2分数: {current_info.get('player2_score', 'N/A')}")

    eval_env.close()
    print("游戏环境已关闭。")
