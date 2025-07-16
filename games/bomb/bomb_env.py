import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional, Union
from games.bomb.bomb_game import BombGame
import config # 导入 config for max_moves

class BombEnv(gym.Env):
    """
    泡泡堂（炸弹人）环境
    实现gymnasium风格接口
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, board_size=15, player_id: int = 1, opponent_agent=None, render_mode: Optional[str] = None, **kwargs):
        super().__init__()
        self.board_size = board_size
        self.player_id = player_id
        self.opponent_agent = opponent_agent
        self.render_mode = render_mode
        self.game = BombGame(board_size=board_size, **kwargs)

        # 定义观察空间
        # board_features: (channels, height, width) -> (1, board_size, board_size) 用于单通道棋盘
        # 值范围从 0 (EMPTY) 到 BombGame.WILL_EXPLOSION (44)
        self.observation_space = spaces.Dict(
            {
                "board_features": spaces.Box(
                    low=0,
                    high=BombGame.WILL_EXPLOSION,
                    shape=(1, self.board_size, self.board_size), # (channels, height, width)
                    dtype=np.int32, # 使用 int32，因为棋盘元素是整数ID
                ),
                # 标量特征: 23个值
                # 包含: 玩家1位置(2), 玩家2位置(2), 玩家1存活(1), 玩家2存活(1),
                # 玩家1属性(炸弹上限, 当前炸弹, 范围, 护盾激活, 护盾计时器, 范围类型[数值]) (6)
                # 玩家2属性(炸弹上限, 当前炸弹, 范围, 护盾激活, 护盾计时器, 范围类型[数值]) (6)
                # 游戏全局状态(当前步数, 最大步数, 赢家, 玩家1分数, 玩家2分数) (5)
                # 总计: 2+2+1+1 + 6 + 6 + 5 = 23 个标量值
                "scalar_features": spaces.Box(
                    low=0,
                    # 定义一个高值，确保能覆盖所有标量特征的最大可能值
                    # 例如，棋盘大小、护盾持续时间、最大移动步数、分数等
                    # 确保是 float 类型
                    high=float(max(self.board_size, BombGame.SHIELD_DURATION, config.GAME_CONFIGS['bomb']['max_moves'], 1000)), # 1000作为分数的安全上限
                    shape=(23,), # 修正后的标量特征数量
                    dtype=np.float32, # 使用 float32，因为RL模型通常需要浮点数
                ),
            }
        )

        # 定义动作空间: 6个离散动作
        # 0: Up (-1, 0)
        # 1: Down (1, 0)
        # 2: Left (0, -1)
        # 3: Right (0, 1)
        # 4: Stay (0, 0)
        # 5: Plant Bomb (0, 0, True) - 原地放置炸弹
        self.action_space = spaces.Discrete(6)

        # 初始化游戏状态
        self.game.reset()

    def _get_observation(self) -> Dict[str, Any]:
        """获取观察，并格式化为Dict observation space"""
        state = self.game.get_state() 
        
        # Board features: 确保它是一个3D数组 (1, H, W)
        # 使用 np.asarray 确保是 NumPy 数组，并使用 .copy() 避免视图问题
        board_features = np.asarray(state['board'], dtype=np.int32).copy()
        if board_features.ndim == 2: # 如果是 (H, W) 2D数组，转换为 (1, H, W)
            board_features = board_features.reshape(1, self.board_size, self.board_size)
        elif board_features.ndim != 3 or board_features.shape[0] != 1:
            # 如果不是预期的 (1, H, W) 形状，则报错
            raise ValueError(f"Board features from game.get_state() has unexpected shape: {board_features.shape}")

        # 标量特征: 转换字符串类型并确保都是数值
        player1_range_type_numeric = 0.0 if state['player1_range_type'] == 'cross' else 1.0
        player2_range_type_numeric = 0.0 if state['player2_range_type'] == 'cross' else 1.0
        winner_numeric = float(state['winner']) if state['winner'] is not None else 0.0

        # 收集标量特征，并确保所有元素都转换为浮点数
        scalar_values = [
            float(state['player1_pos'][0]), float(state['player1_pos'][1]),
            float(state['player2_pos'][0]), float(state['player2_pos'][1]),
            float(int(state['alive1'])), float(int(state['alive2'])), # bool转int再转float
            float(state['player1_bombs_max']), float(state['player1_current_bombs']), float(state['player1_range']),
            float(int(state['player1_shield_active'])), float(state['player1_shield_timer']), player1_range_type_numeric,
            float(state['player2_bombs_max']), float(state['player2_current_bombs']), float(state['player2_range']),
            float(int(state['player2_shield_active'])), float(state['player2_shield_timer']), player2_range_type_numeric,
            float(state['current_moves']), float(state['max_moves']),
            winner_numeric,
            float(state['player1_score']), float(state['player2_score'])
        ]
        scalar_features = np.array(scalar_values, dtype=np.float32)
        
        # 确保标量特征是1D数组且长度正确
        if scalar_features.ndim != 1 or scalar_features.shape[0] != 23:
            raise ValueError(f"Scalar features has unexpected shape: {scalar_features.shape}, expected (23,)")

        return {
            "board_features": board_features,
            "scalar_features": scalar_features,
        }

    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        # 直接从 game 对象获取信息
        info = {
            'player1_pos': self.game.player1_pos,
            'player2_pos': self.game.player2_pos,
            'player1_alive': self.game.alive1,
            'player2_alive': self.game.alive2,
            'player1_bombs_max': self.game.player1_bombs_max,
            'player2_bombs_max': self.game.player2_bombs_max,
            'player1_range': self.game.player1_range,
            'player2_range': self.game.player2_range,
            'player1_current_bombs': self.game.player1_current_bombs,
            'player2_current_bombs': self.game.player2_current_bombs,
            'player1_shield_active': self.game.player1_shield_active,
            'player2_shield_active': self.game.player2_shield_active,
            'player1_shield_timer': self.game.player1_shield_timer,
            'player2_shield_timer': self.game.player2_shield_timer,
            'player1_range_type': self.game.player1_range_type,
            'player2_range_type': self.game.player2_range_type,
            'current_moves': self.game.current_moves,
            'max_moves': self.game.max_moves,
            'winner': self.game.winner,
            'player1_score': self.game.player1_score,
            'player2_score': self.game.player2_score
        }
        return info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        重置环境，返回初始观察和信息。
        Gymnasium风格的reset方法
        """
        super().reset(seed=seed) # 重要: 调用gym.Env的reset以设置种子
        self.game.reset() # 重置底层游戏状态
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        执行一步游戏，接受一个离散动作。
        返回 (observation, reward, terminated, truncated, info)
        """
        # 将离散动作映射到游戏所需的动作格式
        action_player1: Union[Tuple[int, int], Tuple[int, int, bool]]
        if action == 0: # 上
            action_player1 = (-1, 0)
        elif action == 1: # 下
            action_player1 = (1, 0)
        elif action == 2: # 左
            action_player1 = (0, -1)
        elif action == 3: # 右
            action_player1 = (0, 1)
        elif action == 4: # 不动
            action_player1 = (0, 0)
        elif action == 5: # 放置炸弹 (原地放置)
            action_player1 = (0, 0, True)
        else:
            raise ValueError(f"无效动作: {action}")

        # 获取对手的动作 (如果有)
        action_player2 = None
        if self.opponent_agent:
            # 获取对手AI所需的原始游戏状态
            opponent_observation_for_ai = self.game.get_state() # 对手AI可能需要完整的state dict
            action_player2 = self.opponent_agent.get_action(opponent_observation_for_ai, self.game)
            # 确保对手的动作格式正确
            if isinstance(action_player2, tuple) and len(action_player2) == 2:
                action_player2 = (action_player2[0], action_player2[1], False)


        # 执行游戏一步
        # game.step 返回 (reward1, reward2, terminated, truncated)
        reward1, reward2, terminated, truncated = self.game.step(action_player1, action_player2)
        
        # 根据当前RL智能体的player_id选择对应的奖励
        player_reward = reward1 if self.player_id == 1 else reward2

        observation = self._get_observation()
        info = self._get_info()

        return observation, player_reward, terminated, truncated, info

    def render(self):
        """渲染环境 (如果需要可视化)"""
        # 为了简单起见，此环境不直接实现渲染，
        # 因为可能有一个单独的GUI来处理。
        if self.render_mode == "human":
            print("BombEnv中未直接实现human模式的渲染。")
        elif self.render_mode == "rgb_array":
            # 返回一个占位符图像数组
            return np.zeros((100, 100, 3), dtype=np.uint8) 
        pass

    def close(self):
        """关闭环境，释放资源"""
        pass

    def clone(self) -> 'BombEnv':
        """克隆环境"""
        # 这个方法不严格属于Gymnasium API，但可能被外部工具或特定算法使用。
        # 确保它返回一个新的、独立的实例。
        cloned_game = self.game.clone()
        cloned_env = BombEnv(self.board_size, player_id=self.player_id, opponent_agent=self.opponent_agent)
        cloned_env.game = cloned_game
        return cloned_env