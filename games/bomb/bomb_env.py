import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from games.base_env import BaseEnv
from games.bomb.bomb_game import BombGame

class BombEnv(BaseEnv):
    """
    泡泡堂（炸弹人）环境
    实现gym风格接口
    """

    def __init__(self, board_size=15, **kwargs):
        self.board_size = board_size
        self.game = BombGame(board_size=board_size, **kwargs)
        super().__init__(self.game)

    def _setup_spaces(self):
        """设置观察空间和动作空间"""
        # 观察空间: 棋盘状态 (board_size x board_size)
        # 动作空间: 上，下，左，右，放置炸弹，不动
        # 我们在这里不严格定义gym.spaces，但保留方法签名
        self.observation_space = None 
        # 动作是 (dr, dc) 或 (0,0,True)
        self.action_space = None 

    def _get_observation(self) -> Dict[str, Any]:
        """获取观察"""
        return self.game.get_state()

    def _get_action_mask(self, player_id: int) -> np.ndarray:
        """获取动作掩码"""
        # 对于炸弹人，动作掩码可能更复杂，因为动作包含移动和放置炸弹。
        # 这里返回所有可能的原始动作，AI需要自行过滤。
        # 动作定义为: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: BOMB, 5: STAY
        
        # 默认所有动作都不可用
        action_mask = np.zeros(6, dtype=bool) 
        
        valid_game_actions = self.game.get_valid_actions(player_id)
        
        # 将BombGame的动作映射到Env的动作索引
        action_map = {
            (-1, 0): 0,  # UP
            (1, 0): 1,   # DOWN
            (0, -1): 2,  # LEFT
            (0, 1): 3,   # RIGHT
            (0, 0, True): 4, # BOMB
            (0, 0): 5    # STAY
        }

        for action in valid_game_actions:
            # 检查action是否是(0,0,True)这种三元组
            if isinstance(action, tuple) and len(action) == 3 and action[2]:
                if (0, 0, True) in action_map:
                    action_mask[action_map[(0, 0, True)]] = True
            elif action in action_map:
                action_mask[action_map[action]] = True
        
        return action_mask

    def get_valid_actions(self, player_id: int) -> List[Union[Tuple[int, int], Tuple[int, int, bool]]]:
        """
        获取游戏中的有效动作，与BombGame中的定义一致。
        返回 (dr, dc) 或 (0, 0, True) 形式的列表。
        """
        return self.game.get_valid_actions(player_id)

    def is_terminal(self) -> bool:
        """检查游戏是否结束"""
        return self.game.is_terminal()

    def get_winner(self) -> Optional[int]:
        """获取赢家"""
        return self.game.get_winner()

    def get_game_info(self) -> Dict[str, Any]:
        """获取游戏信息"""
        info = self.game.get_game_info()
        info.update({
            'board_size': self.board_size,
            'player1_pos': self.game.player1_pos,
            'player2_pos': self.game.player2_pos,
            'alive1': self.game.alive1,
            'alive2': self.game.alive2,
            'player1_bombs_max': self.game.player1_bombs_max,
            'player2_bombs_max': self.game.player2_bombs_max,
            'player1_range': self.game.player1_range,
            'player2_range': self.game.player2_range,
            'player1_current_bombs': self.game.player1_current_bombs,
            'player2_current_bombs': self.game.player2_current_bombs,
            'player1_shield_active': self.game.player1_shield_active, # 新增
            'player2_shield_active': self.game.player2_shield_active, # 新增
            'player1_shield_timer': self.game.player1_shield_timer, # 新增
            'player2_shield_timer': self.game.player2_shield_timer, # 新增
            'player1_range_type': self.game.player1_range_type, # 新增
            'player2_range_type': self.game.player2_range_type, # 新增
            'current_moves': self.game.current_moves,
            'max_moves': self.game.max_moves,
            'winner': self.game.winner
        })
        return info

    def clone(self) -> 'BombEnv':
        """克隆环境"""
        cloned_game = self.game.clone()
        cloned_env = BombEnv(self.board_size)
        cloned_env.game = cloned_game
        return cloned_env

    def step(self, action1: Union[Tuple[int, int], Tuple[int, int, bool]], action2: Union[Tuple[int, int], Tuple[int, int, bool], None] = None) -> Tuple[Dict[str, Any], float, float, bool, Dict[str, Any]]:
        """
        执行一步游戏，接受两个玩家的动作。
        返回 (observation, reward1, reward2, terminal, info)
        """
        observation, reward1, reward2, terminal, info = self.game.step(action1, action2)
        return observation, reward1, reward2, terminal, info
    
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """重置游戏环境并返回初始观察。"""
        self.game._reset_game()
        return self.game.get_state()
