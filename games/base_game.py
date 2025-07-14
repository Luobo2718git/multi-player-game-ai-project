"""
游戏基类
定义所有游戏的基本接口
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import config


class BaseGame(ABC):
    """游戏基类"""
    
    def __init__(self, game_config: Dict[str, Any] = None):
        self.game_config = game_config or {}
        self.current_player = 1  # 1 或 2
        self.game_state = config.GameState.ONGOING
        self.move_count = 0
        self.start_time = time.time()
        self.last_move_time = time.time()
        self.history = []  # 游戏历史记录
        
        #self.reset()
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """重置游戏状态"""
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: 动作
            
        Returns:
            observation: 观察状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        pass
    
    @abstractmethod
    def get_valid_actions(self, player: int = None) -> List[Any]:
        """获取有效动作列表"""
        pass
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """检查游戏是否结束"""
        pass
    
    @abstractmethod
    def get_winner(self) -> Optional[int]:
        """获取获胜者 (1, 2, 或 None表示平局)"""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """获取当前游戏状态"""
        pass
    
    @abstractmethod
    def render(self) -> Any:
        """渲染游戏画面"""
        pass
    
    def switch_player(self):
        """切换玩家"""
        self.current_player = 3 - self.current_player  # 1 -> 2, 2 -> 1
    
    def is_timeout(self) -> bool:
        """检查是否超时"""
        if 'timeout' in self.game_config:
            return time.time() - self.last_move_time > self.game_config['timeout']
        return False
    
    def is_max_moves_reached(self) -> bool:
        """检查是否达到最大步数"""
        if 'max_moves' in self.game_config:
            return self.move_count >= self.game_config['max_moves']
        return False
    '''
    def update_game_state(self):
        """更新游戏状态"""
        if self.is_terminal():
            winner = self.get_winner()
            if winner == 1:
                self.game_state = config.GameState.PLAYER1_WIN
            elif winner == 2:
                self.game_state = config.GameState.PLAYER2_WIN
            else:
                self.game_state = config.GameState.DRAW
        elif self.is_timeout():
            self.game_state = config.GameState.TIMEOUT
        elif self.is_max_moves_reached():
            self.game_state = config.GameState.DRAW
    
    def get_game_info(self) -> Dict[str, Any]:
        """获取游戏信息"""
        return {
            'current_player': self.current_player,
            'game_state': self.game_state,
            'move_count': self.move_count,
            'elapsed_time': time.time() - self.start_time,
            'last_move_time': time.time() - self.last_move_time,
            'history': self.history.copy()
        }
    '''

    def update_game(self):
        """更新游戏状态"""
        if self.game_over or self.paused:
            return

        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return

        self.last_update = current_time

    # AI回合
        if not isinstance(self.current_agent, HumanAgent):
            try:
                observation = self.env._get_observation()
                action = self.current_agent.get_action(observation, self.env)
                if action:
                    self._make_move(action)
                self.thinking = False
            except Exception as e:
                print(f"AI thinking failed: {e}")
                self.current_agent = self.human_agent
                self.thinking = False

    # 玩家回合时，不要自动移动蛇，等待玩家按键
    
    def record_move(self, player: int, action: Any, result: Dict[str, Any] = None):
        """记录移动"""
        move_record = {
            'player': player,
            'action': action,
            'timestamp': time.time(),
            'result': result or {}
        }
        self.history.append(move_record)
        self.move_count += 1
        self.last_move_time = time.time()
    
    def get_legal_actions(self, player: int = None) -> List[Any]:
        """获取合法动作（别名）"""
        return self.get_valid_actions(player)
    
    def clone(self) -> 'BaseGame':
        """克隆游戏状态"""
        # 子类需要实现具体的克隆逻辑
        raise NotImplementedError("子类必须实现clone方法")
    
    def get_action_space(self) -> Any:
        """获取动作空间"""
        # 子类需要实现具体的动作空间定义
        raise NotImplementedError("子类必须实现get_action_space方法")
    
    def get_observation_space(self) -> Any:
        """获取观察空间"""
        # 子类需要实现具体的观察空间定义
        raise NotImplementedError("子类必须实现get_observation_space方法") 