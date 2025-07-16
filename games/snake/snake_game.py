"""
贪吃蛇游戏逻辑（简化版）
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
from ..base_game import BaseGame
import config


class SnakeGame(BaseGame):
    """双人贪吃蛇游戏"""
    
    def __init__(self, board_size: int = 20, initial_length: int = 3, food_count: int = 5):
        game_config = {
            'board_size': board_size,
            'initial_length': initial_length,
            'food_count': food_count,
            'timeout': config.GAME_CONFIGS['snake']['timeout'],
            'max_moves': config.GAME_CONFIGS['snake']['max_moves']
        }
        super().__init__(game_config)
        
        self.board_size = board_size
        self.initial_length = initial_length
        self.food_count = food_count
        
        # 蛇的位置和方向
        self.snake1 = []  # 玩家1的蛇
        self.snake2 = []  # 玩家2的蛇
        self.direction1 = (0, 1)  # 玩家1的方向
        self.direction2 = (0, -1)  # 玩家2的方向
        
        # 食物位置
        self.foods = []
        
        # 游戏状态
        self.alive1 = True
        self.alive2 = True
        self.reset()
    
    def reset(self) -> Dict[str, Any]:
        """重置游戏状态"""
        # 初始化蛇的位置
        center = self.board_size // 2
        self.snake1 = [(center, center - 2)]
        self.snake2 = [(center, center + 2)]
        
        # 初始化方向
        self.direction1 = (0, 1)  # 向右
        self.direction2 = (0, -1)  # 向左
        
        # 初始化食物
        self.foods = []
        self._generate_foods()
        
        # 重置游戏状态
        self.alive1 = True
        self.alive2 = True
        self.current_player = 1
        self.game_state = config.GameState.ONGOING
        self.move_count = 0
        self.history = []
        
        return self.get_state()
    
    def step(self, action1: Tuple[int, int], action2: Tuple[int, int]) -> Tuple[Dict[str, Any], float, float, bool, Dict[str, Any]]:
        """
        同时移动两条蛇
        Args:
            action1: 玩家1的动作
            action2: 玩家2的动作
        Returns:
            observation: 观察状态
            reward1: 玩家1奖励
            reward2: 玩家2奖励
            done: 是否结束
            info: 额外信息
        """
        # 记录旧存活状态
        old_alive1 = self.alive1
        old_alive2 = self.alive2

        # 更新方向
        if self.alive1:
            self.direction1 = action1
        if self.alive2:
            self.direction2 = action2

        # 同时移动两条蛇
        if self.alive1:
            self._move_snake(1)
        if self.alive2:
            self._move_snake(2)

        # 检查游戏是否结束
        done = self._check_game_over()

        # 分别计算奖励
        reward1 = self._reward_for_player(1, old_alive1)
        reward2 = self._reward_for_player(2, old_alive2)

        # 获取观察状态
        observation = self.get_state()

        # 额外信息
        info = {
            'snake1_length': len(self.snake1),
            'snake2_length': len(self.snake2),
            'food_count': len(self.foods),
            'alive1': self.alive1,
            'alive2': self.alive2
        }
        return observation, reward1, reward2, done, info

    # === AI助手修改: 兼容单人step接口，方便多游戏GUI统一调用 ===
    def step_single(self, action: Tuple[int, int], player: int = 1):
        """
        兼容单人step接口，自动补全另一个玩家的动作为当前方向
        目的：让SnakeGame支持只传一个动作，便于和GomokuGame统一接口
        """
        if player == 1:
            return self.step(action, self.direction2)
        else:
            return self.step(self.direction1, action)

    @property
    def board(self):
        """
        兼容GomokuGame的board属性，返回当前棋盘
        目的：让GUI代码可以统一用game.board访问棋盘
        """
        return self.get_state()['board']

    def update_game_state(self):
        """更新游戏状态"""
        # 贪吃蛇游戏不需要额外的状态更新
        pass
    
    def get_game_info(self) -> Dict[str, Any]:
        """获取游戏信息"""
        return {
            'board_size': self.board_size,
            'initial_length': self.initial_length,
            'food_count': self.food_count,
            'snake1_length': len(self.snake1),
            'snake2_length': len(self.snake2),
            'alive1': self.alive1,
            'alive2': self.alive2,
            'current_player': self.current_player,
            'game_state': self.game_state,
            'move_count': self.move_count
        }
    
    def get_valid_actions(self, player: int = None) -> List[Tuple[int, int]]:
        """获取有效动作列表"""
        # 四个方向：上、下、左、右
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        if player is None:
            player = self.current_player
        
        # 过滤掉反向移动
        current_direction = self.direction1 if player == 1 else self.direction2
        valid_directions = []
        
        for direction in directions:
            if direction != (-current_direction[0], -current_direction[1]):
                valid_directions.append(direction)
        
        return valid_directions
    
    def is_terminal(self) -> bool:
        """检查游戏是否结束"""
        return not (self.alive1 or self.alive2)
    
    def get_winner(self) -> Optional[int]:
        """获取获胜者"""
        # 先判断是否有一方长度达到15
        if len(self.snake1) >= 15 and self.alive1:
            return 1
        if len(self.snake2) >= 15 and self.alive2:
            return 2
        # 再判断死亡
        if not self.alive1 and self.alive2:
            return 2
        if not self.alive2 and self.alive1:
            return 1
        # 同时死亡或都未达成胜利条件
        return None
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前游戏状态"""
        # 创建棋盘
        board = np.zeros((self.board_size, self.board_size), dtype=int)
        
        # 绘制蛇1
        for i, (x, y) in enumerate(self.snake1):
            if 0 <= x < self.board_size and 0 <= y < self.board_size:
                board[x, y] = 1 if i == 0 else 2  # 头部为1，身体为2
        
        # 绘制蛇2
        for i, (x, y) in enumerate(self.snake2):
            if 0 <= x < self.board_size and 0 <= y < self.board_size:
                board[x, y] = 3 if i == 0 else 4  # 头部为3，身体为4
        
        # 绘制食物
        for x, y in self.foods:
            if 0 <= x < self.board_size and 0 <= y < self.board_size:
                board[x, y] = 5
        
        return {
            'board': board,
            'snake1': self.snake1.copy(),
            'snake2': self.snake2.copy(),
            'foods': self.foods.copy(),
            'direction1': self.direction1,
            'direction2': self.direction2,
            'alive1': self.alive1,
            'alive2': self.alive2,
            'current_player': self.current_player,
            'valid_actions': self.get_valid_actions(),
            'game_state': self.game_state,
            'move_count': self.move_count
        }
    
    def render(self) -> np.ndarray:
        """渲染游戏画面"""
        state = self.get_state()
        return state['board']
    
    def clone(self) -> 'SnakeGame':
        """克隆游戏状态"""
        cloned_game = SnakeGame(self.board_size, self.initial_length, self.food_count)
        cloned_game.snake1 = self.snake1.copy()
        cloned_game.snake2 = self.snake2.copy()
        cloned_game.direction1 = self.direction1
        cloned_game.direction2 = self.direction2
        cloned_game.foods = self.foods.copy()
        cloned_game.alive1 = self.alive1
        cloned_game.alive2 = self.alive2
        cloned_game.current_player = self.current_player
        cloned_game.game_state = self.game_state
        cloned_game.move_count = self.move_count
        cloned_game.history = self.history.copy()
        return cloned_game
    
    def get_action_space(self):
        """获取动作空间"""
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def get_observation_space(self):
        """获取观察空间"""
        return {
            'board': (self.board_size, self.board_size),
            'snake1': [],
            'snake2': [],
            'foods': []
        }
    
    def _move_snake(self, player: int):
        """移动蛇"""
        if player == 1:
            snake = self.snake1
            direction = self.direction1
            alive = self.alive1
        else:
            snake = self.snake2
            direction = self.direction2
            alive = self.alive2
        
        if not alive:
            return
        
        # 计算新头部位置
        head = snake[0]
        new_head = (head[0] + direction[0], head[1] + direction[1])
        
        # 检查边界碰撞
        if (new_head[0] < 0 or new_head[0] >= self.board_size or
            new_head[1] < 0 or new_head[1] >= self.board_size):
            if player == 1:
                self.alive1 = False
            else:
                self.alive2 = False
            return
        
        # 检查自身碰撞
        if new_head in snake:
            if player == 1:
                self.alive1 = False
            else:
                self.alive2 = False
            return
        
        # 检查与对方蛇的碰撞
        other_snake = self.snake2 if player == 1 else self.snake1
        if new_head in other_snake:
            if player == 1:
                self.alive1 = False
            else:
                self.alive2 = False
            return
        
        # 移动蛇
        snake.insert(0, new_head)
        
        # 检查是否吃到食物
        if new_head in self.foods:
            self.foods.remove(new_head)
            self._generate_foods()
        else:
            snake.pop()
    
    def _generate_foods(self):
        """生成食物"""
        while len(self.foods) < self.food_count:
            x = random.randint(0, self.board_size - 1)
            y = random.randint(0, self.board_size - 1)
            pos = (x, y)
            
            # 确保食物不在蛇身上
            if pos not in self.snake1 and pos not in self.snake2 and pos not in self.foods:
                self.foods.append(pos)
    
    def _check_game_over(self) -> bool:
        # 如果有一方死亡，游戏结束
        if not self.alive1 or not self.alive2:
            return True
        # 如果有一方长度达到15，游戏结束
        if len(self.snake1) >= 15 or len(self.snake2) >= 15:
            return True
        return False
    
    def _reward_for_player(self, player: int, old_alive: bool) -> float:
        if player == 1:
            if old_alive and not self.alive1:
                return -1.0
            elif old_alive and not self.alive2:
                return 1.0
        else:
            if old_alive and not self.alive2:
                return -1.0
            elif old_alive and not self.alive1:
                return 1.0
        return 0.0 