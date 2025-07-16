import numpy as np
import random
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from ..base_game import BaseGame
import config


class BombGame(BaseGame):
    """
    双人泡泡堂（简化版炸弹人）游戏
    """

    # 定义板上元素
    EMPTY = 0
    WALL = 1
    DESTRUCTIBLE_BLOCK = 2
    BOMB_START_ID = 10 # 炸弹ID从10开始，用于区分不同炸弹的倒计时，如10: 3秒，11: 2秒，12: 1秒
    EXPLOSION = 28
    PLAYER1 = 30
    PLAYER2 = 31
    ITEM_BOMB_UP = 40 # 增加炸弹数量
    ITEM_RANGE_UP = 41 # 增加炸弹范围
    ITEM_SHIELD = 42 # 新增：护盾物品
    ITEM_SQUARE_RANGE_UP = 43 # 新增：方形爆炸范围物品
    WILL_EXPLOSION = 44
    
    # 护盾持续回合数
    SHIELD_DURATION = 30

    # 定义游戏奖励 (已包含您之前提供的奖励)
    REWARDS = {
        'win': 100.0,
        'lose': -100.0,
        'destroy_block': 1.0,
        'hit_enemy': 20.0, # 击中敌人奖励更高
        'collect_item': 5.0,
        'per_step_penalty': -0.1,
        'invalid_action_penalty': -0.5,
        'self_damage_penalty': -50.0, # 自杀惩罚
        'alive_step': 0.1,
        'draw': 0.0, # 添加平局奖励
        'death': -75.0 # 添加死亡惩罚，与self_damage_penalty区分
    }

    def __init__(self, board_size: int = 15, initial_bombs: int = 1, initial_range: int = 1, # 修改：initial_bombs 默认为2
                 destructible_block_density: float = 0.5, item_spawn_chance: float = 0.5): # 修改：item_spawn_chance 增加到0.5
        game_config = {
            'board_size': board_size,
            'initial_bombs': initial_bombs,
            'initial_range': initial_range,
            'destructible_block_density': destructible_block_density,
            'item_spawn_chance': item_spawn_chance,
            'timeout': config.GAME_CONFIGS['bomb']['timeout'],
            'max_moves': config.GAME_CONFIGS['bomb']['max_moves'] # 这将由GUI的config回退机制设置为一个非常高的值
        }
        
        # 最小改动：将 max_moves 的显式设置移到 super().__init__ 之前
        self.max_moves = game_config.get('max_moves', 100000) # 修改：设置一个非常高的默认max_moves

        super().__init__(game_config)

        self.board_size = board_size
        self.initial_bombs = initial_bombs
        self.initial_range = initial_range
        self.destructible_block_density = destructible_block_density
        self.item_spawn_chance = item_spawn_chance

        # 将实际存储棋盘数据的变量改为 _board
        self._board: np.ndarray = np.zeros((board_size, board_size), dtype=int)
        self.player1_pos: Tuple[int, int] = (0, 0)
        self.player2_pos: Tuple[int, int] = (0, 0)
        self.player1_bombs_max: int = initial_bombs # 玩家1最大炸弹数
        self.player2_bombs_max: int = initial_bombs # 玩家2最大炸弹数
        self.player1_range: int = initial_range # 玩家1炸弹范围
        self.player2_range: int = initial_range # 玩家2炸弹范围
        self.player1_current_bombs: int = 0 # 玩家1当前放置的炸弹数
        self.player2_current_bombs: int = 0 # 玩家2当前放置的炸弹数

        # 新增：护盾状态和计时器
        self.player1_shield_active: bool = False
        self.player2_shield_active: bool = False
        self.player1_shield_timer: int = 0
        self.player2_shield_timer: int = 0

        # 新增：爆炸范围类型 ('cross' 为十字，'square' 为方形穿透)
        self.player1_range_type: str = 'cross'
        self.player2_range_type: str = 'cross'

        # 添加：玩家分数
        self.player1_score: int = 0
        self.player2_score: int = 0

        self.bombs: List[Dict[str, Any]] = [] # 存储炸弹信息: {'pos': (x,y), 'timer': N, 'range': R, 'owner': P, 'range_type': 'cross'/'square'}
        self.explosions: List[Dict[str, Any]] = [] # 存储爆炸信息: {'pos': (x,y), 'timer': N}
        self.items: List[Dict[str, Any]] = [] # 修改：改为存储包含位置和类型的字典

        self.alive1: bool = True
        self.alive2: bool = True
        self.winner: Optional[int] = None # 1代表玩家1，2代表玩家2，0代表平局

        self._reset_game()

    def reset(self) -> Dict[str, Any]:
        """
        重置游戏到初始状态。
        返回初始观察。
        """
        self._reset_game()
        return self.get_state()

    def _reset_game(self):
        """内部重置逻辑，供子类实现。"""
        super()._reset_game()
        # 将实际存储棋盘数据的变量改为 _board
        self._board = np.zeros((self.board_size, self.board_size), dtype=int)
        # 玩家初始位置调整为角落，并留出空间
        self.player1_pos = (1, 1)
        self.player2_pos = (self.board_size - 2, self.board_size - 2)
        self.player1_bombs_max = self.initial_bombs
        self.player2_bombs_max = self.initial_bombs
        self.player1_range = self.initial_range
        self.player2_range = self.initial_range
        self.player1_current_bombs = 0
        self.player2_current_bombs = 0

        self.player1_shield_active = False
        self.player2_shield_active = False
        self.player1_shield_timer = 0
        self.player2_shield_timer = 0
        self.player1_range_type = 'cross'
        self.player2_range_type = 'cross'

        # 添加：重置分数
        self.player1_score = 0
        self.player2_score = 0

        self.bombs = []
        self.explosions = []
        self.items = []

        self.alive1 = True
        self.alive2 = True
        self.winner = None

        self._generate_board()
        self._place_initial_elements()

    def _generate_board(self):
        """生成游戏板"""
        # 边缘放置不可破坏的墙
        self._board[0, :] = self.WALL
        self._board[self.board_size - 1, :] = self.WALL
        self._board[:, 0] = self.WALL
        self._board[:, self.board_size - 1] = self.WALL

        # 内部放置不可破坏的墙（每隔一个位置）
        for r in range(2, self.board_size - 2, 2):
            for c in range(2, self.board_size - 2, 2):
                self._board[r, c] = self.WALL

        # 放置可破坏的方块
        for r in range(1, self.board_size - 1):
            for c in range(1, self.board_size - 1):
                if self._board[r, c] == self.EMPTY:
                    if random.random() < self.destructible_block_density:
                        self._board[r, c] = self.DESTRUCTIBLE_BLOCK
        
        # 清除玩家初始位置周围的方块，确保玩家有活动空间
        initial_clear_radius = 2 # 清除玩家周围2格的方块
        for player_pos in [self.player1_pos, self.player2_pos]:
            for r_offset in range(-initial_clear_radius, initial_clear_radius + 1):
                for c_offset in range(-initial_clear_radius, initial_clear_radius + 1):
                    clear_pos = (player_pos[0] + r_offset, player_pos[1] + c_offset)
                    if self._is_valid_position(clear_pos):
                        self._board[clear_pos] = self.EMPTY


    def _place_initial_elements(self):
        """放置玩家"""
        self._board[self.player1_pos] = self.PLAYER1
        self._board[self.player2_pos] = self.PLAYER2

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """检查位置是否在板内"""
        r, c = pos
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    def get_valid_actions(self, player_id: int) -> List[Union[Tuple[int, int], Tuple[int, int, bool]]]:
        """获取指定玩家的有效动作 (上, 下, 左, 右, 放置炸弹, 不动)"""
        if player_id == 1:
            player_pos = self.player1_pos
            alive = self.alive1
            current_bombs = self.player1_current_bombs
            max_bombs = self.player1_bombs_max
        else:
            player_pos = self.player2_pos
            alive = self.alive2
            current_bombs = self.player2_current_bombs
            max_bombs = self.player2_bombs_max

        if not alive:
            return [(0, 0)] # 死亡玩家只能执行无操作

        valid_actions = []
        # 移动动作: (dr, dc)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 上，下，左，右
        
        for dr, dc in moves:
            new_pos = (player_pos[0] + dr, player_pos[1] + dc)
            if self._is_valid_position(new_pos):
                target_cell = self._board[new_pos] # 使用 _board
                # 玩家可以移动到空地或物品上，但不能穿过墙、方块或炸弹
                # 炸弹的位置判断需要考虑炸弹列表，而不是board上的BOMB_START_ID
                is_bomb_at_target = False
                for bomb in self.bombs:
                    if bomb['pos'] == new_pos:
                        is_bomb_at_target = True
                        break

                if (target_cell == self.EMPTY or target_cell >= self.ITEM_BOMB_UP) and not is_bomb_at_target:
                    valid_actions.append((dr, dc))
        
        # 放置炸弹动作: (0, 0, True) 表示在当前位置放置炸弹
        if current_bombs < max_bombs:
            # 检查当前位置是否已有炸弹
            for bomb in self.bombs:
                if bomb['pos'] == player_pos:
                    can_place_bomb = False
                    break
            else: # 如果循环没有被break，说明没有炸弹
                can_place_bomb = True
            
            if can_place_bomb:
                valid_actions.append((0, 0, True)) # 特殊动作，表示放置炸弹
        
        # 不动动作
        valid_actions.append((0, 0)) # (0,0) 表示移动到原地，即不动

        return valid_actions

    def _move_player(self, player_id: int, action: Tuple[int, int]):
        """移动玩家"""
        if player_id == 1:
            current_pos = self.player1_pos
            alive = self.alive1
        else:
            current_pos = self.player2_pos
            alive = self.alive2
        
        if not alive:
            return

        dr, dc = action
        new_pos = (current_pos[0] + dr, current_pos[1] + dc)

        if not self._is_valid_position(new_pos):
            return

        target_cell = self._board[new_pos] # 使用 _board
        
        # 检查是否可以移动到目标位置
        # 可以移动到空地或物品上，但不能穿过墙、方块
        is_movable = (target_cell == self.EMPTY or target_cell >= self.ITEM_BOMB_UP) 
        
        # 检查新位置是否有炸弹
        is_bomb_at_new_pos = False
        for bomb in self.bombs:
            if bomb['pos'] == new_pos:
                is_bomb_at_new_pos = True
                break

        if is_movable and not is_bomb_at_new_pos:
            # 收集物品
            # 查找是否存在位于新位置的物品
            item_to_collect = next((item for item in self.items if item['pos'] == new_pos), None)
            if item_to_collect:
                self._collect_item(player_id, item_to_collect['type'])
                # 从物品列表中移除已收集的物品
                self.items = [item for item in self.items if item['pos'] != new_pos]
                # 物品被收集后，格子变为空，但如果格子上有爆炸，爆炸优先显示
                if self._board[new_pos] != self.EXPLOSION: # 使用 _board
                    self._board[new_pos] = self.EMPTY # 使用 _board

            if player_id == 1:
                self.player1_pos = new_pos
            else:
                self.player2_pos = new_pos

    def _place_bomb(self, player_id: int):
        """玩家放置炸弹"""
        if player_id == 1:
            player_pos = self.player1_pos
            current_bombs = self.player1_current_bombs
            max_bombs = self.player1_bombs_max
            bomb_range = self.player1_range
            bomb_range_type = self.player1_range_type
        else:
            player_pos = self.player2_pos
            current_bombs = self.player2_current_bombs
            max_bombs = self.player2_bombs_max
            bomb_range = self.player2_range
            bomb_range_type = self.player2_range_type

        if current_bombs < max_bombs:
            # 检查当前位置是否已有炸弹
            for bomb in self.bombs:
                if bomb['pos'] == player_pos:
                    return # 已经有炸弹了，不能再放
            
            # 将炸弹计时器设置为 30 回合，以实现 3 秒爆炸 (10 FPS * 3 秒 = 30 回合)
            self.bombs.append({'pos': player_pos, 'timer': 15, 'range': bomb_range, 'owner': player_id, 'range_type': bomb_range_type}) # 修改：计时器改回30以实现3秒爆炸
            # 修改：放置炸弹后，立即更新棋盘以显示炸弹。
            # 这修复了炸弹直到下一个游戏计时才出现的问题。
            self._board[player_pos] = self.BOMB_START_ID + 15 
            if player_id == 1:
                self.player1_current_bombs += 1
            else:
                self.player2_current_bombs += 1

    def _update_bombs(self):
        """更新所有炸弹的倒计时，并处理爆炸"""
        bombs_to_explode = []
        new_bombs = []
        for bomb in self.bombs:
            bomb['timer'] -= 1
            if bomb['timer'] <= 0:
                bombs_to_explode.append(bomb)
            else:
                # 更新板上炸弹的显示ID
                self._board[bomb['pos']] = self.BOMB_START_ID + bomb['timer'] # 使用 _board
                new_bombs.append(bomb)
        self.bombs = new_bombs

        for bomb in bombs_to_explode:
            self._explode_bomb(bomb)

    def _explode_bomb(self, bomb: Dict[str, Any]):
        """处理单个炸弹爆炸"""
        bomb_pos = bomb['pos']
        bomb_range = bomb['range']
        bomb_owner = bomb['owner']
        bomb_range_type = bomb['range_type'] # 获取炸弹的范围类型
        
        # 释放炸弹计数
        if bomb_owner == 1:
            self.player1_current_bombs -= 1
        else:
            self.player2_current_bombs -= 1

        # 清除炸弹在板上的显示 (如果该位置仍然是炸弹，而不是被其他东西覆盖)
        if self._is_valid_position(bomb_pos) and self._board[bomb_pos] >= self.BOMB_START_ID and self._board[bomb_pos] < self.EXPLOSION: # 使用 _board
            self._board[bomb_pos] = self.EMPTY # 使用 _board

        explosion_cells = set()
        explosion_cells.add(bomb_pos) # 炸弹中心

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)] # 右，左，下，上
        
        if bomb_range_type == 'cross':
            for dr, dc in directions:
                for i in range(1, bomb_range + 1):
                    exp_pos = (bomb_pos[0] + dr * i , bomb_pos[1] + dc * i)
                    if not self._is_valid_position(exp_pos):
                        break # 超出边界

                    cell_type = self._board[exp_pos] # 使用 _board
                    
                    # 检查这个位置是否有不可破坏的墙
                    if cell_type == self.WALL:
                        break # 遇到不可破坏的墙，爆炸停止

                    explosion_cells.add(exp_pos)
                    

                    if cell_type == self.DESTRUCTIBLE_BLOCK:
                        # 添加：销毁方块时增加分数
                        if bomb_owner == 1:
                            self.player1_score += 1
                        else:
                            self.player2_score += 1

                        # 如果是十字形爆炸，遇到可破坏方块就停止传播
                        if bomb_range_type == 'cross':
                            self._board[exp_pos] = self.EXPLOSION # 使用 _board
                            # 有几率生成物品
                            if random.random() < self.item_spawn_chance: # 修改：使用更新后的物品生成几率
                                item_type = random.choice([self.ITEM_BOMB_UP, self.ITEM_RANGE_UP, self.ITEM_SHIELD, self.ITEM_SQUARE_RANGE_UP])
                                self.items.append({'pos': exp_pos, 'type': item_type}) # 修改：将物品存储为字典
                                # 不要在这里将_board[exp_pos]设置为物品类型，让_update_explosions在爆炸计时器之后处理
                            self._board[exp_pos] = self.EXPLOSION # 暂时显示爆炸 # 使用 _board
                            break # 遇到可破坏的方块，爆炸停止传播
                        elif bomb_range_type == 'square':
                            # 方形爆炸：破坏方块，但继续传播
                            self._board[exp_pos] = self.EXPLOSION # 使用 _board
                            if random.random() < self.item_spawn_chance: # 修改：使用更新后的物品生成几率
                                item_type = random.choice([self.ITEM_BOMB_UP, self.ITEM_RANGE_UP, self.ITEM_SHIELD, self.ITEM_SQUARE_RANGE_UP])
                                self.items.append({'pos': exp_pos, 'type': item_type}) # 修改：将物品存储为字典
                                # 不要在这里将_board[exp_pos]设置为物品类型，让_update_explosions在爆炸计时器之后处理
                            self._board[exp_pos] = self.EXPLOSION # 使用 _board
                            # 不break，继续传播

                    # 如果遇到其他炸弹，也引爆
                    # 遍历bombs列表，而不是board上的值，因为board上的值是炸弹倒计时
                    for other_bomb in self.bombs:
                        if other_bomb['pos'] == exp_pos and other_bomb['timer'] > 1: # 避免无限递归，只引爆未立即爆炸的炸弹
                            other_bomb['timer'] = 1 # 将其设置为立即爆炸
                            break
                    
                    # 如果是空地或者物品，继续传播，并标记为爆炸
                    # 我们不需要在这里检查ITEM_BOMB_UP等，因为物品现在是单独处理的
                    # 并且将由GUI在顶部绘制。
                    if cell_type == self.EMPTY or cell_type == self.PLAYER1 or cell_type == self.PLAYER2:
                        self._board[exp_pos] = self.EXPLOSION # 暂时显示爆炸 # 使用 _board
        
        elif bomb_range_type == 'square':
            for dx in range(-bomb_range, bomb_range + 1):
                for dy in range(-bomb_range, bomb_range + 1):
                    exp_pos = (bomb_pos[0] + dx , bomb_pos[1] + dy)
                    if not self._is_valid_position(exp_pos):
                        continue # 超出边界

                    cell_type = self._board[exp_pos] # 使用 _board
                    
                    # # 检查这个位置是否有不可破坏的墙
                    # if cell_type == self.WALL:
                    #     break # 遇到不可破坏的墙，爆炸停止

                    explosion_cells.add(exp_pos)
                    

                    if cell_type == self.DESTRUCTIBLE_BLOCK:
                        # 添加：销毁方块时增加分数
                        if bomb_owner == 1:
                            self.player1_score += 1
                        else:
                            self.player2_score += 1

                        # 如果是十字形爆炸，遇到可破坏方块就停止传播
                        if bomb_range_type == 'cross':
                            self._board[exp_pos] = self.EXPLOSION # 使用 _board
                            # 有几率生成物品
                            if random.random() < self.item_spawn_chance: # 修改：使用更新后的物品生成几率
                                item_type = random.choice([self.ITEM_BOMB_UP, self.ITEM_RANGE_UP, self.ITEM_SHIELD, self.ITEM_SQUARE_RANGE_UP])
                                self.items.append({'pos': exp_pos, 'type': item_type}) # 修改：将物品存储为字典
                                # 不要在这里将_board[exp_pos]设置为物品类型，让_update_explosions在爆炸计时器之后处理
                            self._board[exp_pos] = self.EXPLOSION # 暂时显示爆炸 # 使用 _board
                            break # 遇到可破坏的方块，爆炸停止传播
                        elif bomb_range_type == 'square':
                            # 方形爆炸：破坏方块，但继续传播
                            self._board[exp_pos] = self.EXPLOSION # 使用 _board
                            if random.random() < self.item_spawn_chance: # 修改：使用更新后的物品生成几率
                                item_type = random.choice([self.ITEM_BOMB_UP, self.ITEM_RANGE_UP, self.ITEM_SHIELD, self.ITEM_SQUARE_RANGE_UP])
                                self.items.append({'pos': exp_pos, 'type': item_type}) # 修改：将物品存储为字典
                                # 不要在这里将_board[exp_pos]设置为物品类型，让_update_explosions在爆炸计时器之后处理
                            self._board[exp_pos] = self.EXPLOSION # 使用 _board
                            # 不break，继续传播

                    # 如果遇到其他炸弹，也引爆
                    # 遍历bombs列表，而不是board上的值，因为board上的值是炸弹倒计时
                    for other_bomb in self.bombs:
                        if other_bomb['pos'] == exp_pos and other_bomb['timer'] > 1: # 避免无限递归，只引爆未立即爆炸的炸弹
                            other_bomb['timer'] = 1 # 将其设置为立即爆炸
                            break
                    
                    # 如果是空地或者物品，继续传播，并标记为爆炸
                    # 我们不需要在这里检查ITEM_BOMB_UP等，因为物品现在是单独处理的
                    # 并且将由GUI在顶部绘制。
                    if cell_type == self.EMPTY or cell_type == self.PLAYER1 or cell_type == self.PLAYER2:
                        self._board[exp_pos] = self.EXPLOSION # 暂时显示爆炸 # 使用 _board
                    


        # 将爆炸区域加入爆炸列表，用于定时清除
        for exp_pos in explosion_cells:
            self.explosions.append({'pos': exp_pos, 'timer': 5}) # 修改：爆炸持续5帧

        self._apply_explosion_effect(explosion_cells)

    def _apply_explosion_effect(self, explosion_cells: set):
        """应用爆炸效果到玩家"""
        # 检查玩家1
        if self.alive1 and self.player1_pos in explosion_cells:
            if self.player1_shield_active:
                self.player1_shield_active = False # 护盾被消耗
                self.player1_shield_timer = 0
            else:
                self.alive1 = False
        # 检查玩家2
        if self.alive2 and self.player2_pos in explosion_cells:
            if self.player2_shield_active:
                self.player2_shield_active = False # 护盾被消耗
                self.player2_shield_timer = 0
            else:
                self.alive2 = False

    def _update_explosions(self):
        """更新爆炸效果，清除过期的爆炸"""
        new_explosions = []
        for exp in self.explosions:
            exp['timer'] -= 1
            if exp['timer'] <= 0:
                # 爆炸结束，恢复为EMPTY，除非上面有其他炸弹或玩家
                pos = exp['pos']
                is_player_at_pos = (pos == self.player1_pos and self.alive1) or \
                                   (pos == self.player2_pos and self.alive2)
                is_bomb_at_pos = any(b['pos'] == pos for b in self.bombs)
                
                # 查找是否存在位于此位置的物品
                item_at_pos = next((item for item in self.items if item['pos'] == pos), None)

                if not is_player_at_pos and not is_bomb_at_pos:
                    if item_at_pos:
                        self._board[pos] = item_at_pos['type'] # 恢复为物品类型
                    else:
                        self._board[pos] = self.EMPTY # 恢复为空
            else:
                new_explosions.append(exp)
        self.explosions = new_explosions

    def _collect_item(self, player_id: int, item_type: int):
        """玩家收集物品"""
        if item_type == self.ITEM_BOMB_UP:
            if player_id == 1:
                self.player1_bombs_max += 1
            else:
                self.player2_bombs_max += 1
        elif item_type == self.ITEM_RANGE_UP:
            if player_id == 1:
                self.player1_range += 1
            else:
                self.player2_range += 1
        elif item_type == self.ITEM_SHIELD:
            if player_id == 1:
                self.player1_shield_active = True
                self.player1_shield_timer = self.SHIELD_DURATION
            else:
                self.player2_shield_active = True
                self.player2_shield_timer = self.SHIELD_DURATION
        elif item_type == self.ITEM_SQUARE_RANGE_UP:
            if player_id == 1:
                self.player1_range_type = 'square'
            else:
                self.player2_range_type = 'square'


    def step(self, action1: Union[Tuple[int, int], Tuple[int, int, bool]], action2: Union[Tuple[int, int], Tuple[int, int, bool], None] = None) -> Tuple[float, float, bool, bool]: # 修改返回类型提示，只返回4个值
        """
        执行一步游戏。
        动作可以是 (dr, dc) 代表移动，或者 (0, 0, True) 代表放置炸弹。
        """
        self.current_moves += 1
        
        # 重置奖励
        reward1 = 0.0
        reward2 = 0.0
        
        old_alive1 = self.alive1
        old_alive2 = self.alive2

        # 更新护盾计时器
        if self.player1_shield_active:
            self.player1_shield_timer -= 1
            if self.player1_shield_timer <= 0:
                self.player1_shield_active = False
        if self.player2_shield_active:
            self.player2_shield_timer -= 1
            if self.player2_shield_timer <= 0:
                self.player2_shield_active = False

        # 玩家1动作
        if action1 is not None and self.alive1:
            if len(action1) == 3 and action1[2]: # 放置炸弹动作
                self._place_bomb(1)
            else: # 移动动作
                self._move_player(1, (action1[0], action1[1]))

        # 玩家2动作
        if action2 is not None and self.alive2:
            if len(action2) == 3 and action2[2]: # 放置炸弹动作
                self._place_bomb(2)
            else: # 移动动作
                self._move_player(2, (action2[0], action2[1]))
        
        # 更新炸弹和爆炸
        self._update_bombs()
        self._update_explosions()

        # 检查游戏结束
        terminal = self.is_terminal()
        if terminal:
            self.winner = self.get_winner()
            if self.winner == 1:
                reward1 += config.REWARDS['win']
                reward2 += config.REWARDS['lose']
            elif self.winner == 2:
                reward1 += config.REWARDS['lose']
                reward2 += config.REWARDS['win']
            else: # 平局或双方都输
                reward1 += config.REWARDS['draw']
                reward2 += config.REWARDS['draw']
        else:
            # 存活奖励
            if self.alive1 and old_alive1:
                reward1 += config.REWARDS['alive_step']
            if self.alive2 and old_alive2:
                reward2 += config.REWARDS['alive_step']

            # 死亡惩罚
            if not self.alive1 and old_alive1:
                reward1 += config.REWARDS['death']
            if not self.alive2 and old_alive2:
                reward2 += config.REWARDS['death']
            
            # 超时惩罚 (这现在由GUI的计时器处理，但为了保持一致性保留逻辑)
            if self.current_moves >= self.max_moves: # 这个条件现在主要由GUI的计时器触发
                if self.alive1 and self.alive2: # 双方都存活，平局
                    reward1 += config.REWARDS['draw']
                    reward2 += config.REWARDS['draw']
                elif self.alive1: # 玩家1存活，玩家2死亡
                    reward1 += config.REWARDS['win']
                    reward2 += config.REWARDS['lose']
                elif self.alive2: # 玩家2存活，玩家1死亡
                    reward1 += config.REWARDS['lose']
                    reward2 += config.REWARDS['win']

        # 更新板上的玩家位置显示 (覆盖可能出现的EMPTY或EXPLOSION)
        # 确保玩家位置的优先级高于其他元素显示 (除了炸弹，炸弹会显示倒计时)
        # 这一步非常重要，确保玩家始终能被看到，且不会被爆炸或物品覆盖
        # 先清除旧的玩家显示，再重新绘制，避免残留
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self._board[r,c] == self.PLAYER1 or self._board[r,c] == self.PLAYER2: # 使用 _board
                    # 如果该位置是玩家，且该玩家已经死亡，则清除
                    if (r,c) == self.player1_pos and not self.alive1:
                        self._board[r,c] = self.EMPTY # 使用 _board
                    elif (r,c) == self.player2_pos and not self.alive2:
                        self._board[r,c] = self.EMPTY # 使用 _board
                    # 如果玩家还活着但位置被爆炸覆盖，暂时不清除
                    elif self._board[r,c] == self.EXPLOSION: # 使用 _board
                         pass # 爆炸优先显示
                    else:
                        # 如果玩家还活着，且该位置没有爆炸，则清除旧的玩家显示
                        if (r,c) != self.player1_pos and (r,c) != self.player2_pos:
                             self._board[r,c] = self.EMPTY # 使用 _board

        # 重新放置玩家
        if self.alive1:
            self._board[self.player1_pos] = self.PLAYER1 # 使用 _board
        if self.alive2:
            self._board[self.player2_pos] = self.PLAYER2 # 使用 _board

        # 确保炸弹显示不受玩家移动影响
        for bomb in self.bombs:
            self._board[bomb['pos']] = self.BOMB_START_ID + bomb['timer'] # 使用 _board
        
        # 确保爆炸显示不受玩家移动影响
        for exp in self.explosions:
             self._board[exp['pos']] = self.EXPLOSION # 使用 _board
        
        # 确保物品显示 (仅当位置不是玩家、炸弹或爆炸时)
        for item in self.items: # 遍历物品字典
             item_pos = item['pos']
             # 仅当单元格当前不是玩家、炸弹或爆炸时才绘制物品
             if self._is_valid_position(item_pos) and \
                self._board[item_pos] not in [self.PLAYER1, self.PLAYER2, self.EXPLOSION] and \
                self._board[item_pos] < self.BOMB_START_ID: # 确保不是炸弹
                self._board[item_pos] = item['type'] # 将棋盘设置为物品类型


        # observation = self.get_state() # 移除
        # info = self.get_game_info() # 移除
        return reward1, reward2, terminal, terminal # 修改返回语句，只返回4个值

    @property
    def board(self):
        """
        返回当前棋盘状态。
        """
        return self._board.copy() # 返回 _board 的副本

    def get_state(self) -> Dict[str, Any]:
        """获取当前游戏状态"""
        state = {
            'board': self._board.copy(), # 返回 _board 的副本
            'player1_pos': self.player1_pos,
            'player2_pos': self.player2_pos,
            'alive1': self.alive1,
            'alive2': self.alive2,
            'player1_bombs_max': self.player1_bombs_max,
            'player2_bombs_max': self.player2_bombs_max,
            'player1_range': self.player1_range,
            'player2_range': self.player2_range,
            'player1_current_bombs': self.player1_current_bombs,
            'player2_current_bombs': self.player2_current_bombs,
            'player1_shield_active': self.player1_shield_active, # 新增
            'player2_shield_active': self.player2_shield_active, # 新增
            'player1_shield_timer': self.player1_shield_timer, # 新增
            'player2_shield_timer': self.player2_shield_timer, # 新增
            'player1_range_type': self.player1_range_type, # 新增
            'player2_range_type': self.player2_range_type, # 新增
            'player1_score': self.player1_score, # 添加：玩家1分数
            'player2_score': self.player2_score, # 添加：玩家2分数
            'bombs': [{'pos': b['pos'], 'timer': b['timer'], 'range': b['range'], 'owner': b['owner'], 'range_type': b['range_type']} for b in self.bombs], # 包含range_type
            'explosions': [{'pos': e['pos'], 'timer': e['timer']} for e in self.explosions],
            'items': [item.copy() for item in self.items], # 修改：转换为字典列表以便克隆
            'current_moves': self.current_moves,
            'max_moves': self.max_moves,
            'winner': self.winner
        }
        return state

    def is_terminal(self) -> bool:
        """检查游戏是否结束"""
        # 任何一方死亡，或者达到最大步数
        return not (self.alive1 and self.alive2) or self.current_moves >= self.max_moves
    
    def get_winner(self) -> Optional[int]:
        """获取赢家ID (1, 2, 或 0 代表平局/进行中)"""
        # 如果双方玩家都死亡，则平局
        if not self.alive1 and not self.alive2:
            if self.player1_score > self.player2_score:
                return 1
            elif self.player1_score < self.player2_score:
                return 2
            elif self.player1_score == self.player2_score:
                return 0
        # 如果玩家1死亡，玩家2获胜
        elif not self.alive1:
            return 2 
        # 如果玩家2死亡，玩家1获胜
        elif not self.alive2:
            return 1 
        # 如果达到最大步数并且双方都存活，则比较分数
        elif self.current_moves >= self.max_moves: # 这个条件现在主要由GUI的计时器触发
            if self.player1_score > self.player2_score:
                return 1 # 玩家1按分数获胜
            elif self.player2_score > self.player1_score:
                return 2 # 玩家2按分数获胜
            else:
                return 0 # 按分数平局
        return None # 游戏仍在进行中

    def get_game_info(self) -> Dict[str, Any]:
        """获取游戏信息"""
        info = {
            'board_size': self.board_size,
            'player1_pos': self.player1_pos,
            'player2_pos': self.player2_pos,
            'alive1': self.alive1,
            'alive2': self.alive2,
            'player1_bombs_max': self.player1_bombs_max,
            'player2_bombs_max': self.player2_bombs_max,
            'player1_range': self.player1_range,
            'player2_range': self.player2_range,
            'player1_current_bombs': self.player1_current_bombs,
            'player2_current_bombs': self.player2_current_bombs,
            'player1_shield_active': self.player1_shield_active, # 新增
            'player2_shield_active': self.player2_shield_active, # 新增
            'player1_shield_timer': self.player1_shield_timer, # 新增
            'player2_shield_timer': self.player2_shield_timer, # 新增
            'player1_range_type': self.player1_range_type, # 新增
            'player2_range_type': self.player2_range_type, # 新增
            'player1_score': self.player1_score, # 添加：玩家1分数
            'player2_score': self.player2_score, # 添加：玩家2分数
            'current_moves': self.current_moves,
            'max_moves': self.max_moves,
            'winner': self.winner
        }
        return info

    def clone(self) -> 'BombGame':
        """克隆当前游戏状态"""
        cloned_game = BombGame(self.board_size, self.initial_bombs, self.initial_range,
                                self.destructible_block_density, self.item_spawn_chance)
        
        cloned_game._board = self._board.copy() # 克隆 _board
        cloned_game.player1_pos = self.player1_pos
        cloned_game.player2_pos = self.player2_pos
        cloned_game.player1_bombs_max = self.player1_bombs_max
        cloned_game.player2_bombs_max = self.player2_bombs_max
        cloned_game.player1_range = self.player1_range
        cloned_game.player2_range = self.player2_range
        cloned_game.player1_current_bombs = self.player1_current_bombs
        cloned_game.player2_current_bombs = self.player2_current_bombs
        cloned_game.player1_shield_active = self.player1_shield_active # 克隆护盾状态
        cloned_game.player2_shield_active = self.player2_shield_active # 克隆护盾状态
        cloned_game.player1_shield_timer = self.player1_shield_timer # 克隆护盾计时器
        cloned_game.player2_shield_timer = self.player2_shield_timer # 克隆护盾计时器
        cloned_game.player1_range_type = self.player1_range_type # 克隆爆炸范围类型
        cloned_game.player2_range_type = self.player2_range_type # 克隆爆炸范围类型
        cloned_game.player1_score = self.player1_score # 添加：克隆分数
        cloned_game.player2_score = self.player2_score # 添加：克隆分数

        cloned_game.bombs = [b.copy() for b in self.bombs]
        cloned_game.explosions = [e.copy() for e in self.explosions]
        cloned_game.items = [item.copy() for item in self.items] # 修改：深拷贝物品
        cloned_game.alive1 = self.alive1
        cloned_game.alive2 = self.alive2
        cloned_game.winner = self.winner
        cloned_game.current_moves = self.current_moves
        cloned_game.max_moves = self.max_moves # 确保这里也克隆了 max_moves
        
        return cloned_game

    def get_board_state_for_player(self, player_id: int) -> np.ndarray:
        """
        为指定玩家生成板状态的观察。
        可以根据需要进行简化或添加更多信息。
        这里返回完整的板，AI可以根据此进行决策。
        """
        return self._board.copy() # 返回 _board 的副本

    def get_player_info(self, player_id: int) -> Dict[str, Any]:
        """获取指定玩家的详细信息"""
        if player_id == 1:
            return {
                'pos': self.player1_pos,
                'alive': self.alive1,
                'bombs_max': self.player1_bombs_max,
                'current_bombs': self.player1_current_bombs,
                'range': self.player1_range,
                'shield_active': self.player1_shield_active, # 新增
                'shield_timer': self.player1_shield_timer, # 新增
                'range_type': self.player1_range_type, # 新增
                'score': self.player1_score # 添加：玩家分数
            }
        elif player_id == 2:
            return {
                'pos': self.player2_pos,
                'alive': self.alive2,
                'bombs_max': self.player2_bombs_max,
                'current_bombs': self.player2_current_bombs,
                'range': self.player2_range,
                'shield_active': self.player2_shield_active, # 新增
                'shield_timer': self.player2_shield_timer, # 新增
                'range_type': self.player2_range_type, # 新增
                'score': self.player2_score # 添加：玩家分数
            }
        return {}

    def get_other_player_info(self, player_id: int) -> Dict[str, Any]:
        """获取对手玩家的详细信息"""
        if player_id == 1:
            return self.get_player_info(2)
        elif player_id == 2:
            return self.get_player_info(1)
        return {}

    def render(self, mode: str = 'human') -> np.ndarray:
        """
        渲染游戏画面。
        对于泡泡堂游戏，直接返回当前棋盘状态即可，GUI会根据此进行绘制。
        """
        return self._board.copy() # 返回 _board 的副本

