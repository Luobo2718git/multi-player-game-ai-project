import random
import numpy as np
from agents.base_agent import BaseAgent
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque
from games.bomb.bomb_game import BombGame # 导入BombGame，用于访问其常量

class BombAI(BaseAgent):
    """泡泡堂AI智能体"""
    
    def __init__(self, name="BombAI", player_id=2): # 默认为玩家2
        super().__init__(name, player_id)
        self._move_delay_counter = 0 # 控制移动频率的计数器
        self._move_delay_ticks = 2 # AI每2个游戏帧移动一次
        self.escape_path = []

    def get_action(self, observation: Dict[str, Any], env: Any) -> Union[Tuple[int, int], Tuple[int, int, bool]]:
        """
        获取动作。
        observation 包含游戏的完整状态。
        env 提供了获取有效动作和游戏信息的方法。
        """
        # 增加移动延迟计数器
        # self._move_delay_counter =  # 随机增加1-3个计数器值，模拟AI思考时间

        # # 如果未到移动时间，返回不操作动作
        self._move_delay_counter += 1
        # print("count:", self._move_delay_counter % self._move_delay_ticks)
        if self._move_delay_counter % self._move_delay_ticks != 0:
            
            return (0, 0) # 保持原地不动

        # 执行动作后重置计数器
        self._move_delay_counter = 0

        valid_actions = env.get_valid_actions(self.player_id)
        if not valid_actions:
            return (0, 0) # 如果没有有效动作，则原地不动

        game_state = observation
        board = game_state['board']
        player_info = env.game.get_player_info(self.player_id)
        other_player_info = env.game.get_other_player_info(self.player_id)
        current_pos = player_info['pos']
        current_bombs = player_info['current_bombs']
        max_bombs = player_info['bombs_max']
        bomb_range = player_info['range']
        player_shield_active = player_info['shield_active']
        player_shield_timer = player_info['shield_timer']
        player_range_type = player_info['range_type']
        all_bombs = env.game.bombs


        # 2. 躲避炸弹: 如果当前位置或即将移动的位置不安全，寻找逃生路径
        self.escape_path = self._find_escape_path(current_pos, game_state)
        if self.escape_path and len(self.escape_path) > 0:
            # 取逃生路径的第一个点，转为动作
            next_pos = self.escape_path[0]
            dr = next_pos[0] - current_pos[0]
            dc = next_pos[1] - current_pos[1]
            escape_action = (dr, dc)
            # 检查该动作是否在安全移动列表中
            # if escape_action in [a if len(a) == 2 else a[:2] for a in safe_moves]:
            return escape_action
        
        # 1. 安全优先: 避免当前位置和预期移动位置处于爆炸范围
        safe_moves = []
        for action in valid_actions:
            if len(action) == 3 and action[2]: # 放置炸弹动作 (0, 0, True)
                # 在允许放置炸弹之前，检查当前位置是否安全（不受其他炸弹/爆炸影响）
                if self._is_position_safe(current_pos, game_state):
                    safe_moves.append(action)
                continue # 继续检查下一个有效动作
            
            new_pos = (current_pos[0] + action[0], current_pos[1] + action[1])
            if self._is_position_safe(new_pos, game_state):
                safe_moves.append(action)
        
        if not safe_moves:
            return (0, 0) # 如果没有安全移动，则原地不动 (或选择一个尽可能安全的动作，这里简化为不动)
        
        

        # 3. 收集物品 (优先护盾和方形爆炸)
        items = game_state['items']
        if items:
            # 优先寻找护盾或方形爆炸物品
            priority_items_pos = [] # 仅存储用于寻路的位置
            other_items_pos = []    # 仅存储用于寻路的位置
            
            for item_info in items: # item_info 现在是正确的字典
                item_pos = item_info['pos'] # 提取位置元组
                item_type = item_info['type'] # 提取物品类型
                
                if item_type == BombGame.ITEM_SHIELD and not player_shield_active: # 如果没有护盾，优先收集护盾
                    priority_items_pos.append(item_pos)
                elif item_type == BombGame.ITEM_SQUARE_RANGE_UP and player_range_type == 'cross': # 如果是十字形爆炸，优先收集方形爆炸
                    priority_items_pos.append(item_pos)
                else:
                    other_items_pos.append(item_pos)
            
            if priority_items_pos:
                best_item_action = self._get_action_towards_nearest_target(current_pos, priority_items_pos, safe_moves, board, game_state['bombs'])
                if best_item_action:
                    return best_item_action
            elif other_items_pos: # 如果没有优先级物品，收集其他物品
                best_item_action = self._get_action_towards_nearest_target(current_pos, other_items_pos, safe_moves, board, game_state['bombs'])
                if best_item_action:
                    return best_item_action


        # 4. 放置炸弹策略:
        #    - 如果可以放置炸弹且其爆炸范围能破坏方块或击中敌人
        #    - 并且玩家当前位置是安全的 (已通过 safe_moves 检查)
        bomb_desirability = 0 # 期望值，-1表示不期望或不可能放置
        if (0, 0, True) in safe_moves and current_bombs < max_bombs:
            # 模拟在当前位置放置炸弹的信息 (计时器为30，表示3秒后爆炸)
            # 这个模拟是为了检查如果它爆炸，潜在的目标是什么。
            potential_explosion_cells = self._get_explosion_coords(current_pos, bomb_range, board.shape[0], board, player_range_type)
            # print('potential num:', len(potential_explosion_cells))
            destructible_blocks_hit_count = 0
            hits_enemy = False
            other_player_pos = env.game.player1_pos if self.player_id == 2 else env.game.player2_pos
            
            for exp_pos in potential_explosion_cells:
                if env.game._is_valid_position(exp_pos):
                    cell_type = board[exp_pos]
                    # print("exp:", exp_pos, cell_type)
                    if cell_type == BombGame.DESTRUCTIBLE_BLOCK:
                        if player_range_type == 'cross':
                            destructible_blocks_hit_count += 1
                        else:
                            destructible_blocks_hit_count += 0.5
                    if exp_pos == other_player_pos:
                        hits_enemy = True
                # 如果已经找到目标，可以提前退出循环
                if  hits_enemy: # 修正: 使用已初始化的变量
                    break 
            
            if hits_enemy:
                bomb_desirability = 100 # 击中敌人优先级最高
            elif destructible_blocks_hit_count > 0:
                bomb_desirability = destructible_blocks_hit_count # 基于破坏方块数量
                if player_range_type == 'square':
                    bomb_desirability += 1 # 如果是方形炸弹，额外加分，鼓励使用

        # print(bomb_desirability)
        if bomb_desirability > 0:
            # 检查放置炸弹后是否有逃生路径
            simulated_bombs = game_state['bombs'] + [{
                'pos': current_pos,
                'range': bomb_range,
                'timer': 15,  
                'range_type': player_range_type
            }]
            escape_path = self._find_escape_path(current_pos, {**game_state, 'bombs': simulated_bombs})
            if escape_path and len(escape_path) > 0 and len(escape_path) < 6:
                return (0, 0, True) # 放置炸弹

        # 5. 追逐敌人或破坏方块 (如果安全)
        destructible_blocks = []
        for r in range(env.game.board_size):
            for c in range(env.game.board_size):
                if board[r, c] == BombGame.DESTRUCTIBLE_BLOCK:
                    destructible_blocks.append((r, c))
        
        if destructible_blocks:
            best_block_action = self._get_action_towards_nearest_target(current_pos, destructible_blocks, safe_moves, board, game_state['bombs'])
            if best_block_action:
                return best_block_action

        # 6. 随机移动 (在安全移动中选择)
        # 过滤掉放置炸弹的动作，因为我们已经处理了放置炸弹的逻辑
        movement_actions = [a for a in safe_moves if not (len(a) == 3 and a[2])]
        # movement_actions = safe_moves
        if movement_actions:
            # print("ALL:", movement_actions)
            other_player_pos = env.game.get_player_info(1)['pos']
            # print(other_player_pos)
            valid_list = []
            for act in movement_actions:
                if self._leads_towards_enemy(current_pos, act, other_player_pos):
                    valid_list.append(act)
            # print("valid:", valid_list)
            if valid_list != []:
                return random.choice(valid_list)
            else:
                return random.choice(movement_actions)        
        return (0, 0) # 实在没招了，原地不动
    
    def _leads_towards_enemy(self, current_pos: Tuple[int, int], action: Tuple[int, int], enemy_pos: Tuple[int, int]) -> bool:
        """检查一个动作是否导致AI向敌人移动"""
        next_pos = (current_pos[0] + action[0], current_pos[1] + action[1])
        # 简单判断：如果新位置与敌人的曼哈顿距离小于当前位置与敌人的曼哈顿距离
        current_dist = abs(current_pos[0] - enemy_pos[0]) + abs(current_pos[1] - enemy_pos[1])
        next_dist = abs(next_pos[0] - enemy_pos[0]) + abs(next_pos[1] - enemy_pos[1])
        return next_dist < current_dist


    def _is_position_safe(self, pos: Tuple[int, int], game_state: Dict[str, Any]) -> bool:
        """检查给定位置在下一帧是否安全 (没有爆炸)"""
        board_size = game_state['board'].shape[0]
        explosions = game_state['explosions']
        bombs = game_state['bombs']

        # 检查是否在板内
        if not (0 <= pos[0] < board_size and 0 <= pos[1] < board_size):
            return False

        # 检查当前位置是否有即将爆炸的炸弹或现有爆炸
        for bomb in bombs:
            if bomb['timer'] <= 10:
                #模拟爆炸范围
                explosion_coords = self._get_explosion_coords(bomb['pos'], bomb['range'], board_size, game_state['board'], bomb['range_type'])
                if pos in explosion_coords:
                    return False
        
        # 检查是否在当前爆炸范围内
        for exp in explosions:
            if exp['pos'] == pos: # 爆炸还在持续
                return False
        
        return True

    def _get_explosion_coords(self, bomb_pos: Tuple[int, int], bomb_range: int, board_size: int, board: np.ndarray, range_type: str) -> set:
        """计算炸弹的爆炸范围坐标"""
        explosion_cells = set()
        explosion_cells.add(bomb_pos)

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        if range_type == "cross":
            for dr, dc in directions:
                for i in range(1, bomb_range + 1):
                    exp_pos = (bomb_pos[0] + dr * i, bomb_pos[1] + dc * i)
                    if not (0 <= exp_pos[0] < board_size and 0 <= exp_pos[1] < board_size):
                        break

                    cell_type = board[exp_pos]
                    if cell_type == BombGame.WALL:
                        break
                    
                    explosion_cells.add(exp_pos)
                    
        elif range_type == 'square':
            for dx in range(-bomb_range, bomb_range + 1):
                for dy in range(-bomb_range, bomb_range + 1):
                    exp_pos = (bomb_pos[0] + dx, bomb_pos[1] + dy)
                    if not (0 <= exp_pos[0] < board_size and 0 <= exp_pos[1] < board_size):
                        break
                    
                    explosion_cells.add(exp_pos)

        return explosion_cells
    

    def _get_action_towards_nearest_target(self, start_pos: Tuple[int, int], targets: List[Tuple[int, int]], safe_moves: List[Union[Tuple[int, int], Tuple[int, int, bool]]], board: np.ndarray, bombs: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
        """使用BFS寻找前往最近目标的路径，并返回第一个动作"""
        q = deque([(start_pos, [])]) # (当前位置, 路径)
        visited = {start_pos}
        
        board_size = board.shape[0]

        while q:
            current_pos, path = q.popleft()

            if current_pos in targets:
                if len(path) > 0:
                    # 将第一个移动动作转换为 (dr, dc)
                    first_step_pos = path[0]
                    dr = first_step_pos[0] - start_pos[0]
                    dc = first_step_pos[1] - start_pos[1]
                    action_tuple = (dr, dc)
                    # 确保这个动作是安全的移动动作
                    if action_tuple in safe_moves: 
                        return action_tuple
                return (0, 0) # 如果目标是当前位置，或者路径为空，则原地不动

            # 遍历所有可能的移动方向 (不包括放置炸弹，因为这是寻路)
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 上，下，左，右

            for dr, dc in moves:
                next_pos = (current_pos[0] + dr, current_pos[1] + dc)

                if (0 <= next_pos[0] < board_size and 0 <= next_pos[1] < board_size and
                    next_pos not in visited):
                    
                    cell_type = board[next_pos]
                    # 不能移动到墙壁、可破坏方块、炸弹
                    is_obstacle = False
                    if cell_type == BombGame.WALL or cell_type == BombGame.DESTRUCTIBLE_BLOCK:
                        is_obstacle = True
                    
                    is_bomb_at_next_pos = False
                    for bomb_info in bombs:
                        if bomb_info['pos'] == next_pos:
                            is_bomb_at_next_pos = True
                            break
                    
                    if not is_obstacle and not is_bomb_at_next_pos:
                        visited.add(next_pos)
                        q.append((next_pos, path + [next_pos]))
        return None

    def _find_escape_path(self, start_pos: Tuple[int, int], game_state: Dict[str, Any]) -> Optional[List[Tuple[int, int]]]:
        """
        在放置炸弹后，寻找一条安全逃离路径。
        返回一条路径，或者None如果没有安全路径。
        """
        q = deque([(start_pos, [])]) # (当前位置, 路径)
        visited = {start_pos}
        
        board_size = game_state['board'].shape[0]
        board = game_state['board']
        bombs = game_state['bombs'] # 包含新放置的炸弹

        # 模拟下一帧的爆炸区域
        simulated_explosion_cells = set()
        for bomb in bombs:
            simulated_explosion_cells.update(self._get_explosion_coords(bomb['pos'], bomb['range'], board_size, board, bomb['range_type']))

        # BFS寻找安全路径
        while q:
            current_pos, path = q.popleft()

            # 如果当前位置不在模拟爆炸区域内，则找到安全点
            if current_pos not in simulated_explosion_cells:
                # print("path:", path)
                return path # 返回逃离路径

            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 上，下，左，右

            for dr, dc in moves:
                next_pos = (current_pos[0] + dr, current_pos[1] + dc)

                if (0 <= next_pos[0] < board_size and 0 <= next_pos[1] < board_size and
                    next_pos not in visited):
                    
                    cell_type = board[next_pos]
                    # 不能移动到墙壁、可破坏方块、炸弹
                    is_obstacle = False
                    if cell_type == BombGame.WALL or cell_type == BombGame.DESTRUCTIBLE_BLOCK:
                        is_obstacle = True
                    
                    is_bomb_at_next_pos = False
                    for bomb_info in bombs:
                        if bomb_info['pos'] == next_pos:
                            is_bomb_at_next_pos = True
                            break

                    if not is_obstacle and not is_bomb_at_next_pos:
                        visited.add(next_pos)
                        q.append((next_pos, path + [next_pos]))
        return None
