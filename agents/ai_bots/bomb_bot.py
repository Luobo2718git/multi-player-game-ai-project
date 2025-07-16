import random
import numpy as np
from agents.base_agent import BaseAgent
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque
# from heapq import heappush, heappop # 用于A*算法，现在将使用BFS，所以不再需要
from games.bomb.bomb_game import BombGame # 导入BombGame，用于访问其常量

class BombAI(BaseAgent):
    """泡泡堂AI智能体"""
    
    def __init__(self, name="BombAI", player_id=2): # 默认为玩家2
        super().__init__(name, player_id)
        self._move_delay_counter = 0 # 控制移动频率的计数器
        self._move_delay_ticks = 2 # AI每2个游戏帧移动一次

    def get_action(self, observation: Dict[str, Any], env: Any) -> Union[Tuple[int, int], Tuple[int, int, bool]]:
        """
        获取动作。
        observation 包含游戏的完整状态。
        env 提供了获取有效动作和游戏信息的方法。
        """
        # 增加移动延迟计数器
        self._move_delay_counter = random.randint(1, 3) # 随机增加1-3个计数器值，模拟AI思考时间

        # 如果未到移动时间，返回不操作动作
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
        current_pos = player_info['pos']
        current_bombs = player_info['current_bombs']
        max_bombs = player_info['bombs_max']
        bomb_range = player_info['range']
        player_shield_active = player_info['shield_active']
        player_shield_timer = player_info['shield_timer']
        player_range_type = player_info['range_type']
        
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

        # 2. 收集物品 (优先护盾和方形爆炸)
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
                # 使用BFS寻找最近的优先级物品
                best_item_action = self._get_action_towards_nearest_target(current_pos, priority_items_pos, safe_moves, board, game_state['bombs'], game_state)
                if best_item_action:
                    return best_item_action
            elif other_items_pos: # 如果没有优先级物品，收集其他物品
                # 使用BFS寻找最近的其他物品
                best_item_action = self._get_action_towards_nearest_target(current_pos, other_items_pos, safe_moves, board, game_state['bombs'], game_state)
                if best_item_action:
                    return best_item_action

        # 3. 战略性放置炸弹：寻找能破坏方块或击中敌人的最佳放置点
        best_bomb_action_to_take = None
        highest_bomb_desirability = -1
        
        # 存储所有潜在的炸弹放置点及其效益和到达该点的第一个动作
        # 列表元素: (desirability, first_action_to_reach_bomb_spot, path_length_to_spot)
        potential_bomb_opportunities = [] 

        other_player_pos = env.game.player1_pos if self.player_id == 2 else env.game.player2_pos

        # 遍历棋盘上所有可能的炸弹放置点 (空地)
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                potential_bomb_pos = (r, c)
                cell_type = board[potential_bomb_pos]

                # 只能在空地放置炸弹，且该位置当前没有炸弹
                is_bomb_at_pos = False
                for bomb_info in game_state['bombs']:
                    if bomb_info['pos'] == potential_bomb_pos:
                        is_bomb_at_pos = True
                        break

                if cell_type == BombGame.EMPTY and not is_bomb_at_pos:
                    # 模拟在此位置放置炸弹的效益
                    simulated_explosion_cells = self._get_explosion_coords(potential_bomb_pos, bomb_range, board.shape[0], board, player_range_type)
                    
                    sim_destructible_blocks_hit_count = 0
                    sim_hits_enemy = False
                    for exp_pos in simulated_explosion_cells:
                        if env.game._is_valid_position(exp_pos):
                            cell_type_at_exp = board[exp_pos]
                            if cell_type_at_exp == BombGame.DESTRUCTIBLE_BLOCK:
                                sim_destructible_blocks_hit_count += 1
                            if exp_pos == other_player_pos:
                                sim_hits_enemy = True
                    
                    sim_desirability = -1
                    if sim_hits_enemy:
                        sim_desirability = 100 # 击中敌人优先级最高
                    elif sim_destructible_blocks_hit_count > 0:
                        sim_desirability = sim_destructible_blocks_hit_count # 基于破坏方块数量
                        if player_range_type == 'square':
                            sim_desirability += 5 # 如果是方形炸弹，额外加分

                    if sim_desirability > 0:
                        # 检查从当前位置到这个潜在炸弹放置点的路径
                        path_to_spot = self._bfs_pathfinding(current_pos, potential_bomb_pos, game_state, board, game_state['bombs'])
                        
                        if path_to_spot: # 路径存在
                            # 模拟放置炸弹后的游戏状态，以检查逃离路径
                            temp_bombs_after_placement = game_state['bombs'] + [{'pos': potential_bomb_pos, 'timer': 3, 'range': bomb_range, 'owner': self.player_id, 'range_type': player_range_type}]
                            temp_game_state_after_placement = game_state.copy()
                            temp_game_state_after_placement['bombs'] = temp_bombs_after_placement
                            
                            escape_paths_after_bomb = self._find_escape_path(potential_bomb_pos, temp_game_state_after_placement, bomb_range, player_range_type)
                            
                            if escape_paths_after_bomb:
                                # 如果已经在目标位置，则动作是放置炸弹
                                if current_pos == potential_bomb_pos:
                                    action_to_take = (0, 0, True)
                                else:
                                    # 否则，动作是走向目标位置的第一步
                                    first_step_pos = path_to_spot[1]
                                    dr = first_step_pos[0] - current_pos[0]
                                    dc = first_step_pos[1] - current_pos[1]
                                    action_to_take = (dr, dc)
                                
                                # 确保这个动作是安全移动或放置炸弹
                                if action_to_take in safe_moves:
                                    potential_bomb_opportunities.append((sim_desirability, action_to_take, len(path_to_spot))) # (效益, 动作, 路径长度)
        
        # 从所有潜在的炸弹放置机会中选择最佳的
        if potential_bomb_opportunities:
            # 优先选择效益最高的，其次选择路径最短的
            potential_bomb_opportunities.sort(key=lambda x: (x[0], x[2]), reverse=True) # 按效益降序，路径长度升序
            best_bomb_action_to_take = potential_bomb_opportunities[0][1] 
            if best_bomb_action_to_take:
                return best_bomb_action_to_take


        # 4. 追逐敌人 (如果安全且敌人未被护盾保护)
        other_player_pos = env.game.player1_pos if self.player_id == 2 else env.game.player2_pos
        other_player_info = env.game.get_player_info(env.game.get_other_player_id(self.player_id))
        enemy_alive = other_player_info['alive']
        enemy_shielded = other_player_info['shield_active']

        if enemy_alive and not enemy_shielded:
            # 尝试使用BFS寻找前往敌人的路径
            path_to_enemy = self._bfs_pathfinding(current_pos, other_player_pos, game_state, board, game_state['bombs'])
            if path_to_enemy and len(path_to_enemy) > 1:
                first_step_pos = path_to_enemy[1]
                dr = first_step_pos[0] - current_pos[0]
                dc = first_step_pos[1] - current_pos[1]
                action_to_enemy = (dr, dc)
                if action_to_enemy in safe_moves:
                    return action_to_enemy # 如果找到安全路径，则朝敌人移动

        # 5. 破坏方块 (如果安全) - 这部分逻辑现在可能被战略性放置炸弹部分覆盖，但作为备选保留
        # 如果AI没有选择放置炸弹，它仍然可能需要移动到可破坏方块旁边来为下次放置炸弹做准备
        destructible_blocks = []
        for r in range(env.game.board_size):
            for c in range(env.game.board_size):
                if board[r, c] == BombGame.DESTRUCTIBLE_BLOCK:
                    destructible_blocks.append((r, c))
        
        if destructible_blocks:
            # 使用BFS寻找最近的可破坏方块
            best_block_action = self._get_action_towards_nearest_target(current_pos, destructible_blocks, safe_moves, board, game_state['bombs'], game_state)
            if best_block_action:
                return best_block_action

        # 6. 随机移动 (在安全移动中选择)
        # 过滤掉放置炸弹的动作
        movement_actions = [a for a in safe_moves if not (len(a) == 3 and a[2])]
        if movement_actions:
            return random.choice(movement_actions)
        
        return (0, 0) # 实在没招了，原地不动

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
            if bomb['timer'] <= 1: # 即将在下一帧爆炸
                # 模拟爆炸范围
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

        for dr, dc in directions:
            for i in range(1, bomb_range + 1):
                exp_pos = (bomb_pos[0] + dr * i, bomb_pos[1] + dc * i)
                if not (0 <= exp_pos[0] < board_size and 0 <= exp_pos[1] < board_size):
                    break

                cell_type = board[exp_pos]
                if cell_type == BombGame.WALL:
                    break
                
                explosion_cells.add(exp_pos)
                
                if cell_type == BombGame.DESTRUCTIBLE_BLOCK:
                    if range_type == 'cross': # 十字形爆炸遇到可破坏方块就停止
                        break
                    # 如果是方形爆炸，则穿透可破坏方块，不break
        return explosion_cells

    def _bfs_pathfinding(self, start: Tuple[int, int], goal: Tuple[int, int], game_state: Dict[str, Any], board: np.ndarray, bombs: List[Dict[str, Any]]) -> Optional[List[Tuple[int, int]]]:
        """BFS寻路算法"""
        
        def get_neighbors(pos):
            """获取当前位置的有效邻居"""
            x, y = pos
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # 上，下，左，右
                nx, ny = x + dr, y + dc
                next_pos = (nx, ny)

                # 检查边界
                if not (0 <= nx < board_size and 0 <= ny < board_size):
                    continue

                cell_type = board[next_pos]
                
                # 检查墙壁和可破坏方块 (作为移动障碍)
                if cell_type == BombGame.WALL or cell_type == BombGame.DESTRUCTIBLE_BLOCK:
                    continue 
                
                # 检查未爆炸的炸弹 (不能移动到炸弹上)
                is_bomb_at_next_pos = False
                for bomb_info in bombs:
                    if bomb_info['pos'] == next_pos:
                        is_bomb_at_next_pos = True
                        break
                if is_bomb_at_next_pos:
                    continue

                # 检查下一个位置在下一帧是否安全 (不受爆炸影响)
                if not self._is_position_safe(next_pos, game_state):
                    continue

                neighbors.append(next_pos)
            return neighbors
        
        board_size = board.shape[0] # 从棋盘数组获取棋盘大小

        q = deque([(start, [start])]) # (当前位置, 路径)
        visited = {start}

        while q:
            current, path = q.popleft()
            
            if current == goal:
                return path # 返回从起点到目标的路径
            
            for neighbor in get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, path + [neighbor]))
        
        return None  # 没有找到路径

    def _get_action_towards_nearest_target(self, start_pos: Tuple[int, int], targets: List[Tuple[int, int]], safe_moves: List[Union[Tuple[int, int], Tuple[int, int, bool]]], board: np.ndarray, bombs: List[Dict[str, Any]], game_state: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """使用BFS寻找前往最近目标的路径，并返回第一个动作"""
        
        best_action = None
        min_path_len = float('inf')

        for target in targets:
            # 调用BFS寻路
            path = self._bfs_pathfinding(start_pos, target, game_state, board, bombs)
            if path and len(path) > 1: # 路径存在且不只是起点本身
                # 将第一个移动动作转换为 (dr, dc)
                first_step_pos = path[1] # path[0] 是 start_pos
                dr = first_step_pos[0] - start_pos[0]
                dc = first_step_pos[1] - start_pos[1]
                action_tuple = (dr, dc)

                # 确保这个动作是安全的移动动作
                if action_tuple in safe_moves: 
                    # 如果有多个目标，选择路径最短的那个
                    if len(path) < min_path_len:
                        min_path_len = len(path)
                        best_action = action_tuple
        return best_action

    def _find_escape_path(self, start_pos: Tuple[int, int], game_state: Dict[str, Any], bomb_range: int, bomb_range_type: str) -> Optional[List[Tuple[int, int]]]:
        """
        在放置炸弹后，寻找一条安全逃离路径。
        返回一条路径，或者None如果没有安全路径。
        这个函数现在会考虑新放置的炸弹在未来爆炸时的危险区域。
        """
        q = deque([(start_pos, [])]) # (当前位置, 路径)
        visited = {start_pos}
        
        board_size = game_state['board'].shape[0]
        board = game_state['board']
        bombs_in_play = game_state['bombs'] # 包含新放置的炸弹

        # 最大炸弹计时器通常为3。我们需要模拟未来所有可能爆炸的区域。
        max_sim_ticks = 3 

        all_danger_zones_over_time = set()
        
        # 创建炸弹的深拷贝以模拟它们的计时器变化
        simulated_bombs = [b.copy() for b in bombs_in_play]

        # 模拟从当前帧到炸弹爆炸的帧，收集所有危险区域
        for tick_offset in range(max_sim_ticks + 1): # 模拟从计时器3到0
            current_tick_explosions = set()
            bombs_to_explode_this_tick = []

            for bomb in simulated_bombs:
                # 检查计时器是否有效 (大于0)
                if bomb['timer'] is not None and bomb['timer'] <= 0: # 这个炸弹将在当前模拟帧爆炸
                    current_tick_explosions.update(self._get_explosion_coords(bomb['pos'], bomb['range'], board_size, board, bomb['range_type']))
                    bombs_to_explode_this_tick.append(bomb)
                if bomb['timer'] is not None:
                    bomb['timer'] -= 1 # 减少计时器，用于下一帧模拟
            
            all_danger_zones_over_time.update(current_tick_explosions)
            # 从模拟列表中移除已爆炸的炸弹
            simulated_bombs = [b for b in simulated_bombs if b not in bombs_to_explode_this_tick]

        # BFS寻找安全路径
        while q:
            current_pos, path = q.popleft()

            # 如果当前位置不在任何未来爆炸区域内，则找到安全点
            if current_pos not in all_danger_zones_over_time:
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
                    
                    # 检查未爆炸的炸弹 (不能移动到炸弹上) - 这里需要检查原始的炸弹列表
                    is_bomb_at_next_pos = False
                    for bomb_info in game_state['bombs']: # 使用原始的 game_state 炸弹
                        if bomb_info['pos'] == next_pos and bomb_info['timer'] > 0: # 仍然是活跃的炸弹
                            is_bomb_at_next_pos = True
                            break

                    if not is_obstacle and not is_bomb_at_next_pos:
                        visited.add(next_pos)
                        q.append((next_pos, path + [next_pos]))
        return None

    # _leads_towards_enemy 方法不再直接用于决策，但为了完整性保留
    def _leads_towards_enemy(self, current_pos: Tuple[int, int], action: Tuple[int, int], enemy_pos: Tuple[int, int]) -> bool:
        """检查一个动作是否导致AI向敌人移动"""
        next_pos = (current_pos[0] + action[0], current_pos[1] + action[1])
        # 简单判断：如果新位置与敌人的曼哈顿距离小于当前位置与敌人的曼哈顿距离
        current_dist = abs(current_pos[0] - enemy_pos[0]) + abs(current_pos[1] - enemy_pos[1])
        next_dist = abs(next_pos[0] - enemy_pos[0]) + abs(next_pos[1] - enemy_pos[1])
        return next_dist < current_dist
