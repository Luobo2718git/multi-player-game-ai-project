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
        
        # 1. Safety First: Avoid current position and expected move position being in explosion range
        safe_moves = []
        for action in valid_actions:
            if len(action) == 3 and action[2]: # Place bomb action (0, 0, True)
                # Before allowing placing a bomb, check if the current position is safe (not affected by other bombs/explosions)
                if self._is_position_safe(current_pos, game_state):
                    safe_moves.append(action)
                continue # Continue checking the next valid action
            
            new_pos = (current_pos[0] + action[0], current_pos[1] + action[1])
            if self._is_position_safe(new_pos, game_state):
                safe_moves.append(action)
        
        if not safe_moves:
            return (0, 0) # If no safe moves, stay in place (or choose the safest possible action, simplified to no-op here)

        # 2. Collect items (prioritize shield and square explosion)
        items = game_state['items']
        if items:
            # Prioritize finding shield or square explosion items
            priority_items_pos = [] # Store only positions for pathfinding
            other_items_pos = []    # Store only positions for pathfinding
            
            for item_info in items: # item_info is now correctly a dictionary
                item_pos = item_info['pos'] # Extract the position tuple
                item_type = item_info['type'] # Extract the item type
                
                if item_type == BombGame.ITEM_SHIELD and not player_shield_active: # Prioritize shield if no shield
                    priority_items_pos.append(item_pos)
                elif item_type == BombGame.ITEM_SQUARE_RANGE_UP and player_range_type == 'cross': # Prioritize square explosion if cross explosion
                    priority_items_pos.append(item_pos)
                else:
                    other_items_pos.append(item_pos)
            
            if priority_items_pos:
                best_item_action = self._get_action_towards_nearest_target(current_pos, priority_items_pos, safe_moves, board, game_state['bombs'])
                if best_item_action:
                    return best_item_action
            elif other_items_pos: # If no priority items, collect other items
                best_item_action = self._get_action_towards_nearest_target(current_pos, other_items_pos, safe_moves, board, game_state['bombs'])
                if best_item_action:
                    return best_item_action


        # 3. Place bomb strategy:
        #    - If a bomb can be placed and its explosion range can destroy blocks or hit enemies
        #    - And the player's current position is safe (already checked via safe_moves)
        bomb_desirability = -1 # Desirability value, -1 means not desirable or impossible to place
        if (0, 0, True) in safe_moves and current_bombs < max_bombs:
            # Simulate placing a bomb at the current position (timer is 30, meaning 3 seconds until explosion)
            # This simulation is to check potential targets if it explodes.
            potential_explosion_cells = self._get_explosion_coords(current_pos, bomb_range, board.shape[0], board, player_range_type)

            destructible_blocks_hit_count = 0
            hits_enemy = False
            other_player_pos = env.game.player1_pos if self.player_id == 2 else env.game.player2_pos
            
            for exp_pos in potential_explosion_cells:
                if env.game._is_valid_position(exp_pos):
                    cell_type = board[exp_pos]
                    if cell_type == BombGame.DESTRUCTIBLE_BLOCK:
                        destructible_blocks_hit_count += 1
                    if exp_pos == other_player_pos:
                        hits_enemy = True
                # If a target has been found, can exit the loop early
                if destructible_blocks_hit_count or hits_enemy: # Fixed: Use the initialized variable
                    break 
            
            if hits_enemy:
                bomb_desirability = 100 # Hitting an enemy has the highest priority
            elif destructible_blocks_hit_count > 0:
                bomb_desirability = destructible_blocks_hit_count # Based on the number of blocks destroyed
                if player_range_type == 'square':
                    bomb_desirability += 5 # Additional points if it's a square bomb, encourages its use

        if bomb_desirability > 0: # If placing a bomb is desirable
            return (0, 0, True) # Place bomb

        # 4. Chase enemy or destroy blocks (if safe)
        # Prioritize destroying blocks, as it can yield items and open paths
        destructible_blocks = []
        for r in range(env.game.board_size):
            for c in range(env.game.board_size):
                if board[r, c] == BombGame.DESTRUCTIBLE_BLOCK:
                    destructible_blocks.append((r, c))
        
        if destructible_blocks:
            best_block_action = self._get_action_towards_nearest_target(current_pos, destructible_blocks, safe_moves, board, game_state['bombs'])
            if best_block_action:
                return best_block_action

        # 5. Random movement (choose from safe moves)
        # Filter out bomb-placing actions, as we've already handled bomb-placing logic
        movement_actions = [a for a in safe_moves if not (len(a) == 3 and a[2])]
        if movement_actions:
            return random.choice(movement_actions)
        
        return (0, 0) # If all else fails, stay in place

    def _is_position_safe(self, pos: Tuple[int, int], game_state: Dict[str, Any]) -> bool:
        """Check if a given position will be safe in the next frame (no explosion)"""
        board_size = game_state['board'].shape[0]
        explosions = game_state['explosions']
        bombs = game_state['bombs']

        # Check if within board boundaries
        if not (0 <= pos[0] < board_size and 0 <= pos[1] < board_size):
            return False

        # Check if there are bombs about to explode or existing explosions at the current position
        for bomb in bombs:
            if bomb['timer'] <= 1: # About to explode in the next frame
                # Simulate explosion range
                explosion_coords = self._get_explosion_coords(bomb['pos'], bomb['range'], board_size, game_state['board'], bomb['range_type'])
                if pos in explosion_coords:
                    return False
        
        # Check if within current explosion range
        for exp in explosions:
            if exp['pos'] == pos: # Explosion is still active
                return False
        
        return True

    def _get_explosion_coords(self, bomb_pos: Tuple[int, int], bomb_range: int, board_size: int, board: np.ndarray, range_type: str) -> set:
        """Calculate the explosion range coordinates of a bomb"""
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
                    if range_type == 'cross': # Cross-shaped explosion stops at destructible blocks
                        break
                    # If it's a square explosion, it penetrates destructible blocks, no break
        return explosion_cells

    def _get_action_towards_nearest_target(self, start_pos: Tuple[int, int], targets: List[Tuple[int, int]], safe_moves: List[Union[Tuple[int, int], Tuple[int, int, bool]]], board: np.ndarray, bombs: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
        """Use BFS to find the path to the nearest target and return the first action"""
        q = deque([(start_pos, [])]) # (current position, path)
        visited = {start_pos}
        
        board_size = board.shape[0]

        while q:
            current_pos, path = q.popleft()

            if current_pos in targets:
                if len(path) > 0:
                    # Convert the first movement action to (dr, dc)
                    first_step_pos = path[0]
                    dr = first_step_pos[0] - start_pos[0]
                    dc = first_step_pos[1] - start_pos[1]
                    action_tuple = (dr, dc)
                    # Ensure this action is a safe movement action
                    if action_tuple in safe_moves: 
                        return action_tuple
                return (0, 0) # If target is current position, or path is empty, stay in place

            # Iterate through all possible movement directions (excluding placing bombs, as this is for pathfinding)
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right

            for dr, dc in moves:
                next_pos = (current_pos[0] + dr, current_pos[1] + dc)

                if (0 <= next_pos[0] < board_size and 0 <= next_pos[1] < board_size and
                    next_pos not in visited):
                    
                    cell_type = board[next_pos]
                    # Cannot move to walls, destructible blocks, bombs
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

    def _find_escape_path(self, start_pos: Tuple[int, int], game_state: Dict[str, Any], bomb_range: int, bomb_range_type: str) -> Optional[List[Tuple[int, int]]]:
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
            # 只模拟即将爆炸的炸弹（timer <= 1）
            if bomb['timer'] <= 1: 
                simulated_explosion_cells.update(self._get_explosion_coords(bomb['pos'], bomb['range'], board_size, board, bomb['range_type']))

        # BFS寻找安全路径
        while q:
            current_pos, path = q.popleft()

            # 如果当前位置不在模拟爆炸区域内，则找到安全点
            if current_pos not in simulated_explosion_cells:
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
