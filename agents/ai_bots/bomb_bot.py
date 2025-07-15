import random
import numpy as np
from agents.base_agent import BaseAgent
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque
from games.bomb.bomb_game import BombGame # Import BombGame to access its constants

class BombAI(BaseAgent):
    """Bomberman AI Agent"""
    
    def __init__(self, name="BombAI", player_id=2): # Default to Player 2
        super().__init__(name, player_id)
        self._move_delay_counter = 0 # Counter to control movement frequency
        self._move_delay_ticks = 2 # AI moves once every 5 ticks

    def get_action(self, observation: Dict[str, Any], env: Any) -> Union[Tuple[int, int], Tuple[int, int, bool]]:
        """
        Get action.
        Observation contains the full state of the game.
        Env provides methods to get valid actions and game information.
        """
        # Increment the move delay counter
        self._move_delay_counter += 1

        # If it's not time to move, return a no-op action
        if self._move_delay_counter % self._move_delay_ticks != 0:
            return (0, 0) # Stay in place

        # Reset the counter after taking an action
        self._move_delay_counter = 0

        valid_actions = env.get_valid_actions(self.player_id)
        if not valid_actions:
            return (0, 0) # If no valid actions, stay in place

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
            if len(action) == 3 and action[2]: # Place bomb action
                # Placing a bomb itself does not immediately kill the player, unless the player is killed by their own bomb.
                # But it is necessary to ensure that there is a safe escape path after placing the bomb.
                # Simplified handling here: assume placing a bomb is safe, but subsequent escape is needed.
                safe_moves.append(action)
                continue
            
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
        #    - If a bomb can be placed and there are destructible blocks or enemies around
        should_place_bomb = False
        if (0, 0, True) in safe_moves and current_bombs < max_bombs:
            # Check for destructible blocks or enemies around
            # Simple check in four directions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                target_pos = (current_pos[0] + dr, current_pos[1] + dc)
                if env.game._is_valid_position(target_pos):
                    cell_type = board[target_pos]
                    if cell_type == BombGame.DESTRUCTIBLE_BLOCK:
                        should_place_bomb = True
                        break
                    # Check enemy position
                    other_player_pos = env.game.player1_pos if self.player_id == 2 else env.game.player2_pos
                    if target_pos == other_player_pos:
                        should_place_bomb = True
                        break
            if should_place_bomb:
                # After placing a bomb, need to check for a safe escape path
                # Simulate placing a bomb
                temp_bombs = game_state['bombs'] + [{'pos': current_pos, 'timer': 3, 'range': bomb_range, 'owner': self.player_id, 'range_type': player_range_type}]
                temp_game_state = game_state.copy()
                temp_game_state['bombs'] = temp_bombs

                # Check if an escape path exists from the explosion range
                escape_paths = self._find_escape_path(current_pos, temp_game_state, bomb_range, player_range_type)
                if escape_paths:
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
                    
                    # Check if the new position is occupied by a bomb (cannot pass through unexploded bombs)
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
        After placing a bomb, find a safe escape path.
        Returns a path, or None if no safe path.
        """
        q = deque([(start_pos, [])]) # (current position, path)
        visited = {start_pos}
        
        board_size = game_state['board'].shape[0]
        board = game_state['board']
        bombs = game_state['bombs'] # Contains the newly placed bomb

        # Simulate explosion area for the next frame
        simulated_explosion_cells = set()
        for bomb in bombs:
            # Only simulate bombs that are about to explode (timer <= 1)
            if bomb['timer'] <= 1: 
                simulated_explosion_cells.update(self._get_explosion_coords(bomb['pos'], bomb['range'], board_size, board, bomb['range_type']))

        # BFS to find a safe path
        while q:
            current_pos, path = q.popleft()

            # If the current position is not within the simulated explosion area, a safe spot is found
            if current_pos not in simulated_explosion_cells:
                return path # Return the escape path

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
