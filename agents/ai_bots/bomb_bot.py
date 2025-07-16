import random
import numpy as np
from agents.base_agent import BaseAgent
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque
from games.bomb.bomb_game import BombGame # 导入BombGame，用于访问其常量
from games.bomb.bomb_env import BombEnv

# 导入Stable Baselines3 PPO模型和环境包装器
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym # 导入gymnasium



# 为了让BombAI能够使用RL模型，我们需要一个函数来创建与训练时兼容的环境
# 这个函数将被BombAI内部调用，用于模型预测
def _create_env_for_rl_inference(board_size: int, player_id: int, opponent_agent=None):
    """
    为RL模型推理创建一个简单的BombEnv环境。
    注意：这里不需要SubprocVecEnv，因为只是单步推理。
    """
    # 确保BombEnv的__init__方法可以接受这些参数
    env = BombEnv(board_size=board_size, player_id=player_id, opponent_agent=opponent_agent)
    return env

class BombAI(BaseAgent):
    """泡泡堂AI智能体，可选择使用RL模型或内置逻辑"""
    
    def __init__(self, name="BombAI", player_id=2, use_rl_model: bool = False, rl_model_path: Optional[str] = None):
        super().__init__(name, player_id)
        self._move_delay_counter = 0 # 控制移动频率的计数器
        self._move_delay_ticks = 2 # AI每2个游戏帧移动一次
        self.escape_path = []
        
        self.use_rl_model = use_rl_model
        self.rl_model: Optional[PPO] = None
        self.rl_env_for_inference: Optional[DummyVecEnv] = None # 用于RL模型推理的虚拟环境

        if self.use_rl_model and rl_model_path:
            try:
                print(f"BombAI: Loading RL model from {rl_model_path} for player {self.player_id}...")
                self.rl_model = PPO.load(rl_model_path)
                print("BombAI: RL model loaded successfully.")
            except Exception as e:
                print(f"BombAI: Error loading RL model: {e}")
                self.use_rl_model = False # 加载失败则回退到内置AI
        
    def get_action(self, observation: Dict[str, Any], game_instance: Any) -> Union[Tuple[int, int], Tuple[int, int, bool]]:
        """
        获取动作。
        observation 包含游戏的完整状态。
        game_instance 是 BombGame 的实例，提供了获取有效动作和游戏信息的方法。
        """
        # 如果使用RL模型
        if self.use_rl_model and self.rl_model:
            # 确保rl_env_for_inference已初始化
            if self.rl_env_for_inference is None:
                # 在这里创建一个临时的环境实例，用于RL模型的观测空间转换
                # 注意：这里我们只关心观测空间的结构，实际的游戏逻辑由 game_instance 驱动
                # opponent_agent 需要传递None，因为这个环境只用于RL推理，不涉及另一个AI
                self.rl_env_for_inference = DummyVecEnv([lambda: _create_env_for_rl_inference(
                    board_size=game_instance.board_size,
                    player_id=self.player_id,
                    opponent_agent=None # RL模型推理时，不需对手AI
                )])
                # 强制设置环境的game实例为当前的game_instance，以获取最新状态
                self.rl_env_for_inference.envs[0].game = game_instance

            # 更新RL推理环境的内部游戏状态
            self.rl_env_for_inference.envs[0].game = game_instance
            
            # 获取RL模型所需的观察
            # _get_observation() 方法在 BombEnv 中，它会从 game_instance 获取状态并格式化
            rl_obs = self.rl_env_for_inference.envs[0]._get_observation()

            # RL智能体预测动作
            # model.predict 期望的是批处理的观察，即使只有一个环境，也需要将其包装
            # rl_obs 已经是 Dict 格式，DummyVecEnv 会将其转换为批处理格式
            action_idx, _states = self.rl_model.predict(rl_obs, deterministic=True)
            
            # 将离散动作索引转换回游戏动作格式
            # 动作空间: 0: Up, 1: Down, 2: Left, 3: Right, 4: Stay, 5: Plant Bomb
            if action_idx == 0: # Up
                return (-1, 0)
            elif action_idx == 1: # Down
                return (1, 0)
            elif action_idx == 2: # Left
                return (0, -1)
            elif action_idx == 3: # Right
                return (0, 1)
            elif action_idx == 4: # Stay
                return (0, 0)
            elif action_idx == 5: # Plant Bomb
                return (0, 0, True)
            else:
                return (0, 0) # 默认不操作
        
        # 以下是原始的AI逻辑 (如果use_rl_model为False或RL模型加载失败)
        self._move_delay_counter += 1
        if self._move_delay_counter % self._move_delay_ticks != 0:
            return (0, 0) # 保持原地不动

        self._move_delay_counter = 0

        valid_actions = game_instance.get_valid_actions(self.player_id)
        if not valid_actions:
            return (0, 0)

        game_state = observation
        board = game_state['board']
        player_info = game_instance.get_player_info(self.player_id)
        current_pos = player_info['pos']
        current_bombs = player_info['current_bombs']
        max_bombs = player_info['bombs_max']
        bomb_range = player_info['range']
        player_shield_active = player_info['shield_active']
        player_range_type = player_info['range_type']
        all_bombs = game_instance.bombs


        self.escape_path = self._find_escape_path(current_pos, game_state)
        if self.escape_path and len(self.escape_path) > 0:
            next_pos = self.escape_path[0]
            dr = next_pos[0] - current_pos[0]
            dc = next_pos[1] - current_pos[1]
            escape_action = (dr, dc)
            return escape_action
        
        safe_moves = []
        for action in valid_actions:
            if len(action) == 3 and action[2]:
                if self._is_position_safe(current_pos, game_state):
                    safe_moves.append(action)
                continue
            
            new_pos = (current_pos[0] + action[0], current_pos[1] + action[1])
            if self._is_position_safe(new_pos, game_state):
                safe_moves.append(action)
        
        if not safe_moves:
            return (0, 0)
        
        
        items = game_state['items']
        if items:
            priority_items_pos = []
            other_items_pos = []
            
            for item_info in items:
                item_pos = item_info['pos']
                item_type = item_info['type']
                
                if item_type == BombGame.ITEM_SHIELD and not player_shield_active:
                    priority_items_pos.append(item_pos)
                elif item_type == BombGame.ITEM_SQUARE_RANGE_UP and player_range_type == 'cross':
                    priority_items_pos.append(item_pos)
                else:
                    other_items_pos.append(item_pos)
            
            if priority_items_pos:
                best_item_action = self._get_action_towards_nearest_target(current_pos, priority_items_pos, safe_moves, board, game_state['bombs'])
                if best_item_action:
                    return best_item_action
            elif other_items_pos:
                best_item_action = self._get_action_towards_nearest_target(current_pos, other_items_pos, safe_moves, board, game_state['bombs'])
                if best_item_action:
                    return best_item_action


        bomb_desirability = 0
        if (0, 0, True) in safe_moves and current_bombs < max_bombs:
            potential_explosion_cells = self._get_explosion_coords(current_pos, bomb_range, board.shape[0], board, player_range_type)
            destructible_blocks_hit_count = 0
            hits_enemy = False
            other_player_pos = game_instance.player1_pos if self.player_id == 2 else game_instance.player2_pos
            
            for exp_pos in potential_explosion_cells:
                if game_instance._is_valid_position(exp_pos):
                    cell_type = board[exp_pos]
                    if cell_type == BombGame.DESTRUCTIBLE_BLOCK:
                        if player_range_type == 'cross':
                            destructible_blocks_hit_count += 1
                        else:
                            destructible_blocks_hit_count += 0.5
                    if exp_pos == other_player_pos:
                        hits_enemy = True
                if  hits_enemy:
                    break 
            
            if hits_enemy:
                bomb_desirability = 100
            elif destructible_blocks_hit_count > 0:
                bomb_desirability = destructible_blocks_hit_count
                if player_range_type == 'square':
                    bomb_desirability += 1

        if bomb_desirability > 0:
            simulated_bombs = game_state['bombs'] + [{
                'pos': current_pos,
                'range': bomb_range,
                'timer': 15,  
                'range_type': player_range_type
            }]
            escape_path = self._find_escape_path(current_pos, {**game_state, 'bombs': simulated_bombs})
            if escape_path and len(escape_path) > 0 and len(escape_path) < 6:
                return (0, 0, True)

        destructible_blocks = []
        for r in range(game_instance.board_size):
            for c in range(game_instance.board_size):
                if board[r, c] == BombGame.DESTRUCTIBLE_BLOCK:
                    destructible_blocks.append((r, c))
        
        if destructible_blocks:
            best_block_action = self._get_action_towards_nearest_target(current_pos, destructible_blocks, safe_moves, board, game_state['bombs'])
            if best_block_action:
                return best_block_action

        movement_actions = [a for a in safe_moves if not (len(a) == 3 and a[2])]
        if movement_actions:
            other_player_pos = game_instance.get_player_info(1)['pos']
            valid_list = []
            for act in movement_actions:
                if self._leads_towards_enemy(current_pos, act, other_player_pos):
                    valid_list.append(act)
            if valid_list != []:
                return random.choice(valid_list)
            else:
                return random.choice(movement_actions)        
        return (0, 0)
    
    def _leads_towards_enemy(self, current_pos: Tuple[int, int], action: Tuple[int, int], enemy_pos: Tuple[int, int]) -> bool:
        """检查一个动作是否导致AI向敌人移动"""
        next_pos = (current_pos[0] + action[0], current_pos[1] + action[1])
        current_dist = abs(current_pos[0] - enemy_pos[0]) + abs(current_pos[1] - enemy_pos[1])
        next_dist = abs(next_pos[0] - enemy_pos[0]) + abs(next_pos[1] - enemy_pos[1])
        return next_dist < current_dist


    def _is_position_safe(self, pos: Tuple[int, int], game_state: Dict[str, Any]) -> bool:
        """检查给定位置在下一帧是否安全 (没有爆炸)"""
        board_size = game_state['board'].shape[0]
        explosions = game_state['explosions']
        bombs = game_state['bombs']

        if not (0 <= pos[0] < board_size and 0 <= pos[1] < board_size):
            return False

        for bomb in bombs:
            if bomb['timer'] <= 10:
                explosion_coords = self._get_explosion_coords(bomb['pos'], bomb['range'], board_size, game_state['board'], bomb['range_type'])
                if pos in explosion_coords:
                    return False
        
        for exp in explosions:
            if exp['pos'] == pos:
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

