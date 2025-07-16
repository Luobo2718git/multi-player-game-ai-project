import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from games.base_env import BaseEnv
from games.bomb.bomb_game import BombGame
import gymnasium as gym # 导入 gymnasium
from gymnasium import spaces # 导入 spaces
import config # 导入 config

class BombEnv(BaseEnv):
    """
    泡泡堂（炸弹人）环境
    实现gym风格接口
    """

    def __init__(self, board_size=15, player_id=1, opponent_agent=None, **kwargs):
        self.board_size = board_size
        self.game = BombGame(board_size=board_size, **kwargs)
        super().__init__(self.game)

        # 指定当前RL智能体控制的玩家ID
        self.player_id = player_id
        # 存储对手智能体
        self.opponent_agent = opponent_agent

        self._setup_spaces() # 初始化观察和动作空间

    def _setup_spaces(self):
        """设置观察空间和动作空间"""
        # 动作空间: 6个离散动作
        self.action_space = spaces.Discrete(6)

        # 观察空间定义 (用于 CNN 输入)
        # board: (board_size, board_size) values 0-44 (BombGame.WILL_EXPLOSION)
        # player_pos: (board_size, board_size) one-hot (1 at pos, 0 else)
        # bombs_timer: (board_size, board_size) values 0-3 (countdown)
        # explosions: (board_size, board_size) binary (1 for explosion, 0 else)
        # items: (board_size, board_size) different values for different items (BombGame.ITEM_BOMB_UP, etc.)

        num_channels = 0
        num_channels += 1 # 棋盘层 (墙, 可破坏块, 空地)
        num_channels += 2 # 玩家位置层 (玩家1, 玩家2)
        num_channels += 1 # 炸弹计时器层
        num_channels += 1 # 爆炸区域层
        num_channels += 1 # 物品层

        # 所有的板状信息组合成一个 (channels, height, width) 的张量
        self.image_observation_space = spaces.Box(low=0, high=BombGame.WILL_EXPLOSION, 
                                                shape=(num_channels, self.board_size, self.board_size), 
                                                dtype=np.uint8)

        # 额外的标量信息:
        # player1_bombs_max, player1_current_bombs, player1_range, player1_shield_active, player1_shield_timer, player1_range_type, player1_score
        # player2_bombs_max, player2_current_bombs, player2_range, player2_shield_active, player2_shield_timer, player2_range_type, player2_score
        # current_moves, max_moves
        
        # 每个玩家7个属性，共2个玩家，加上2个游戏状态属性 = 7*2 + 2 = 16个标量特征
        scalar_feature_dim = 16 
        self.scalar_observation_space = spaces.Box(low=0, high=self.board_size * self.board_size * 2, # 高值可以根据实际情况调整，确保能覆盖分数、最大移动数等
                                                shape=(scalar_feature_dim,), dtype=np.float32)

        # 使用 Dict 组合图像和标量观察空间
        self.observation_space = spaces.Dict({
            "board_features": self.image_observation_space,
            "scalar_features": self.scalar_observation_space
        })


    def _get_observation(self) -> Dict[str, Any]:
        """
        获取观察。
        将原始游戏状态转换为RL智能体可用的格式 (Dict Space)。
        """
        state = self.game.get_state()
        board = np.array(state['board'], dtype=np.uint8)
        board_size = self.board_size

        # 构建图像特征
        board_features = np.zeros((self.image_observation_space.shape[0], board_size, board_size), dtype=np.uint8)
        
        # 通道 0: 棋盘基础元素 (墙, 可破坏块, 空地)
        board_features[0] = board

        # 通道 1: 玩家1位置
        p1_pos_layer = np.zeros((board_size, board_size), dtype=np.uint8)
        p1_pos = state['player1_pos']
        if p1_pos: # 确保玩家存在
            p1_pos_layer[p1_pos[0], p1_pos[1]] = 1
        board_features[1] = p1_pos_layer

        # 通道 2: 玩家2位置
        p2_pos_layer = np.zeros((board_size, board_size), dtype=np.uint8)
        p2_pos = state['player2_pos']
        if p2_pos: # 确保玩家存在
            p2_pos_layer[p2_pos[0], p2_pos[1]] = 1
        board_features[2] = p2_pos_layer

        # 通道 3: 炸弹计时器
        bomb_timer_layer = np.zeros((board_size, board_size), dtype=np.uint8)
        for bomb_info in state['bombs']:
            r, c = bomb_info['pos']
            # 炸弹ID是10-12，我们可以将其映射到1-3的倒计时
            # 假设 BombGame.BOMB_START_ID + 2 是初始倒计时
            bomb_timer_layer[r, c] = bomb_info['id'] - BombGame.BOMB_START_ID + 1
        board_features[3] = bomb_timer_layer
        
        # 通道 4: 爆炸区域
        explosion_layer = np.zeros((board_size, board_size), dtype=np.uint8)
        for r, c in state['explosions']:
            explosion_layer[r, c] = 1
        board_features[4] = explosion_layer

        # 通道 5: 物品
        item_layer = np.zeros((board_size, board_size), dtype=np.uint8)
        for item_pos, item_type in state['items'].items():
            r, c = item_pos
            item_layer[r, c] = item_type # 使用物品的ID作为值
        board_features[5] = item_layer


        # 构建标量特征
        scalar_features = np.array([
            state['player1_bombs_max'], state['player1_current_bombs'], state['player1_range'], 
            float(state['player1_shield_active']), state['player1_shield_timer'], 
            state['player1_range_type'], state['player1_score'],
            state['player2_bombs_max'], state['player2_current_bombs'], state['player2_range'], 
            float(state['player2_shield_active']), state['player2_shield_timer'], 
            state['player2_range_type'], state['player2_score'],
            state['current_moves'], state['max_moves']
        ], dtype=np.float32)

        return {
            "board_features": board_features,
            "scalar_features": scalar_features
        }

    def _get_action_mask(self, player_id: int) -> np.ndarray:
        """
        获取动作掩码。
        返回一个布尔数组，指示哪些动作是当前可行的。
        """
        valid_actions_game_format = self.game.get_valid_actions(player_id)
        action_mask = np.zeros(6, dtype=bool) # 6个动作

        action_map = {
            (-1, 0): 0, # UP
            (1, 0): 1,  # DOWN
            (0, -1): 2, # LEFT
            (0, 1): 3,  # RIGHT
            (0, 0, True): 4, # BOMB
            (0, 0): 5   # STAY
        }

        for action_tuple in valid_actions_game_format:
            if action_tuple in action_map:
                action_mask[action_map[action_tuple]] = True
            # 特殊处理炸弹动作，需要确保有炸弹可用
            elif action_tuple == (0,0,True) and self.game.get_player_info(player_id)['current_bombs'] > 0:
                 action_mask[4] = True


        return action_mask

    def step(self, action_rl: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        执行一步游戏，RL智能体执行action_rl，对手智能体执行其动作。
        返回 (observation, reward, done, truncated, info)
        """
        # 将RL智能体的离散动作转换为游戏需要的动作元组
        action_map_reverse = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
            4: (0, 0, True),
            5: (0, 0)
        }
        action_rl_tuple = action_map_reverse[action_rl]

        # 获取RL智能体和对手智能体的玩家ID
        rl_player_id = self.player_id
        opponent_player_id = 3 - self.player_id # 如果玩家1是RL，则玩家2是对手；反之亦然

        action_p1 = None
        action_p2 = None

        # 记录RL智能体执行动作前的玩家信息，用于计算奖励
        rl_player_info_before = self.game.get_player_info(rl_player_id)
        opponent_player_info_before = self.game.get_player_info(opponent_player_id)


        if rl_player_id == 1:
            action_p1 = action_rl_tuple
            # 让对手智能体决定动作
            if self.opponent_agent:
                # 对手AI可能需要原始的state dict，而不是RL格式的observation
                opponent_obs_for_ai = self.game.get_state()
                action_p2 = self.opponent_agent.get_action(opponent_obs_for_ai, self.game)
        else: # rl_player_id == 2
            action_p2 = action_rl_tuple
            if self.opponent_agent:
                opponent_obs_for_ai = self.game.get_state()
                action_p1 = self.opponent_agent.get_action(opponent_obs_for_ai, self.game)

        # 执行游戏一步
        self.game.update(action_p1, action_p2)


        # 计算奖励
        reward = self._calculate_reward(rl_player_id, rl_player_info_before, opponent_player_info_before)

        observation = self._get_observation()
        done = self.game.is_terminal()
        truncated = self.game.is_terminal() # 游戏结束也视为 truncated
        info = self._get_info() # 包含其他信息，如分数、赢家等

        return observation, reward, done, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed) # 调用BaseEnv的reset，它会调用self.game.reset()
        self.game = BombGame(self.board_size) # 确保游戏也被重置
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def _calculate_reward(self, player_id: int, 
                          rl_player_info_before: Dict[str, Any],
                          opponent_player_info_before: Dict[str, Any]) -> float:
        """
        根据游戏状态变化计算当前玩家的奖励。
        player_id: 当前RL智能体的玩家ID
        rl_player_info_before: RL智能体在动作执行前的玩家信息
        opponent_player_info_before: 对手智能体在动作执行前的玩家信息
        """
        reward = config.REWARDS['per_step_penalty'] # 每步惩罚

        current_rl_player_info = self.game.get_player_info(player_id)
        current_opponent_player_info = self.game.get_player_info(3 - player_id)

        # 检查RL玩家是否死亡
        if not current_rl_player_info['alive']:
            reward += config.REWARDS['lose'] # 玩家死亡，给予失败惩罚
            return reward # 直接返回，因为游戏结束了

        # 检查对手是否死亡 (击中敌人，导致对手死亡)
        if opponent_player_info_before['alive'] and not current_opponent_player_info['alive']:
            reward += config.REWARDS['win'] # 对手死亡，给予胜利奖励 (也包含了击中敌人的最高奖励)
            # 如果游戏结构复杂，有血量等，这里可以额外增加一个击中但未致死的奖励
            return reward # 直接返回，因为游戏结束了

        # 检查分数变化 (破坏方块)
        score_diff = current_rl_player_info['score'] - rl_player_info_before['score']
        if score_diff > 0:
            reward += score_diff * config.REWARDS['destroy_block']

        # 检查是否收集到物品
        # 炸弹数量增加
        if current_rl_player_info['bombs_max'] > rl_player_info_before['bombs_max']:
            reward += config.REWARDS['collect_item']
        # 炸弹范围增加
        if current_rl_player_info['range'] > rl_player_info_before['range']:
            reward += config.REWARDS['collect_item']
        # 护盾激活
        if not rl_player_info_before['shield_active'] and current_rl_player_info['shield_active']:
            reward += config.REWARDS['collect_item'] * 2 # 护盾可能更重要，给更高奖励
        # 方形范围增加 (如果游戏有这个机制)
        if current_rl_player_info['range_type'] > rl_player_info_before['range_type']:
            reward += config.REWARDS['collect_item']


        # 游戏结束时的胜负奖励 (如果到时间结束)
        if self.game.is_terminal():
            if self.game.get_winner() == player_id:
                reward += config.REWARDS['win']
            elif self.game.get_winner() == 0: # 平局
                pass # 不奖励也不惩罚
            else: # 对手赢了
                reward += config.REWARDS['lose']

        return reward

    def clone(self) -> 'BombEnv':
        """克隆环境"""
        cloned_game = self.game.clone()
        # 克隆环境时，也需要克隆 player_id 和 opponent_agent
        cloned_env = BombEnv(self.board_size, player_id=self.player_id, opponent_agent=self.opponent_agent)
        cloned_env.game = cloned_game
        return cloned_env