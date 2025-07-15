"""
MCTS Bot - 改进的蒙特卡洛树搜索算法

特性：
- 防守优先策略：优先防守对手威胁
- 智能进攻选择：无威胁时积极寻找进攻机会
- 复杂威胁检测：识别双三、双四等复杂威胁
- 潜在威胁预测：预测对手可能的下一步动作
- 启发式模拟：使用启发式而非完全随机模拟
- 渐进偏差：在扩展阶段使用启发式选择
- RAVE算法：快速动作价值评估
- 虚拟损失：避免重复访问同一节点
"""
import time
import random
import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from agents.base_agent import BaseAgent
import config
import copy


class MCTSNode:
    """
    MCTS节点类
    
    包含节点的所有必要信息：
    - 状态：当前游戏状态
    - 父子关系：与父节点和子节点的连接
    - 统计信息：访问次数、价值、RAVE统计
    - 探索参数：UCB1常数、虚拟损失
    """
    
    def __init__(self, state, parent=None, action=None):
        """
        初始化MCTS节点
        
        Args:
            state: 游戏状态
            parent: 父节点
            action: 到达此节点的动作
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []  # 子节点列表
        self.visits = 0  # 访问次数
        self.value = 0.0  # 累计价值
        self.untried_actions = self._get_untried_actions()  # 未尝试的动作
        self.exploration_constant = 1.414  # UCB1探索常数
        self.virtual_loss = 0  # 虚拟损失（避免重复访问）
        self.rave_visits = 0  # RAVE访问次数
        self.rave_value = 0.0  # RAVE累计价值
    
    def _get_untried_actions(self):
        """
        获取未尝试的动作列表
        
        Returns:
            有效动作列表
        """
        if hasattr(self.state, 'get_valid_actions'):
            return self.state.get_valid_actions()
        return []
    
    def is_fully_expanded(self):
        """
        检查节点是否完全展开
        
        Returns:
            True如果所有动作都已尝试，否则False
        """
        return len(self.untried_actions) == 0
    
    def is_terminal(self):
        """
        检查是否为终止节点（游戏结束）
        
        Returns:
            True如果游戏结束，否则False
        """
        if hasattr(self.state, 'is_terminal'):
            return self.state.is_terminal()
        return False
    
    def get_winner(self):
        """
        获取游戏获胜者
        
        Returns:
            获胜者ID，如果游戏未结束则返回None
        """
        if hasattr(self.state, 'get_winner'):
            return self.state.get_winner()
        return None
    
    def clone_state(self):
        """
        克隆游戏状态
        
        Returns:
            克隆的游戏状态
        """
        if hasattr(self.state, 'clone'):
            return self.state.clone()
        return self.state
    
    def add_child(self, action, state):
        """
        添加子节点
        
        Args:
            action: 动作
            state: 新状态
            
        Returns:
            新创建的子节点
        """
        child = MCTSNode(state, parent=self, action=action)
        self.children.append(child)
        return child
    
    def get_ucb1_value(self, total_visits):
        """
        计算UCB1值（包含虚拟损失和RAVE）
        
        UCB1公式：exploitation + exploration + RAVE
        
        Args:
            total_visits: 总访问次数
            
        Returns:
            UCB1值
        """
        if self.visits == 0:
            return float('inf')
        
        # 基础UCB1：利用项 + 探索项
        exploitation = (self.value - self.virtual_loss) / self.visits
        exploration = self.exploration_constant * math.sqrt(math.log(total_visits) / self.visits)
        
        # RAVE项：快速动作价值评估
        rave_score = 0
        if self.rave_visits > 0:
            rave_score = self.rave_value / self.rave_visits
            # RAVE权重随访问次数递减（访问越多，RAVE影响越小）
            rave_weight = max(0, 1 - self.visits / 50)
            exploitation = (1 - rave_weight) * exploitation + rave_weight * rave_score
        
        return exploitation + exploration
    
    def select_child(self, total_visits):
        """
        选择最佳子节点
        
        Args:
            total_visits: 总访问次数
            
        Returns:
            最佳子节点，如果没有子节点则返回None
        """
        if not self.children:
            return None
        
        # 选择UCB1值最高的子节点
        best_child = max(self.children, key=lambda c: c.get_ucb1_value(total_visits))
        return best_child


class MCTSBot(BaseAgent):
    """
    改进的MCTS Bot
    
    特性：
    - 防守优先策略：优先防守对手威胁
    - 智能进攻选择：无威胁时积极寻找进攻机会
    - 复杂威胁检测：识别双三、双四等复杂威胁
    - 潜在威胁预测：预测对手可能的下一步动作
    - 启发式模拟：使用启发式而非完全随机模拟
    - 渐进偏差：在扩展阶段使用启发式选择
    - RAVE算法：快速动作价值评估
    - 虚拟损失：避免重复访问同一节点
    """
    
    def __init__(self, name: str = "MCTSBot", player_id: int = 1, 
                 simulation_count: int = 1000, exploration_constant: float = 1.414):
        """
        初始化MCTS Bot
        
        Args:
            name: Bot名称
            player_id: 玩家ID (1或2)
            simulation_count: 模拟次数
            exploration_constant: 探索常数
        """
        super().__init__(name, player_id)
        self.simulation_count = simulation_count
        self.exploration_constant = exploration_constant
        
        # 从配置获取参数
        ai_config = config.AI_CONFIGS.get('mcts', {})
        self.simulation_count = ai_config.get('simulation_count', simulation_count)
        self.exploration_constant = ai_config.get('exploration_constant', exploration_constant)
        self.timeout = ai_config.get('timeout', 2)  # 减少超时时间到2秒
        
        # 改进的MCTS参数
        self.virtual_loss = 3  # 虚拟损失值（避免重复访问）
        self.rave_threshold = 10  # RAVE阈值
        self.progressive_bias = True  # 渐进偏差（使用启发式选择）
        self.nodes_created = 0  # 创建的节点数
        self.simulations_performed = 0  # 执行的模拟次数
    
    def get_action(self, observation: Any, env: Any) -> Any:
        """
        使用改进的MCTS选择动作（防守优先，无威胁时积极进攻）
        
        决策优先级：
        1. 直接获胜机会（最高优先级）
        2. 防守威胁（成五、活四、冲四、复杂威胁）
        3. 进攻机会（成五、活四、冲四、活三）
        4. 战略布局（围绕现有棋局）
        5. MCTS搜索（深度搜索最佳动作）
        
        Args:
            observation: 当前观察
            env: 环境对象
            
        Returns:
            选择的动作
        """
        start_time = time.time()
        
        # 获取有效动作
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            return None
        
        # 第一步：处理AI第一步的特殊情况
        board = env.game.board
        board_size = env.game.board_size
        my_count = np.sum(board == self.player_id)
        opp_count = np.sum(board == (3 - self.player_id))
        
        if my_count == 0:
            return self._handle_first_move(board, board_size, valid_actions)

        # 第二步：检查直接获胜机会（最高优先级）
        winning_moves = self._find_winning_moves(env)
        if winning_moves:
            return winning_moves[0]  # 直接获胜

        # 第三步：检查防守需求（包括复杂威胁预测）
        defensive_moves = self._find_defensive_moves(env)
        if defensive_moves:
            return self._select_best_defense(defensive_moves, env)

        # 第四步：无威胁时积极进攻
        offensive_moves = self._find_offensive_moves(env)
        if offensive_moves:
            return self._select_best_offense(offensive_moves, env)
        
        # 第五步：无进攻机会时，围绕现有棋局思考
        strategic_moves = self._find_strategic_moves(env)
        if strategic_moves:
            return strategic_moves[0]
        
        # 第六步：使用MCTS搜索最佳动作
        return self._mcts_search(env, valid_actions, start_time)
    
    def _handle_first_move(self, board, board_size, valid_actions):
        """
        处理AI第一步的特殊逻辑
        
        Args:
            board: 当前棋盘
            board_size: 棋盘大小
            valid_actions: 有效动作列表
            
        Returns:
            选择的第一步动作
        """
        # 找到玩家首子
        opp_pos = np.argwhere(board == (3 - self.player_id))
        if len(opp_pos) > 0:
            opp_first = opp_pos[0]
            # 以玩家首子为中心，优先在其周围落子
            candidates = []
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    x, y = opp_first[0] + dx, opp_first[1] + dy
                    if 0 <= x < board_size and 0 <= y < board_size and board[x, y] == 0:
                        candidates.append((x, y))
            if candidates:
                # 优先选择靠近中心的位置
                center = board_size // 2
                candidates.sort(key=lambda pos: abs(pos[0]-center)+abs(pos[1]-center))
                return candidates[0]
        
        # 否则优先选择中心位置
        center = board_size // 2
        if board[center, center] == 0:
            return (center, center)
        
        # 最后选择第一个有效动作
        return valid_actions[0]
    
    def _select_best_defense(self, defensive_moves, env):
        """
        选择最佳防守点
        
        Args:
            defensive_moves: 防守点列表
            env: 游戏环境
            
        Returns:
            最佳防守动作
        """
        # 按威胁等级排序：成五 > 活四 > 冲四 > 复杂威胁 > 潜在威胁
        for move in defensive_moves:
            if self._can_win_at_position(env.game.board, move[0], move[1], 3 - self.player_id, env.game.board_size):
                return move  # 成五威胁
        for move in defensive_moves:
            if self._can_form_live_four(env.game.board, move[0], move[1], 3 - self.player_id, env.game.board_size):
                return move  # 活四威胁
        for move in defensive_moves:
            if self._can_form_blocked_four(env.game.board, move[0], move[1], 3 - self.player_id, env.game.board_size):
                return move  # 冲四威胁
        for move in defensive_moves:
            if self._evaluate_complex_threat(env.game.board, move[0], move[1], 3 - self.player_id, env.game.board_size) >= 3:
                return move  # 复杂威胁
        for move in defensive_moves:
            if self._evaluate_potential_threat(env.game.board, move[0], move[1], 3 - self.player_id, env.game.board_size) >= 2:
                return move  # 潜在威胁
        return defensive_moves[0]  # 默认选择第一个防守点
    
    def _select_best_offense(self, offensive_moves, env):
        """
        选择最佳进攻点
        
        Args:
            offensive_moves: 进攻点列表
            env: 游戏环境
            
        Returns:
            最佳进攻动作
        """
        # 按威胁等级排序：成五 > 活四 > 冲四 > 活三
        for move in offensive_moves:
            if self._can_win_at_position(env.game.board, move[0], move[1], self.player_id, env.game.board_size):
                return move  # 成五机会
        for move in offensive_moves:
            if self._can_form_live_four(env.game.board, move[0], move[1], self.player_id, env.game.board_size):
                return move  # 活四机会
        for move in offensive_moves:
            if self._can_form_blocked_four(env.game.board, move[0], move[1], self.player_id, env.game.board_size):
                return move  # 冲四机会
        return offensive_moves[0]  # 默认选择第一个进攻点
    
    def _mcts_search(self, env, valid_actions, start_time):
        """
        使用MCTS算法搜索最佳动作
        
        Args:
            env: 游戏环境
            valid_actions: 有效动作列表
            start_time: 开始时间
            
        Returns:
            最佳动作
        """
        # 创建根节点
        root = MCTSNode(env.game.clone())
        root.exploration_constant = self.exploration_constant
        
        # 动态调整模拟次数
        dynamic_simulations = min(self.simulation_count, 
                                max(50, int(self.timeout * 50)))
        
        # 执行MCTS搜索
        for i in range(dynamic_simulations):
            # 检查时间限制
            if time.time() - start_time > self.timeout:
                break
            
            # 选择
            node = self._select(root)
            
            # 扩展
            if node and not node.is_terminal():
                expanded_node = self._expand(node)
                if expanded_node:
                    node = expanded_node
                    self.nodes_created += 1
            
            # 模拟
            if node and not node.is_terminal():
                result = self._simulate(node.state)
                self.simulations_performed += 1
            else:
                result = self._get_result(node.state) if node else 0
            
            # 回传
            if node:
                self._backpropagate(node, result)
        
        # 选择最佳动作
        best_action = self._select_best_action(root, valid_actions)
        
        # 保证返回一个动作
        if best_action is None:
            best_action = valid_actions[0] if valid_actions else None
        
        # 更新统计
        move_time = time.time() - start_time
        self.total_moves += 1
        self.total_time += move_time
        
        return best_action


    
    def _select(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        选择阶段：使用改进的UCB1选择节点
        
        从根节点开始，沿着UCB1值最高的路径向下选择，
        直到找到一个未完全展开的节点或终止节点。
        
        Args:
            node: 当前节点
            
        Returns:
            选择的节点（未完全展开或终止节点）
        """
        while node.children and not node.is_terminal():
            # 如果当前节点未完全展开，返回该节点
            if not node.is_fully_expanded():
                return node
            
            # 应用虚拟损失（避免重复访问同一节点）
            for child in node.children:
                child.virtual_loss += self.virtual_loss
            
            # 选择UCB1值最高的子节点
            selected_child = node.select_child(node.visits)
            if selected_child is None:
                return node
            node = selected_child
        return node
    
    def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        扩展阶段：创建新子节点（使用渐进偏差）
        
        从未尝试的动作中选择一个，创建新的子节点。
        使用渐进偏差来优先选择启发式价值高的动作。
        
        Args:
            node: 要扩展的节点
            
        Returns:
            新创建的子节点，如果无法扩展则返回原节点
        """
        if node.is_terminal():
            return node
        
        if not node.untried_actions:
            return node
        
        # 使用渐进偏差选择动作（启发式选择）
        if self.progressive_bias and hasattr(node.state, 'get_valid_actions'):
            action = self._select_action_with_bias(node.untried_actions, node.state)
        else:
            action = random.choice(node.untried_actions)
        
        # 从未尝试动作列表中移除选中的动作
        node.untried_actions.remove(action)
        
        # 创建新状态
        new_state = node.state.clone()
        new_state.step(action)
        
        # 创建子节点
        child = node.add_child(action, new_state)
        return child
    
    def _select_action_with_bias(self, actions: List[Any], state) -> Any:
        """
        使用渐进偏差选择动作
        
        根据启发式评估为每个动作分配分数，然后使用轮盘赌选择。
        这样既考虑了启发式价值，又保持了随机性。
        
        Args:
            actions: 可用动作列表
            state: 当前游戏状态
            
        Returns:
            选择的动作
        """
        if not actions:
            return None
        
        # 计算每个动作的启发式分数
        action_scores = []
        for action in actions:
            score = self._heuristic_evaluation(action, state)
            action_scores.append((score, action))
        
        # 按分数排序（降序）
        action_scores.sort(reverse=True)
        
        # 使用轮盘赌选择（保持随机性）
        total_score = sum(score for score, _ in action_scores)
        if total_score > 0:
            r = random.uniform(0, total_score)
            cumulative = 0
            for score, action in action_scores:
                cumulative += score
                if cumulative >= r:
                    return action
        
        # 如果所有分数都为0，返回最高分数的动作
        return action_scores[0][1]
    
    def _heuristic_evaluation(self, action: Any, state) -> float:
        """
        启发式评估动作（增强防守版）
        
        综合考虑位置价值、威胁程度、防守价值和复杂威胁，
        优先考虑防守，确保AI能够有效应对对手威胁。
        
        Args:
            action: 要评估的动作
            state: 当前游戏状态
            
        Returns:
            动作的启发式评分
        """
        if not isinstance(action, tuple) or len(action) != 2:
            return 1.0
        
        row, col = action
        
        # 基础位置价值（中心优先）
        if hasattr(state, 'board'):
            board_size = len(state.board)
            center = board_size // 2
            distance_to_center = abs(row - center) + abs(col - center)
            base_score = max(0, 10 - distance_to_center)
            
            # 检查在此落子的威胁程度
            threat_score = self._evaluate_gomoku_threat(state, row, col)
            
            # 检查防守价值
            defensive_score = self._evaluate_defensive_value(state, row, col)
            
            # 检查复杂威胁
            complex_threat_score = self._evaluate_complex_threat(state.board, row, col, 3 - self.player_id, board_size)
            
            # 综合评分：防守优先
            total_score = base_score + threat_score * 10 + defensive_score * 15 + complex_threat_score * 20
            
            return total_score
        
        return 1.0
    
    def _evaluate_defensive_value(self, state, row, col):
        """评估防守价值"""
        board = state.board
        board_size = len(board)
        defensive_score = 0
        
        # 检查是否能阻止对手的威胁
        opponent = 3 - self.player_id
        
        # 检查四个方向
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            # 检查对手在此方向的威胁
            opponent_threat = self._evaluate_line_for_player(board, row, col, dx, dy, board_size, opponent)
            if opponent_threat >= 1000:  # 活四威胁
                defensive_score += 100
            elif opponent_threat >= 500:  # 冲四威胁
                defensive_score += 50
            elif opponent_threat >= 200:  # 活三威胁
                defensive_score += 20
            elif opponent_threat >= 20:   # 活二威胁
                defensive_score += 5
        
        return defensive_score
    
    def _evaluate_gomoku_threat(self, state, row, col):
        """评估五子棋位置的威胁程度"""
        board = state.board
        board_size = len(board)
        score = 0
        
        # 检查四个方向的威胁
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            line_score = self._evaluate_gomoku_line_threat(board, row, col, dx, dy, board_size)
            score += line_score
        
        return score
    
    def _evaluate_gomoku_line_threat(self, board, row, col, dx, dy, board_size):
        """评估五子棋某个方向的威胁（改进版）"""
        # 检查我方和对手的威胁
        my_threat = self._evaluate_line_for_player(board, row, col, dx, dy, board_size, 1)
        opponent_threat = self._evaluate_line_for_player(board, row, col, dx, dy, board_size, 2)
        
        # 优先考虑直接获胜和防守
        if my_threat >= 10000:  # 我方能直接获胜
            return my_threat * 10
        elif opponent_threat >= 10000:  # 对手能直接获胜，必须防守
            return opponent_threat * 8
        elif my_threat >= 1000:  # 我方能形成活四
            return my_threat * 5
        elif opponent_threat >= 1000:  # 对手能形成活四，必须防守
            return opponent_threat * 4
        else:
            # 其他威胁按正常权重计算
            return my_threat * 2 + opponent_threat
    
    def _evaluate_line_for_player(self, board, row, col, dx, dy, board_size, player):
        """评估某个方向对特定玩家的威胁"""
        # 临时在此位置放置棋子
        original_value = board[row, col]
        board[row, col] = player
        
        count = 1  # 当前位置
        blocked = 0
        
        # 向一个方向检查
        for k in range(1, 5):
            x, y = row + dx*k, col + dy*k
            if 0 <= x < board_size and 0 <= y < board_size:
                if board[x, y] == player:
                    count += 1
                elif board[x, y] != 0:
                    blocked += 1
                    break
                else:
                    break
            else:
                blocked += 1
                break
        
        # 向相反方向检查
        for k in range(1, 5):
            x, y = row - dx*k, col - dy*k
            if 0 <= x < board_size and 0 <= y < board_size:
                if board[x, y] == player:
                    count += 1
                elif board[x, y] != 0:
                    blocked += 1
                    break
                else:
                    break
            else:
                blocked += 1
                break
        
        # 恢复原值
        board[row, col] = original_value
        
        # 根据连子数和阻塞情况评分（修复权重）
        if count >= 5:
            return 10000  # 成五 - 最高优先级
        elif count == 4 and blocked == 0:
            return 1000   # 活四 - 次高优先级
        elif count == 4 and blocked == 1:
            return 500    # 冲四 - 降低权重
        elif count == 3 and blocked == 0:
            return 200    # 活三 - 降低权重
        elif count == 3 and blocked == 1:
            return 50     # 眠三 - 大幅降低权重
        elif count == 2 and blocked == 0:
            return 20     # 活二 - 降低权重
        elif count == 2 and blocked == 1:
            return 5      # 眠二 - 最低权重
        else:
            return 0

    def _simulate(self, state) -> float:
        """模拟阶段：使用启发式模拟到游戏结束（改进版）"""
        current_state = state.clone()
        
        # 限制模拟步数，避免无限循环
        max_simulation_steps = 50
        step_count = 0
        
        while not current_state.is_terminal() and step_count < max_simulation_steps:
            valid_actions = current_state.get_valid_actions()
            if not valid_actions:
                break
            
            # 使用启发式选择动作，而不是完全随机
            action = self._select_action_heuristic(current_state, valid_actions)
            current_state.step(action)
            step_count += 1
        
        return self._get_result(current_state)
    
    def _select_action_heuristic(self, state, valid_actions):
        """使用启发式选择动作"""
        if not valid_actions:
            return None
        
        # 计算每个动作的启发式分数
        action_scores = []
        for action in valid_actions:
            score = self._heuristic_evaluation(action, state)
            action_scores.append((score, action))
        
        # 按分数排序
        action_scores.sort(reverse=True)
        
        # 使用轮盘赌选择，但偏向高分动作
        total_score = sum(score for score, _ in action_scores)
        if total_score > 0:
            r = random.uniform(0, total_score)
            cumulative = 0
            for score, action in action_scores:
                cumulative += score
                if cumulative >= r:
                    return action
        
        # 如果所有分数都为0，随机选择
        return random.choice(valid_actions)
    
    def _get_result(self, state) -> float:
        """获取游戏结果（改进版）"""
        winner = state.get_winner()
        
        if winner == self.player_id:
            return 1.0
        elif winner is not None:
            return 0.0
        else:
            # 如果没有明确的胜负，使用局面评估
            return self._evaluate_position_result(state)
    
    def _evaluate_position_result(self, state):
        """评估局面的相对优势"""
        if not hasattr(state, 'board'):
            return 0.5
        
        board = state.board
        board_size = len(board)
        my_score = 0
        opponent_score = 0
        
        # 评估每个位置的棋型
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] == 0:
                    continue
                
                player = board[i, j]
                if player == self.player_id:
                    my_score += self._evaluate_position_for_player(board, i, j, board_size, player)
                else:
                    opponent_score += self._evaluate_position_for_player(board, i, j, board_size, player)
        
        # 归一化到0-1范围
        total_score = my_score + opponent_score
        if total_score == 0:
            return 0.5
        
        return my_score / total_score
    
    def _evaluate_position_for_player(self, board, row, col, board_size, player):
        """评估某个位置对特定玩家的价值"""
        score = 0
        
        # 检查四个方向的连子情况
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            line_score = self._evaluate_line_for_player(board, row, col, dx, dy, board_size, player)
            score += line_score
        
        return score
    
    def _find_winning_moves(self, env):
        """查找直接获胜的动作"""
        winning_moves = []
        board = env.game.board
        board_size = env.game.board_size
        
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] == 0:  # 空位置
                    # 检查直接获胜（成五）
                    if self._can_win_at_position(board, i, j, self.player_id, board_size):
                        winning_moves.append((i, j))
        
        return winning_moves
    
    def _find_defensive_moves(self, env):
        """检测所有防守点（包括双三、双四等复杂威胁）"""
        defensive_moves = []
        board = env.game.board
        board_size = env.game.board_size
        
        # 第一优先级：直接威胁
        urgent_threats = []
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] == 0:
                    if self._can_win_at_position(board, i, j, 3 - self.player_id, board_size):
                        urgent_threats.append((i, j, 5))  # 成五威胁
                    elif self._can_form_live_four(board, i, j, 3 - self.player_id, board_size):
                        urgent_threats.append((i, j, 4))  # 活四威胁
                    elif self._can_form_blocked_four(board, i, j, 3 - self.player_id, board_size):
                        urgent_threats.append((i, j, 3))  # 冲四威胁
        
        # 第二优先级：复杂威胁（双三、双四等）
        complex_threats = self._find_complex_threats(board, board_size)
        
        # 第三优先级：潜在威胁
        potential_threats = self._find_potential_threats(board, board_size)
        
        # 按优先级排序
        all_threats = urgent_threats + complex_threats + potential_threats
        all_threats.sort(key=lambda x: x[2], reverse=True)
        
        # 返回去重后的防守点
        seen = set()
        for threat in all_threats:
            if (threat[0], threat[1]) not in seen:
                defensive_moves.append((threat[0], threat[1]))
                seen.add((threat[0], threat[1]))
        
        return defensive_moves
    
    def _find_offensive_moves(self, env):
        """查找进攻机会（成五、活四、冲四、活三）"""
        offensive_moves = []
        board = env.game.board
        board_size = env.game.board_size
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] == 0:
                    if self._can_win_at_position(board, i, j, self.player_id, board_size):
                        offensive_moves.append((i, j))
                    elif self._can_form_live_four(board, i, j, self.player_id, board_size):
                        offensive_moves.append((i, j))
                    elif self._can_form_blocked_four(board, i, j, self.player_id, board_size):
                        offensive_moves.append((i, j))
                    elif self._can_form_live_three(board, i, j, self.player_id, board_size):
                        offensive_moves.append((i, j))
        return offensive_moves
    
    def _find_strategic_moves(self, env):
        """查找战略位置（围绕现有棋局，避免边角）"""
        strategic_moves = []
        board = env.game.board
        board_size = env.game.board_size
        
        # 找到所有现有棋子的位置
        existing_pieces = []
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] != 0:
                    existing_pieces.append((i, j))
        
        if not existing_pieces:
            # 如果没有棋子，优先选择中心位置
            center = board_size // 2
            if board[center, center] == 0:
                return [(center, center)]
            else:
                # 选择中心附近的空位
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        x, y = center + dx, center + dy
                        if 0 <= x < board_size and 0 <= y < board_size and board[x, y] == 0:
                            strategic_moves.append((x, y))
                return strategic_moves if strategic_moves else []
        
        # 计算现有棋子的中心
        avg_x = sum(pos[0] for pos in existing_pieces) / len(existing_pieces)
        avg_y = sum(pos[1] for pos in existing_pieces) / len(existing_pieces)
        
        # 在现有棋子周围寻找战略位置
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] == 0:  # 空位置
                    # 避免边角位置
                    if (i == 0 or i == board_size-1) and (j == 0 or j == board_size-1):
                        continue
                    
                    # 计算到现有棋子中心的距离
                    distance_to_center = abs(i - avg_x) + abs(j - avg_y)
                    
                    # 检查是否在现有棋子附近（距离不超过3）
                    near_existing = False
                    for piece in existing_pieces:
                        if abs(i - piece[0]) + abs(j - piece[1]) <= 3:
                            near_existing = True
                            break
                    
                    if near_existing and distance_to_center <= 5:
                        strategic_moves.append((i, j, distance_to_center))
        
        # 按距离排序，优先选择靠近现有棋子的位置
        strategic_moves.sort(key=lambda x: x[2])
        return [(x[0], x[1]) for x in strategic_moves]
    
    def _can_win_at_position(self, board, row, col, player, board_size):
        """检查在指定位置落子是否能直接获胜"""
        # 临时在此位置放置棋子
        original_value = board[row, col]
        board[row, col] = player
        
        # 检查四个方向是否能成五
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            count = 1  # 当前位置
            
            # 向一个方向检查
            for k in range(1, 5):
                x, y = row + dx*k, col + dy*k
                if 0 <= x < board_size and 0 <= y < board_size and board[x, y] == player:
                    count += 1
                else:
                    break
            
            # 向相反方向检查
            for k in range(1, 5):
                x, y = row - dx*k, col - dy*k
                if 0 <= x < board_size and 0 <= y < board_size and board[x, y] == player:
                    count += 1
                else:
                    break
            
            if count >= 5:
                board[row, col] = original_value
                return True
        
        # 恢复原值
        board[row, col] = original_value
        return False
    
    def _can_form_live_four(self, board, row, col, player, board_size):
        """检查是否能形成活四"""
        # 临时在此位置放置棋子
        original_value = board[row, col]
        board[row, col] = player
        
        has_live_four = False
        
        # 检查四个方向
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            count = 1  # 当前位置
            blocked = 0
            
            # 向一个方向检查
            for k in range(1, 5):
                x, y = row + dx*k, col + dy*k
                if 0 <= x < board_size and 0 <= y < board_size:
                    if board[x, y] == player:
                        count += 1
                    elif board[x, y] != 0:
                        blocked += 1
                        break
                    else:
                        break
                else:
                    blocked += 1
                    break
            
            # 向相反方向检查
            for k in range(1, 5):
                x, y = row - dx*k, col - dy*k
                if 0 <= x < board_size and 0 <= y < board_size:
                    if board[x, y] == player:
                        count += 1
                    elif board[x, y] != 0:
                        blocked += 1
                        break
                    else:
                        break
                else:
                    blocked += 1
                    break
            
            # 检查是否形成活四
            if count >= 4 and blocked == 0:
                has_live_four = True
                break
        
        # 恢复原值
        board[row, col] = original_value
        return has_live_four
    
    def _can_form_blocked_four(self, board, row, col, player, board_size):
        """检查是否能形成冲四"""
        # 临时在此位置放置棋子
        original_value = board[row, col]
        board[row, col] = player
        
        has_blocked_four = False
        
        # 检查四个方向
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            count = 1  # 当前位置
            blocked = 0
            
            # 向一个方向检查
            for k in range(1, 5):
                x, y = row + dx*k, col + dy*k
                if 0 <= x < board_size and 0 <= y < board_size:
                    if board[x, y] == player:
                        count += 1
                    elif board[x, y] != 0:
                        blocked += 1
                        break
                    else:
                        break
                else:
                    blocked += 1
                    break
            
            # 向相反方向检查
            for k in range(1, 5):
                x, y = row - dx*k, col - dy*k
                if 0 <= x < board_size and 0 <= y < board_size:
                    if board[x, y] == player:
                        count += 1
                    elif board[x, y] != 0:
                        blocked += 1
                        break
                    else:
                        break
                else:
                    blocked += 1
                    break
            
            # 检查是否形成冲四
            if count >= 4 and blocked == 1:
                has_blocked_four = True
                break
        
        # 恢复原值
        board[row, col] = original_value
        return has_blocked_four
    
    def _can_form_live_three(self, board, row, col, player, board_size):
        """检查是否能形成活三"""
        # 临时在此位置放置棋子
        original_value = board[row, col]
        board[row, col] = player
        
        has_live_three = False
        
        # 检查四个方向
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            count = 1  # 当前位置
            blocked = 0
            
            # 向一个方向检查
            for k in range(1, 5):
                x, y = row + dx*k, col + dy*k
                if 0 <= x < board_size and 0 <= y < board_size:
                    if board[x, y] == player:
                        count += 1
                    elif board[x, y] != 0:
                        blocked += 1
                        break
                    else:
                        break
                else:
                    blocked += 1
                    break
            
            # 向相反方向检查
            for k in range(1, 5):
                x, y = row - dx*k, col - dy*k
                if 0 <= x < board_size and 0 <= y < board_size:
                    if board[x, y] == player:
                        count += 1
                    elif board[x, y] != 0:
                        blocked += 1
                        break
                    else:
                        break
                else:
                    blocked += 1
                    break
            
            # 检查是否形成活三
            if count >= 3 and blocked == 0:
                has_live_three = True
                break
        
        # 恢复原值
        board[row, col] = original_value
        return has_live_three
    
    def _find_complex_threats(self, board, board_size):
        """检测复杂威胁（双三、双四等）"""
        complex_threats = []
        opponent = 3 - self.player_id
        
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] == 0:
                    threat_level = self._evaluate_complex_threat(board, i, j, opponent, board_size)
                    if threat_level > 0:
                        complex_threats.append((i, j, threat_level))
        
        return complex_threats
    
    def _evaluate_complex_threat(self, board, row, col, player, board_size):
        """评估复杂威胁等级"""
        # 临时在此位置放置棋子
        original_value = board[row, col]
        board[row, col] = player
        
        threat_level = 0
        live_threes = 0
        blocked_fours = 0
        live_twos = 0
        
        # 检查四个方向
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            line_score = self._evaluate_line_for_player(board, row, col, dx, dy, board_size, player)
            
            # 统计各种棋型
            if line_score >= 1000:  # 活四
                threat_level += 4
            elif line_score >= 500:  # 冲四
                blocked_fours += 1
            elif line_score >= 200:  # 活三
                live_threes += 1
            elif line_score >= 20:   # 活二
                live_twos += 1
        
        # 复杂威胁评分
        if live_threes >= 2:  # 双活三
            threat_level += 3
        elif live_threes >= 1 and blocked_fours >= 1:  # 活三+冲四
            threat_level += 3
        elif blocked_fours >= 2:  # 双冲四
            threat_level += 2
        elif live_threes >= 1 and live_twos >= 2:  # 活三+双活二
            threat_level += 2
        elif live_twos >= 3:  # 三活二
            threat_level += 1
        
        # 恢复原值
        board[row, col] = original_value
        return threat_level
    
    def _find_potential_threats(self, board, board_size):
        """检测潜在威胁（预测走势）"""
        potential_threats = []
        opponent = 3 - self.player_id
        
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] == 0:
                    threat_level = self._evaluate_potential_threat(board, i, j, opponent, board_size)
                    if threat_level > 0:
                        potential_threats.append((i, j, threat_level))
        
        return potential_threats
    
    def _evaluate_potential_threat(self, board, row, col, player, board_size):
        """评估潜在威胁（预测对手下一步）"""
        # 临时在此位置放置棋子
        original_value = board[row, col]
        board[row, col] = player
        
        threat_level = 0
        
        # 检查对手在此落子后能形成的威胁
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            # 检查这个方向上的连子情况
            count = 1
            blocked = 0
            space_before = 0
            space_after = 0
            
            # 向一个方向检查
            for k in range(1, 5):
                x, y = row + dx*k, col + dy*k
                if 0 <= x < board_size and 0 <= y < board_size:
                    if board[x, y] == player:
                        count += 1
                    elif board[x, y] == 0:
                        space_after += 1
                        break
                    else:
                        blocked += 1
                        break
                else:
                    blocked += 1
                    break
            
            # 向相反方向检查
            for k in range(1, 5):
                x, y = row - dx*k, col - dy*k
                if 0 <= x < board_size and 0 <= y < board_size:
                    if board[x, y] == player:
                        count += 1
                    elif board[x, y] == 0:
                        space_before += 1
                        break
                    else:
                        blocked += 1
                        break
                else:
                    blocked += 1
                    break
            
            # 评估潜在威胁
            if count >= 3 and blocked == 0:
                # 活三潜力
                if space_before >= 1 and space_after >= 1:
                    threat_level += 2
            elif count >= 2 and blocked == 0:
                # 活二潜力
                if space_before >= 2 and space_after >= 2:
                    threat_level += 1
        
        # 恢复原值
        board[row, col] = original_value
        return threat_level
    
    def _predict_opponent_moves(self, board, board_size):
        """预测对手可能的下一步动作"""
        opponent = 3 - self.player_id
        predicted_moves = []
        
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] == 0:
                    # 评估对手在此落子的价值
                    value = self._evaluate_position_for_player(board, i, j, board_size, opponent)
                    if value > 50:  # 只考虑有价值的动作
                        predicted_moves.append((i, j, value))
        
        # 按价值排序
        predicted_moves.sort(key=lambda x: x[2], reverse=True)
        return predicted_moves[:5]  # 返回前5个最可能的动作
    
    def _select_best_action(self, root: MCTSNode, valid_actions: List[Any]) -> Any:
        """选择最佳动作（考虑访问次数和价值，保证永远返回动作）"""
        if not root.children:
            # 如果没有子节点，使用启发式选择
            if valid_actions:
                # 使用启发式评估选择最佳动作
                best_action = None
                best_score = float('-inf')
                for action in valid_actions:
                    score = self._heuristic_evaluation(action, root.state)
                    if score > best_score:
                        best_score = score
                        best_action = action
                return best_action if best_action is not None else valid_actions[0]
            else:
                return None
        
        # 使用UCB1值选择最佳动作
        best_child = max(root.children, key=lambda c: c.get_ucb1_value(root.visits))
        return best_child.action
    
    def _backpropagate(self, node: MCTSNode, result: float):
        """回传阶段：更新节点统计（包含RAVE）"""
        path_actions = []
        current = node
        
        # 收集路径上的动作
        while current.parent is not None:
            if current.action is not None:
                path_actions.append(current.action)
            current = current.parent
        
        # 更新路径上的所有节点
        current = node
        while current is not None:
            current.visits += 1
            current.value += result
            
            # 更新RAVE统计
            if current.parent is not None:
                for action in path_actions:
                    for child in current.parent.children:
                        if child.action == action:
                            child.rave_visits += 1
                            child.rave_value += result
                            break
            
            # 清除虚拟损失
            current.virtual_loss = 0
            current = current.parent
    
    def reset(self):
        """重置MCTS Bot"""
        super().reset()
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取MCTS Bot详细信息
        
        返回的信息包括：
        - 基础信息：名称、类型、描述
        - 策略信息：模拟次数、探索常数
        - 性能统计：创建的节点数、执行的模拟次数
        - 算法参数：虚拟损失、RAVE阈值、渐进偏差
        - 时间控制：超时设置
        
        Returns:
            包含Bot详细信息的字典
        """
        info = super().get_info()
        info.update({
            'type': 'MCTS',
            'description': '使用改进的蒙特卡洛树搜索的Bot，具有防守优先策略',
            'strategy': f'MCTS with {self.simulation_count} simulations',
            'exploration_constant': self.exploration_constant,
            'timeout': self.timeout,
            'virtual_loss': self.virtual_loss,
            'rave_threshold': self.rave_threshold,
            'progressive_bias': self.progressive_bias,
            'nodes_created': self.nodes_created,
            'simulations_performed': self.simulations_performed,
            'features': [
                '防守优先策略',
                '复杂威胁检测',
                '潜在威胁预测',
                '启发式模拟',
                '渐进偏差选择',
                'RAVE算法',
                '虚拟损失机制',
                '动态模拟调整'
            ]
        })
        return info 