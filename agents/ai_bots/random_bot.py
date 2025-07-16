import random
from agents.base_agent import BaseAgent
import numpy as np

class RandomBot(BaseAgent):
    def get_action(self, observation, env):
        valid_actions = env.get_valid_actions()
        # 五子棋智能随机
        if hasattr(env, 'game') and hasattr(env.game, 'board') and hasattr(env.game, '_check_win'):
            board = env.game.board
            board_size = env.game.board_size
            # 优先中心及周围
            center = board_size // 2
            center_actions = [(i, j) for i, j in valid_actions if abs(i - center) <= 2 and abs(j - center) <= 2]
            if center_actions:
                valid_actions = center_actions
            # 简单安全性过滤：避免直接被对方五连
            safe_actions = []
            for action in valid_actions:
                row, col = action
                if board[row, col] != 0:
                    continue
                # 模拟落子后对方能否直接五连
                board[row, col] = env.game.current_player
                env.game.switch_player()
                opp_win = False
                for opp_action in env.get_valid_actions():
                    r, c = opp_action
                    if board[r, c] == 0:
                        board[r, c] = env.game.current_player
                        if env.game._check_win(r, c, env.game.current_player):
                            opp_win = True
                        board[r, c] = 0
                        if opp_win:
                            break
                env.game.switch_player()
                board[row, col] = 0
                if not opp_win:
                    safe_actions.append(action)
            if safe_actions:
                valid_actions = safe_actions
            return random.choice(valid_actions)
        # 贪吃蛇智能随机
        if hasattr(env, 'game') and hasattr(env.game, 'snake1') and hasattr(env.game, 'board_size'):
            # 只选安全方向
            head = env.game.snake1[0] if hasattr(env.game, 'snake1') and env.game.snake1 else None
            snakes = env.game.snake1 + env.game.snake2 if hasattr(env.game, 'snake2') else env.game.snake1
            board_size = env.game.board_size
            safe_actions = []
            for action in valid_actions:
                if head is None:
                    continue
                new_head = (head[0] + action[0], head[1] + action[1])
                # 不撞墙
                if not (0 <= new_head[0] < board_size and 0 <= new_head[1] < board_size):
                    continue
                # 不撞自己或对手
                if new_head in snakes:
                    continue
                safe_actions.append(action)
            if safe_actions:
                return random.choice(safe_actions)
            else:
                return random.choice(valid_actions)
        # 其它情况，原始随机
        return random.choice(valid_actions)

class GreedyBot(BaseAgent):
    """五子棋/贪吃蛇贪心AI，支持自定义奖励函数"""
    def __init__(self, name="GreedyBot", player_id=1, reward_fn=None):
        super().__init__(name, player_id)
        self.reward_fn = reward_fn

    def get_action(self, observation, env):
        valid_actions = env.get_valid_actions()
        best_action = None
        best_score = float('-inf')
        for action in valid_actions:
            score = self.evaluate_action(action, observation, env)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def evaluate_action(self, action, observation, env):
        # 五子棋贪心评估
        if hasattr(env, 'game') and hasattr(env.game, 'board') and hasattr(env.game, 'board_size'):
            board = env.game.board.copy()
            row, col = action
            player = env.game.current_player if hasattr(env.game, 'current_player') else self.player_id
            opponent = 3 - player
            # 1. 立即获胜
            board[row, col] = player
            if env.game._check_win(row, col, player):
                return 1e6
            # 2. 阻止对手立即获胜
            board[row, col] = opponent
            if env.game._check_win(row, col, opponent):
                return 1e5
            board[row, col] = player
            # 3. 连子数奖励（进攻）
            max_conn = self.count_connections(board, row, col, player)
            # 4. 阻止对手连子（防守）
            max_opp_conn = self.count_connections(board, row, col, opponent)
            # 5. 位置奖励（中心优先）
            center = board.shape[0] // 2
            dist_center = abs(row - center) + abs(col - center)
            # 6. 组合分数
            score = max_conn * 100 + max_opp_conn * 80 + (10 - dist_center)
            if self.reward_fn:
                score += self.reward_fn(action, observation, env)
            return score
        # 贪吃蛇贪心评估
        if hasattr(env, 'game') and hasattr(env.game, 'snake1') and hasattr(env.game, 'board_size'):
            head = env.game.snake1[0] if hasattr(env.game, 'snake1') and env.game.snake1 else None
            foods = env.game.foods if hasattr(env.game, 'foods') else []
            board_size = env.game.board_size
            new_head = (head[0] + action[0], head[1] + action[1]) if head else None
            # 1. 安全性
            if not (0 <= new_head[0] < board_size and 0 <= new_head[1] < board_size):
                return -1e6
            if new_head in env.game.snake1 or (hasattr(env.game, 'snake2') and new_head in env.game.snake2):
                return -1e6
            # 2. 距离食物奖励
            min_food_dist = min([abs(new_head[0] - f[0]) + abs(new_head[1] - f[1]) for f in foods], default=100)
            score = 100 - min_food_dist * 10
            # 3. 靠近中心奖励
            center = board_size // 2
            score += 10 - (abs(new_head[0] - center) + abs(new_head[1] - center))
            if self.reward_fn:
                score += self.reward_fn(action, observation, env)
            return score
        # 其它情况
        return 0

    def count_connections(self, board, row, col, player):
        max_conn = 0
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1
            # 正方向
            r, c = row + dr, col + dc
            while 0 <= r < board.shape[0] and 0 <= c < board.shape[1] and board[r, c] == player:
                count += 1
                r += dr
                c += dc
            # 负方向
            r, c = row - dr, col - dc
            while 0 <= r < board.shape[0] and 0 <= c < board.shape[1] and board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            max_conn = max(max_conn, count)
        return max_conn 