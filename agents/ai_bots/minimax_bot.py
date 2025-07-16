import numpy as np
import time
from games.gomoku.gomoku_game import GomokuGame

class MinimaxBot:
    def __init__(self, name="Minimax AI", player_id=2, search_depth=3, time_limit=5.0, max_depth=None, **kwargs):
        self.name = name
        self.player_id = player_id
        # 兼容max_depth和search_depth参数
        if max_depth is not None:
            self.search_depth = max_depth
        else:
            self.search_depth = search_depth
        self.time_limit = time_limit  # 单步最大思考时间（秒）
        self.start_time = None

    def get_action(self, board, player):
        """
        综合防守优先、动态深度、时间控制和alpha-beta剪枝的主入口
        """
        # 兼容observation为dict或ndarray
        if isinstance(board, dict):
            if 'board' in board:
                board = board['board']
            else:
                raise ValueError('Observation dict missing "board" key')
        self.start_time = time.time()
        # 兼容player为int或env对象
        if hasattr(player, 'game') and hasattr(player.game, 'current_player'):
            player_id = player.game.current_player
        else:
            player_id = player
        opponent = 3 - player_id
        candidates = self.get_candidate_moves(board)
        # 动态深度调整：候选点多时自动降低深度
        if len(candidates) > 12:
            search_depth = max(1, self.search_depth - 1)
        else:
            search_depth = self.search_depth
        # 防守优先：五连、活四、活三、冲三
        for move in candidates:
            new_board = board.copy()
            new_board[move[0], move[1]] = opponent
            if self.check_five(new_board, opponent):
                return move
        for move in candidates:
            if self.pattern_count_with_move(board, opponent, move, 4, open_ends=2) > 0:
                return move
        for move in candidates:
            if self.pattern_count_with_move(board, opponent, move, 3, open_ends=2) > 0:
                return move
        for move in candidates:
            if self.pattern_count_with_move(board, opponent, move, 3, open_ends=1) > 0:
                return move
        # alpha-beta剪枝搜索
        best_score = -float('inf')
        best_move = None
        alpha = -float('inf')
        beta = float('inf')
        for move in candidates:
            new_board = board.copy()
            new_board[move[0], move[1]] = player_id
            score = self.alphabeta(new_board, search_depth-1, False, player_id, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            # 时间控制：超时立即返回当前最优
            if time.time() - self.start_time > self.time_limit:
                break
        return best_move

    def alphabeta(self, board, depth, maximizing, player, alpha, beta):
        """
        完整alpha-beta剪枝，带时间控制
        """
        # 时间控制：超时直接返回评估值
        if time.time() - self.start_time > self.time_limit:
            return self.evaluate(board, player)
        opponent = 3 - player
        if depth == 0 or self.is_terminal(board):
            return self.evaluate(board, player)
        if maximizing:
            max_eval = -float('inf')
            for move in self.get_candidate_moves(board):
                new_board = board.copy()
                new_board[move[0], move[1]] = player
                eval = self.alphabeta(new_board, depth-1, False, player, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # beta剪枝
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.get_candidate_moves(board):
                new_board = board.copy()
                new_board[move[0], move[1]] = opponent
                eval = self.alphabeta(new_board, depth-1, True, player, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # alpha剪枝
            return min_eval

    def get_candidate_moves(self, board):
        # 只返回有子周围1格内的空点，极大减少分支
        size = board.shape[0]
        candidates = set()
        for x in range(size):
            for y in range(size):
                if board[x, y] != 0:
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < size and 0 <= ny < size and board[nx, ny] == 0:
                                candidates.add((nx, ny))
        candidates = list(candidates)
        # 只取前20个点
        return candidates[:20] if len(candidates) > 20 else candidates

    def is_terminal(self, board):
        # 只要有五连就终止
        return self.check_five(board, 1) or self.check_five(board, 2)

    def check_five(self, board, player):
        # 检查是否有五连
        size = board.shape[0]
        for x in range(size):
            for y in range(size):
                if board[x, y] != player:
                    continue
                for dx, dy in [(1,0),(0,1),(1,1),(1,-1)]:
                    cnt = 0
                    for k in range(5):
                        nx, ny = x+dx*k, y+dy*k
                        if 0<=nx<size and 0<=ny<size and board[nx,ny]==player:
                            cnt += 1
                        else:
                            break
                    if cnt == 5:
                        return True
        return False

    def evaluate(self, board, player):
        """
        启发式评估函数：进攻+防守分数，防守权重更高
        可扩展更多棋型和权重
        """
        my_score = self.score_board(board, player)
        opp_score = self.score_board(board, 3-player)
        return my_score - 1.2 * opp_score

    def score_board(self, board, player):
        # 检查五连、活四、冲四、活三、冲三等
        patterns = [
            (100000, self.pattern_count(board, player, 5, open_ends=2)),  # 五连
            (20000, self.pattern_count(board, player, 4, open_ends=2)),   # 活四
            (5000, self.pattern_count(board, player, 4, open_ends=1)),    # 冲四
            (5000, self.pattern_count(board, player, 3, open_ends=2)),    # 活三
            (2000, self.pattern_count(board, player, 3, open_ends=1)),    # 冲三
        ]
        score = sum([v * n for v, n in patterns])
        return score

    def pattern_count(self, board, player, length, open_ends=2):
        # 统计指定长度、指定活性（两头/一头）的棋型数量
        size = board.shape[0]
        count = 0
        directions = [(1,0),(0,1),(1,1),(1,-1)]
        for x in range(size):
            for y in range(size):
                for dx, dy in directions:
                    line = []
                    for k in range(-1, length+1):
                        nx, ny = x+dx*k, y+dy*k
                        if 0<=nx<size and 0<=ny<size:
                            line.append(board[nx,ny])
                        else:
                            line.append(-1)  # 边界
                    # 检查中间length个是否全是player
                    if line[1:1+length].count(player) == length:
                        # 检查两头活性
                        ends = [line[0], line[-1]]
                        empty_ends = ends.count(0)
                        if empty_ends == open_ends:
                            count += 1
        return count

    def pattern_count_with_move(self, board, player, move, length, open_ends=2):
        # 检查在move落子后是否形成指定棋型（如活四）
        temp_board = board.copy()
        temp_board[move[0], move[1]] = player
        return self.pattern_count(temp_board, player, length, open_ends) 