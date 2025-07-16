"""
多游戏图形界面
支持五子棋和贪吃蛇的人机对战，修复中文显示问题
"""

import pygame
import sys
import time
import os
from typing import Optional, Tuple, Dict, Any
from games.gomoku import GomokuGame, GomokuEnv
from games.snake import SnakeGame, SnakeEnv
from agents import RandomBot, MinimaxBot, MCTSBot, HumanAgent, SnakeAI, SmartSnakeAI
import config

# 颜色定义
COLORS = {
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "BROWN": (139, 69, 19),
    "LIGHT_BROWN": (205, 133, 63),
    "RED": (255, 0, 0),
    "BLUE": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "GRAY": (128, 128, 128),
    "LIGHT_GRAY": (211, 211, 211),
    "DARK_GRAY": (64, 64, 64),
    "YELLOW": (255, 255, 0),
    "ORANGE": (255, 165, 0),
    "PURPLE": (128, 0, 128),
    "CYAN": (0, 255, 255),
    "PURPLE": (160, 32, 240),
}


class MultiGameGUI:
    """多游戏图形界面"""

    def __init__(self):
        # 初始化pygame
        pygame.init()

        # 设置中文字体
        self.font_path = self._get_chinese_font()
        self.font_large = pygame.font.Font(self.font_path, 28)
        self.font_medium = pygame.font.Font(self.font_path, 20)
        self.font_small = pygame.font.Font(self.font_path, 16)

        self.window_width = 900
        self.window_height = 700
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("多游戏AI对战平台")
        self.clock = pygame.time.Clock()

        # 游戏状态
        self.current_game = "gomoku"  # "gomoku" 或 "snake"
        self.env = None
        self.human_agent = None
        self.ai_agent = None
        self.current_agent = None
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.thinking = False
        self.selected_ai = "RandomBot"
        self.paused = False

        # UI元素
        self.buttons = self._create_buttons()
        self.cell_size = 25
        self.margin = 50

        # 游戏计时
        self.last_update = time.time()
        self.update_interval = 0.3  # 贪吃蛇更新间隔
        # === 新增：方向变量 ===
        self.player_direction = None
        self.ai_direction = None
        self.victory_length = 25  # 贪吃蛇胜利长度
        self._switch_game("gomoku")

    def _get_chinese_font(self):
        """获取中文字体路径"""
        # 尝试不同系统的中文字体
        font_paths = [
            # macOS
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            # Windows
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]

        for font_path in font_paths:
            if os.path.exists(font_path):
                return font_path

        # 如果没有找到中文字体，使用pygame默认字体
        return None

    def _create_buttons(self) -> Dict[str, Dict[str, Any]]:
        """创建UI按钮"""
        button_width = 120
        button_height = 30
        start_x = 650

        buttons = {
            # 游戏选择
            "gomoku_game": {
                "rect": pygame.Rect(start_x, 50, button_width, button_height),
                "text": "Gomoku",
                "color": COLORS["YELLOW"],
            },
            "snake_game": {
                "rect": pygame.Rect(start_x, 90, button_width, button_height),
                "text": "Snake",
                "color": COLORS["LIGHT_GRAY"],
            },
            # AI选择（全部创建，后续根据游戏类型动态显示/禁用）
            "random_ai": {
                "rect": pygame.Rect(start_x, 150, button_width, button_height),
                "text": "基础AI",
                "color": COLORS["YELLOW"],
            },
            "minimax_ai": {
                "rect": pygame.Rect(start_x, 190, button_width, button_height),
                "text": "Minimax AI",
                "color": COLORS["LIGHT_GRAY"],
            },
            "mcts_ai": {
                "rect": pygame.Rect(start_x, 230, button_width, button_height),
                "text": "MCTS AI",
                "color": COLORS["LIGHT_GRAY"],
            },
            "snake_ai": {
                "rect": pygame.Rect(start_x, 190, button_width, button_height),
                "text": "进阶AI",
                "color": COLORS["LIGHT_GRAY"],
            },
            # 控制按钮
            "new_game": {
                "rect": pygame.Rect(start_x, 290, button_width, button_height),
                "text": "New Game",
                "color": COLORS["GREEN"],
            },
            "pause": {
                "rect": pygame.Rect(start_x, 330, button_width, button_height),
                "text": "Pause",
                "color": COLORS["ORANGE"],
            },
            "quit": {
                "rect": pygame.Rect(start_x, 370, button_width, button_height),
                "text": "Quit",
                "color": COLORS["RED"],
            },
        }
        return buttons

    # AI按钮与AI类名映射，增加snake_ai
    AI_BUTTON_TO_CLASS = {
        "random_ai": "RandomBot",
        "minimax_ai": "MinimaxBot",
        "mcts_ai": "MCTSBot",
        "snake_ai": "SnakeAI"
    }
    AI_CLASS_TO_BUTTON = {v: k for k, v in AI_BUTTON_TO_CLASS.items()}

    def _switch_game(self, game_type):
        """切换游戏类型"""
        self.current_game = game_type
        # 更新游戏选择按钮颜色
        for btn_name in ["gomoku_game", "snake_game"]:
            self.buttons[btn_name]["color"] = COLORS["LIGHT_GRAY"]
        self.buttons[f"{game_type}_game"]["color"] = COLORS["YELLOW"]
        # 动态调整AI按钮可用性和显示
        if game_type == "gomoku":
            # 只允许随机、minimax、mcts
            self.buttons["random_ai"]["text"] = "基础AI"
            self.buttons["random_ai"]["color"] = COLORS["YELLOW"] if self.selected_ai == "RandomBot" else COLORS["LIGHT_GRAY"]
            self.buttons["random_ai"]["visible"] = True
            self.buttons["minimax_ai"]["color"] = COLORS["YELLOW"] if self.selected_ai == "MinimaxBot" else COLORS["LIGHT_GRAY"]
            self.buttons["minimax_ai"]["visible"] = True
            self.buttons["mcts_ai"]["color"] = COLORS["YELLOW"] if self.selected_ai == "MCTSBot" else COLORS["LIGHT_GRAY"]
            self.buttons["mcts_ai"]["visible"] = True
            self.buttons["snake_ai"]["visible"] = False
        elif game_type == "snake":
            # 只允许随机、snake ai
            self.buttons["random_ai"]["text"] = "基础AI"
            self.buttons["random_ai"]["color"] = COLORS["YELLOW"] if self.selected_ai == "RandomBot" else COLORS["LIGHT_GRAY"]
            self.buttons["random_ai"]["visible"] = True
            self.buttons["snake_ai"]["text"] = "进阶AI"
            self.buttons["snake_ai"]["color"] = COLORS["YELLOW"] if self.selected_ai == "SnakeAI" else COLORS["LIGHT_GRAY"]
            self.buttons["snake_ai"]["visible"] = True
            self.buttons["minimax_ai"]["visible"] = False
            self.buttons["mcts_ai"]["visible"] = False
        # 创建对应的环境和智能体
        if game_type == "gomoku":
            self.env = GomokuEnv(board_size=15, win_length=5)
            self.cell_size = 30
            self.update_interval = 1.0
            if self.selected_ai not in ["RandomBot", "MinimaxBot", "MCTSBot"]:
                self.selected_ai = "RandomBot"
        elif game_type == "snake":
            self.env = SnakeEnv(board_size=20)
            self.cell_size = 25
            self.update_interval = 0.2
            if self.selected_ai not in ["RandomBot", "SnakeAI"]:
                self.selected_ai = "RandomBot"
        self.human_agent = HumanAgent(name="Human Player", player_id=1)
        self._create_ai_agent()
        self.reset_game()

    def _create_ai_agent(self):
        """创建AI智能体"""
        if self.selected_ai == "RandomBot":
            self.ai_agent = RandomBot(name="基础AI", player_id=2)
        elif self.selected_ai == "MinimaxBot":
            self.ai_agent = MinimaxBot(name="Minimax AI", player_id=2, search_depth=1)
        elif self.selected_ai == "MCTSBot":
            self.ai_agent = MCTSBot(name="MCTS AI", simulation_count=50)
        elif self.selected_ai == "SnakeAI":
            self.ai_agent = SnakeAI(name="进阶AI", player_id=2)

    def reset_game(self):
        """重置游戏"""
        if self.env is not None:
            self.env.reset()
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.thinking = False
        self.current_agent = self.human_agent
        self.last_update = time.time()
        self.paused = False
        self.buttons["pause"]["text"] = "Pause"
        # === 新增：初始化方向变量 ===
        if self.current_game == "snake" and self.env is not None and hasattr(self.env, "game"):
            self.player_direction = getattr(self.env.game, "direction1", (0,1))
            self.ai_direction = getattr(self.env.game, "direction2", (0,-1))
        else:
            self.player_direction = None
            self.ai_direction = None

    def handle_events(self) -> bool:
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                # 只有未暂停时才允许玩家输入
                if (
                    self.current_game == "snake"
                    and isinstance(self.current_agent, HumanAgent)
                    and not self.game_over
                    and not self.thinking
                    and not self.paused
                ):
                    self._handle_snake_input(event.key)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    # 优化：暂停时只允许点击pause和quit按钮
                    if self.paused:
                        if self.buttons["pause"]["rect"].collidepoint(mouse_pos):
                            self.paused = not self.paused
                            self.buttons["pause"]["text"] = "Resume" if self.paused else "Pause"
                            return True
                        elif self.buttons["quit"]["rect"].collidepoint(mouse_pos):
                            return False
                        return True # 暂停时点击pause不返回True，只返回True
                    # 未暂停时正常处理
                    click_result = self._handle_button_click(mouse_pos)
                    # 只有new_game、AI按钮、切换游戏按钮才reset_game
                    if click_result is False:
                        return False
                    # 不再对pause按钮返回True时调用reset_game
                    if (
                        self.current_game == "gomoku"
                        and not self.game_over
                        and isinstance(self.current_agent, HumanAgent)
                        and not self.thinking
                        and not self.paused
                    ):
                        self._handle_gomoku_click(mouse_pos)

        return True

    def _handle_button_click(self, mouse_pos: Tuple[int, int]) -> bool:
        """处理按钮点击"""
        for button_name, button_info in self.buttons.items():
            if not button_info.get("visible", True):
                continue
            if button_info["rect"].collidepoint(mouse_pos):
                if button_name == "new_game":
                    self.reset_game()
                    return True
                elif button_name == "quit":
                    return False
                elif button_name == "pause":
                    self.paused = not self.paused
                    self.buttons["pause"]["text"] = "Resume" if self.paused else "Pause"
                    return None
                elif button_name in ["gomoku_game", "snake_game"]:
                    game_type = button_name.split("_")[0]
                    self._switch_game(game_type)
                    self.reset_game()
                    return True
                elif button_name in self.AI_BUTTON_TO_CLASS:
                    # 只允许当前可见的AI按钮
                    if not button_info.get("visible", True):
                        continue
                    old_btn = self.AI_CLASS_TO_BUTTON.get(self.selected_ai)
                    if old_btn and old_btn in self.buttons:
                        self.buttons[old_btn]["color"] = COLORS["LIGHT_GRAY"]
                    self.selected_ai = self.AI_BUTTON_TO_CLASS[button_name]
                    self.buttons[button_name]["color"] = COLORS["YELLOW"]
                    self._create_ai_agent()
                    self.reset_game()
                    return True
        return None

    def _handle_gomoku_click(self, mouse_pos: Tuple[int, int]):
        """处理五子棋棋盘点击"""
        x, y = mouse_pos
        board_x = x - self.margin
        board_y = y - self.margin

        if board_x < 0 or board_y < 0:
            return

        col = round(board_x / self.cell_size)
        row = round(board_y / self.cell_size)

        if 0 <= row < 15 and 0 <= col < 15:
            action = (row, col)
            if action in self.env.get_valid_actions():
                self._make_move(action)

    def _handle_snake_input(self, key):
        """处理贪吃蛇键盘输入"""
        key_to_action = {
            pygame.K_UP: (-1, 0),  # 上
            pygame.K_w: (-1, 0),
            pygame.K_DOWN: (1, 0),  # 下
            pygame.K_s: (1, 0),
            pygame.K_LEFT: (0, -1),  # 左
            pygame.K_a: (0, -1),
            pygame.K_RIGHT: (0, 1),  # 右
            pygame.K_d: (0, 1),
        }

        if key in key_to_action:
            action = key_to_action[key]
            self.player_direction = action  # 只更新方向，不直接移动

    def _make_move(self, action):
        """执行移动"""
        if self.game_over or self.paused:
            return

        try:
            # 执行动作
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.last_move = action

            # 检查游戏是否结束
            if terminated or truncated:
                self.game_over = True
                self.winner = self.env.get_winner()
            else:
                # 切换玩家
                self._switch_player()

        except Exception as e:
            print(f"Move execution failed: {e}")

    def _switch_player(self):
        """切换玩家"""
        if isinstance(self.current_agent, HumanAgent):
            self.current_agent = self.ai_agent
            self.thinking = True
        else:
            self.current_agent = self.human_agent

    def update_game(self):
        """更新游戏"""
        if self.game_over or self.paused:
            return
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
        self.last_update = current_time
        # === 贪吃蛇持续移动逻辑 ===
        if self.current_game == "snake" and self.env is not None and hasattr(self.env, "game"):
            # AI每帧都决策
            if getattr(self.env.game, "alive2", False):
                observation = self.env._get_observation()
                ai_action = self.ai_agent.get_action(observation, self.env)
                if ai_action is not None:
                    self.ai_direction = ai_action
            player_dir = self.player_direction if self.player_direction is not None else getattr(self.env.game, "direction1", (0,1))
            ai_dir = self.ai_direction if self.ai_direction is not None else getattr(self.env.game, "direction2", (0,-1))
            step_fn = getattr(self.env, "step", None)
            if callable(step_fn):
                result = self.env.step(player_dir, ai_dir)
                if result and len(result) >= 5:
                    observation, reward1, reward2, done, info = result[:5]
                else:
                    observation, reward1, reward2, done, info = None, 0, 0, False, {}
            else:
                observation, reward1, reward2, done, info = None, 0, 0, False, {}
            # === 胜负判定 ===
            snake1 = getattr(self.env.game, "snake1", [])
            snake2 = getattr(self.env.game, "snake2", [])
            alive1 = getattr(self.env.game, "alive1", False)
            alive2 = getattr(self.env.game, "alive2", False)
            if len(snake1) >= self.victory_length and alive1:
                self.game_over = True
                self.winner = 1
            elif len(snake2) >= self.victory_length and alive2:
                self.game_over = True
                self.winner = 2
            elif not alive2 and alive1:
                self.game_over = True
                self.winner = 1
            elif not alive1 and alive2:
                self.game_over = True
                self.winner = 2
            elif done:
                self.game_over = True
                winner = self.env.get_winner()
                self.winner = winner if winner is not None else 0
        else:
            # GomokuGame: 只需一个动作
            if self.game_over:
                return
            if not isinstance(self.current_agent, HumanAgent) and self.thinking:
                try:
                    if isinstance(self.ai_agent, MinimaxBot) and self.current_game == "gomoku":
                        action = self.ai_agent.get_action(self.env.get_board_state(), 2)
                    elif isinstance(self.ai_agent, MCTSBot) and self.current_game == "gomoku":
                        action = self.ai_agent.get_action(None, self.env)
                    else:
                        observation = self.env._get_observation()
                        action = self.current_agent.get_action(observation, self.env)
                    if action:
                        self._make_move(action)
                    self.thinking = False
                except Exception as e:
                    print(f"AI thinking failed: {e}")
                    self.thinking = False
            elif (
                self.current_game == "gomoku"
                and isinstance(self.current_agent, HumanAgent)
                and not self.thinking
            ):
                # 五子棋等待玩家输入
                pass

    def draw(self):
        """绘制游戏界面"""
        # 清空屏幕
        self.screen.fill(COLORS["WHITE"])

        # 绘制游戏区域
        if self.current_game == "gomoku":
            self._draw_gomoku()
        elif self.current_game == "snake":
            self._draw_snake()

        # 绘制UI
        self._draw_ui()

        # 绘制游戏状态
        self._draw_game_status()

        # 更新显示
        pygame.display.flip()

    def _draw_gomoku(self):
        """绘制五子棋"""
        board_size = 15

        # 绘制棋盘背景
        board_rect = pygame.Rect(
            self.margin - 20,
            self.margin - 20,
            board_size * self.cell_size + 40,
            board_size * self.cell_size + 40,
        )
        pygame.draw.rect(self.screen, COLORS["LIGHT_BROWN"], board_rect)

        # 绘制网格线
        for i in range(board_size):
            # 垂直线
            start_pos = (self.margin + i * self.cell_size, self.margin)
            end_pos = (
                self.margin + i * self.cell_size,
                self.margin + (board_size - 1) * self.cell_size,
            )
            pygame.draw.line(self.screen, COLORS["BLACK"], start_pos, end_pos, 2)

            # 水平线
            start_pos = (self.margin, self.margin + i * self.cell_size)
            end_pos = (
                self.margin + (board_size - 1) * self.cell_size,
                self.margin + i * self.cell_size,
            )
            pygame.draw.line(self.screen, COLORS["BLACK"], start_pos, end_pos, 2)

        # 绘制星位
        star_positions = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
        for row, col in star_positions:
            center = (
                self.margin + col * self.cell_size,
                self.margin + row * self.cell_size,
            )
            pygame.draw.circle(self.screen, COLORS["BLACK"], center, 4)

        # 绘制棋子
        board = self.env.game.board
        for row in range(board_size):
            for col in range(board_size):
                if board[row, col] != 0:
                    center = (
                        self.margin + col * self.cell_size,
                        self.margin + row * self.cell_size,
                    )

                    if board[row, col] == 1:  # 人类玩家
                        color = COLORS["BLACK"]
                        border_color = COLORS["WHITE"]
                    else:  # AI玩家
                        color = COLORS["WHITE"]
                        border_color = COLORS["BLACK"]

                    pygame.draw.circle(self.screen, color, center, 12)
                    pygame.draw.circle(self.screen, border_color, center, 12, 2)

        # 绘制最后一步标记
        if (
            self.last_move
            and isinstance(self.last_move, tuple)
            and len(self.last_move) == 2
        ):
            row, col = self.last_move
            center = (
                self.margin + col * self.cell_size,
                self.margin + row * self.cell_size,
            )
            pygame.draw.circle(self.screen, COLORS["RED"], center, 6, 3)

    def _draw_snake(self):
        """绘制贪吃蛇"""
        board_size = 20

        # 绘制游戏区域背景
        game_rect = pygame.Rect(
            self.margin,
            self.margin,
            board_size * self.cell_size,
            board_size * self.cell_size,
        )
        pygame.draw.rect(self.screen, COLORS["LIGHT_GRAY"], game_rect)
        pygame.draw.rect(self.screen, COLORS["BLACK"], game_rect, 2)

        # 绘制网格
        for i in range(board_size + 1):
            # 垂直线
            x = self.margin + i * self.cell_size
            pygame.draw.line(
                self.screen,
                COLORS["GRAY"],
                (x, self.margin),
                (x, self.margin + board_size * self.cell_size),
                1,
            )
            # 水平线
            y = self.margin + i * self.cell_size
            pygame.draw.line(
                self.screen,
                COLORS["GRAY"],
                (self.margin, y),
                (self.margin + board_size * self.cell_size, y),
                1,
            )

        # 绘制游戏元素
        board = self.env.game.board
        for row in range(board_size):
            for col in range(board_size):
                if board[row, col] != 0:
                    x = self.margin + col * self.cell_size + 2
                    y = self.margin + row * self.cell_size + 2
                    rect = pygame.Rect(x, y, self.cell_size - 4, self.cell_size - 4)

                    if board[row, col] == 1:  # 蛇1头部
                        pygame.draw.rect(self.screen, COLORS["BLUE"], rect)
                    elif board[row, col] == 2:  # 蛇1身体
                        pygame.draw.rect(self.screen, COLORS["CYAN"], rect)
                    elif board[row, col] == 3:  # 蛇2头部
                        pygame.draw.rect(self.screen, COLORS["RED"], rect)
                    elif board[row, col] == 4:  # 蛇2身体
                        pygame.draw.rect(self.screen, COLORS["ORANGE"], rect)
                    elif board[row, col] == 5:  # 食物
                        pygame.draw.rect(self.screen, COLORS["GREEN"], rect)

        # 绘制普通食物
        for x, y in getattr(self.env.game, 'foods', []):
            rect = pygame.Rect(self.margin + y * self.cell_size + 4, self.margin + x * self.cell_size + 4, self.cell_size - 8, self.cell_size - 8)
            pygame.draw.ellipse(self.screen, COLORS["GREEN"], rect)
        # 绘制奖励球（super food）
        for x, y in getattr(self.env.game, 'super_foods', []):
            rect = pygame.Rect(self.margin + y * self.cell_size, self.margin + x * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.ellipse(self.screen, COLORS["PURPLE"], rect)

    def _draw_ui(self):
        """绘制UI界面"""
        # 绘制按钮（只绘制visible为True的按钮）
        for button_name, button_info in self.buttons.items():
            if not button_info.get("visible", True):
                continue
            pygame.draw.rect(self.screen, button_info["color"], button_info["rect"])
            pygame.draw.rect(self.screen, COLORS["BLACK"], button_info["rect"], 2)
            text_surface = self.font_medium.render(
                button_info["text"], True, COLORS["BLACK"]
            )
            text_rect = text_surface.get_rect(center=button_info["rect"].center)
            self.screen.blit(text_surface, text_rect)
        # 绘制标题
        title_text = self.font_medium.render("游戏选择:", True, COLORS["BLACK"])
        self.screen.blit(title_text, (self.buttons["gomoku_game"]["rect"].x, 25))
        # AI选择区提示只保留“AI选择:”
        ai_title_text = self.font_medium.render("AI选择:", True, COLORS["BLACK"])
        self.screen.blit(ai_title_text, (self.buttons["random_ai"]["rect"].x, 125))
        # 绘制操作说明
        if self.current_game == "gomoku":
            instructions = [
                "五子棋操作:",
                "• 鼠标点击落子",
                "• 连成5子获胜",
            ]
        else:
            instructions = [
                "贪吃蛇操作:",
                "• 方向键/WASD控制移动",
                "• 吃食物变长",
                "• 避免碰撞",
            ]
        start_y = 420
        for i, instruction in enumerate(instructions):
            text = self.font_small.render(instruction, True, COLORS["DARK_GRAY"])
            self.screen.blit(
                text, (self.buttons["new_game"]["rect"].x, start_y + i * 20)
            )

    def _draw_game_status(self):
        """绘制游戏状态"""
        status_x = 20
        status_y = self.window_height - 100

        if self.paused:
            status_text = "Game Paused..."
            color = COLORS["ORANGE"]
        elif self.game_over:
            if self.winner == 1:
                status_text = "Congratulations! You Win!"
                color = COLORS["GREEN"]
            elif self.winner == 2:
                status_text = "AI Wins! Try Again!"
                color = COLORS["RED"]
            else:
                status_text = "Draw!"
                color = COLORS["ORANGE"]
        else:
            if isinstance(self.current_agent, HumanAgent):
                if self.current_game == "gomoku":
                    status_text = "Your Turn - Click to Place Stone"
                else:
                    status_text = "Your Turn - Use Arrow Keys"
                color = COLORS["BLUE"]
            else:
                if self.thinking:
                    status_text = f"{self.ai_agent.name} is Thinking..."
                    color = COLORS["ORANGE"]
                else:
                    status_text = f"{self.ai_agent.name}'s Turn"
                    color = COLORS["RED"]

        text_surface = self.font_large.render(status_text, True, color)
        self.screen.blit(text_surface, (status_x, status_y))

        # 游戏信息
        info_y = status_y + 40
        if self.current_game == "gomoku":
            player_info = f"Black: Human Player  White: {self.ai_agent.name if self.ai_agent else 'AI'}"
        else:
            if hasattr(self.env.game, "snake1") and hasattr(self.env.game, "snake2"):
                len1 = len(self.env.game.snake1) if self.env.game.alive1 else 0
                len2 = len(self.env.game.snake2) if self.env.game.alive2 else 0
                alive1 = "Alive" if self.env.game.alive1 else "Dead"
                alive2 = "Alive" if self.env.game.alive2 else "Dead"
                player_info = f"Blue Snake(You): {len1} segments({alive1})  Red Snake(AI): {len2} segments({alive2})"
            else:
                player_info = "Snake Battle in Progress..."

        info_surface = self.font_small.render(player_info, True, COLORS["DARK_GRAY"])
        self.screen.blit(info_surface, (status_x, info_y))

        # UI右下角胜利条件提示
        if self.current_game == "snake":
            victory_text = f"胜利条件：蛇长度达到{self.victory_length}或对方死亡"
            victory_surface = self.font_small.render(victory_text, True, COLORS["BLACK"])
            self.screen.blit(victory_surface, (self.window_width - 320, self.window_height - 60))
            # 新增奖励球提示
            super_food_text = "奖励球限时出现，长度+3"
            super_food_surface = self.font_small.render(super_food_text, True, COLORS["PURPLE"])
            self.screen.blit(super_food_surface, (self.window_width - 320, self.window_height - 40))

    def run(self):
        """运行游戏主循环"""
        running = True

        while running:
            # 处理事件
            running = self.handle_events()

            # 更新游戏
            self.update_game()

            # 绘制界面
            self.draw()

            # 控制帧率
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


def main():
    """主函数"""
    print("Starting Multi-Game AI Battle Platform...")
    print("Supported Games:")
    print("- Gomoku: Click to place stones")
    print("- Snake: Arrow keys/WASD to control")
    print("- Multiple AI difficulty levels")
    print("- Real-time human vs AI battles")

    try:
        game = MultiGameGUI()
        game.run()
    except Exception as e:
        print(f"Game error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

