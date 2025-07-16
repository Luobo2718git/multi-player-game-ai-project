import pygame
import sys
import time
import os
from typing import Optional, Tuple, Dict, Any, Union
# 只导入泡泡堂游戏和相关智能体
from games.bomb import BombGame, BombEnv
from agents import bombHumanAgent, BombAI

# 颜色定义
COLORS = {
    'WHITE': (220, 228, 235),  # 柔和白
    'BLACK': (30, 30, 30),     # 柔和黑
    'RED': (235, 172, 183),    # 柔和红
    'BLUE': (135, 206, 235),   # 天蓝色 (修改)
    'GREEN': (124, 218, 124),  # 柔和绿
    'GRAY': (172, 172, 172),   # 柔和灰
    'LIGHT_GRAY': (200, 200, 200), # 更浅的柔和灰
    'DARK_GRAY': (85, 85, 85),  # 更深的柔和灰
    'YELLOW': (235, 235, 133), # 柔和黄
    'ORANGE': (235, 203, 166), # 柔和橙
    'CYAN': (155, 218, 218),   # 柔和青
    'BROWN': (190, 160, 120),  # 柔和棕
    'PURPLE': (201, 140, 201), # 柔和紫
    'GOLD': (235, 198, 165),   # 作为护盾色，与柔和橙相似
    'DARK_GREEN': (132, 231, 132) # 柔和深绿
}
class GameGUI:
    """泡泡堂游戏图形界面"""
    
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("泡泡堂 AI 对战")

        self.board_size = 19 # 泡泡堂棋盘大小
        self.cell_size = 40 # 调整单元格大小以提高可见性

        self.font_path = self._get_chinese_font()
      
        self.font_large = pygame.font.Font(self.font_path, 28)
        self.font_medium = pygame.font.Font(self.font_path, 20)
        self.font_small = pygame.font.Font(self.font_path, 16)

        # 定义顶部信息栏的高度
        self.header_height = 50 

        self.screen_width = self.board_size * self.cell_size
        # 屏幕高度 = 顶部信息栏 + 棋盘高度 + 底部信息栏高度
        self.screen_height = self.header_height + self.board_size * self.cell_size + 100 
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        self.game_type: str = "bomb" # 游戏类型固定为泡泡堂
        self.env: BombEnv = BombEnv(board_size=self.board_size)
        self.player1_agent = bombHumanAgent(player_id=1) # 玩家1始终是人类
        self.player2_agent: BombAI = BombAI(player_id=2) # 玩家2是泡泡堂AI
        self.game_active = False # 标志，指示游戏是否活跃

        # 游戏计时器变量
        self.game_duration = 60 # 秒
        self.game_start_time: float = 0.0

        # 用于跟踪按下按键以实现连续移动的字典
        self.pressed_keys: Dict[int, bool] = {}
        # 将Pygame按键常量映射到移动/动作元组
        self.key_to_action_map = {
            pygame.K_UP: (-1, 0),
            pygame.K_w: (-1, 0),
            pygame.K_DOWN: (1, 0),
            pygame.K_s: (1, 0),
            pygame.K_LEFT: (0, -1),
            pygame.K_a: (0, -1),
            pygame.K_RIGHT: (0, 1),
            pygame.K_d: (0, 1),
            pygame.K_SPACE: (0, 0, True), # 放置炸弹动作
        }

        self._initialize_game() # 直接初始化游戏

    def _initialize_game(self):
        """初始化泡泡堂游戏环境和AI"""
        self.env = BombEnv(board_size=self.board_size)
        self.player2_agent = BombAI(player_id=2) # 泡泡堂AI
        pygame.display.set_caption("泡泡堂 AI 对战")
        
        # 重置屏幕大小（已在__init__中设置，但为了保持一致性）
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        
        self.game_active = True
        self.observation = self.env.reset()
        self.player1_agent.reset() # 重置人类玩家的动作状态
        self.game_start_time = time.time() # 在初始化时重置游戏开始时间
        self.pressed_keys = {} # 清除按下按键的状态

    def handle_events(self):
        """处理Pygame事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: # 添加ESC键退出
                    return False
                
                # 如果游戏不活跃 (即游戏结束时)，只处理 'R' 和 'ESC' 键
                if not self.game_active:
                    if event.key == pygame.K_r: # 重置游戏
                        self._initialize_game() # 重新初始化当前游戏
                        return True # 游戏已重置，继续运行
                    elif event.key == pygame.K_ESCAPE: # 退出游戏
                        return False # 退出游戏循环
                
                # 游戏活跃时，处理其他按键
                if self.game_active and event.key in self.key_to_action_map:
                    self.pressed_keys[event.key] = True
            
            if event.type == pygame.KEYUP:
                # 从pressed_keys中移除松开的按键
                if event.key in self.key_to_action_map:
                    self.pressed_keys[event.key] = False # 标记为未按下

        return True
    
    def _get_chinese_font(self):
        """获取中文字体路径"""
        # 尝试不同系统的中文字体
        # font_paths = [
        #     # macOS
        #     "/System/Library/Fonts/PingFang.ttc",
        #     "/System/Library/Fonts/Helvetica.ttc",
        #     "/Library/Fonts/Arial Unicode.ttf",
        #     # Windows 
        #     "C:/Windows/Fonts/msyh.ttc",
        #     "C:/Windows/Fonts/simhei.ttf",
        #     "C:/Windows/Fonts/simsun.ttc",
           
        #     # Linux
        #     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        #     "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # ]

        # for font_path in font_paths:
        #     if os.path.exists(font_path):
        #         print(f"找到中文字体: {font_path}")
        #         return font_path
            

        # # 如果没有找到中文字体，使用pygame默认字体
        # return None
        return "Fonts\Font1.ttf"

    def update_game(self):
        """更新游戏状态"""
        if not self.game_active or not self.env:
            return

        # 检查游戏计时器
        elapsed_time = time.time() - self.game_start_time
        if elapsed_time >= self.game_duration:
            print("游戏时间到！")
            self.game_active = False # 结束游戏
            # 当时间到时，强制触发终止状态以进行分数比较
            # 这确保了BombGame的get_winner方法会根据分数决定胜者
            self.env.game.current_moves = self.env.game.max_moves # 强制max_moves以触发终止状态
            return # 如果时间到，停止更新

        # 根据当前按下的按键确定玩家1的动作
        action1: Union[Tuple[int, int], Tuple[int, int, bool]] = (0, 0) # 默认不操作

        # 如果空格键按下，优先放置炸弹
        if self.pressed_keys.get(pygame.K_SPACE, False):
            action1 = self.key_to_action_map[pygame.K_SPACE]
        else:
            # 以特定顺序检查移动键（例如：上、下、左、右）
            # 这确保了在同时按下多个移动键时行为一致
            for key in [pygame.K_UP, pygame.K_w, pygame.K_DOWN, pygame.K_s, pygame.K_LEFT, pygame.K_a, pygame.K_RIGHT, pygame.K_d]:
                if self.pressed_keys.get(key, False):
                    action1 = self.key_to_action_map[key]
                    break # 采取检测到的第一个移动键

        # 确保选择的动作有效
        if action1 not in self.env.get_valid_actions(self.player1_agent.player_id):
            action1 = (0, 0) # 如果无效，则恢复为不操作

        self.player1_agent.set_action(action1) # 设置人类玩家的动作

        # 获取玩家2的动作（AI）
        action2 = self.player2_agent.get_action(self.observation, self.env)

        # 执行游戏步骤
        self.observation, reward1, reward2, terminal, info = self.env.step(action1, action2)

        if terminal:
            self.game_active = False # 游戏结束

    def _draw_message_box(self, messages: list[str], text_color: Tuple[int, int, int] = COLORS['BLACK']):
        """
        绘制一个居中的消息弹窗。
        Args:
            messages: 要显示的文本行列表。
            text_color: 文本颜色。
        """
        # 半透明背景
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # 黑色，透明度180 (0-255)
        self.screen.blit(overlay, (0, 0))

        # 消息框的尺寸和位置
        box_width = self.screen_width * 0.7
        box_height_per_line = self.font_medium.get_height() + 10 # 每行文本的高度 + 间距
        box_height = len(messages) * box_height_per_line + 40 # 总高度，加上上下边距
        
        box_x = (self.screen_width - box_width) // 2
        box_y = (self.screen_height - box_height) // 2
        message_box_rect = pygame.Rect(box_x, box_y, box_width, box_height)

        pygame.draw.rect(self.screen, COLORS['LIGHT_GRAY'], message_box_rect, border_radius=10) # 弹窗背景
        pygame.draw.rect(self.screen, COLORS['DARK_GRAY'], message_box_rect, 3, border_radius=10) # 弹窗边框

        # 绘制消息文本
        current_y = message_box_rect.top + 20
        for msg in messages:
            text_surface = self.font_medium.render(msg, True, text_color)
            text_rect = text_surface.get_rect(center=(message_box_rect.centerx, current_y))
            self.screen.blit(text_surface, text_rect)
            current_y += box_height_per_line


    def draw(self):
        """绘制游戏界面"""
        self.screen.fill(COLORS['WHITE']) # 清空屏幕

        if not self.env:
            return

        # 显示游戏计时器 (挪至游戏面板上方)
        # 修复：游戏结束后倒计时显示为0
        display_time = max(0, self.game_duration - int(time.time() - self.game_start_time)) if self.game_active else 0
        timer_text = self.font_medium.render(f"倒计时: {display_time} 秒", True, COLORS['BLACK'])
        timer_rect = timer_text.get_rect(center=(self.screen_width // 2, self.header_height // 2)) # 放置在顶部信息栏中央
        self.screen.blit(timer_text, timer_rect)

        board_state = self.observation['board']
        
        # 绘制棋盘
        for r in range(self.board_size):
            for c in range(self.board_size):
                # 调整棋盘元素的Y坐标
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size + self.header_height, self.cell_size, self.cell_size)
                cell_value = board_state[r, c]

                # 绘制方块和背景
                if cell_value == self.env.game.WALL:
                    pygame.draw.rect(self.screen, COLORS['DARK_GRAY'], rect)
                elif cell_value == self.env.game.DESTRUCTIBLE_BLOCK:
                    pygame.draw.rect(self.screen, COLORS['BROWN'], rect)
                elif cell_value >= self.env.game.BOMB_START_ID and cell_value < self.env.game.EXPLOSION:
                    # 炸弹 - 现在绘制为圆形
                    pygame.draw.circle(self.screen, COLORS['GRAY'], rect.center, self.cell_size // 2 - 2) # 绘制圆形
                    bomb_timer = cell_value - self.env.game.BOMB_START_ID
                    timer_text = self.font_small.render(str(bomb_timer), True, COLORS['WHITE'])
                    text_rect = timer_text.get_rect(center=rect.center)
                    self.screen.blit(timer_text, text_rect)
                elif cell_value == self.env.game.EXPLOSION:
                    pygame.draw.rect(self.screen, COLORS['ORANGE'], rect) # 爆炸
                elif cell_value == self.env.game.ITEM_BOMB_UP:
                    pygame.draw.rect(self.screen, COLORS['CYAN'], rect) # 炸弹增强物品
                    item_text = self.font_small.render("B+", True, COLORS['BLACK'])
                    text_rect = item_text.get_rect(center=rect.center)
                    self.screen.blit(item_text, text_rect)
                elif cell_value == self.env.game.ITEM_RANGE_UP:
                    pygame.draw.rect(self.screen, COLORS['YELLOW'], rect) # 范围增强物品
                    item_text = self.font_small.render("R+", True, COLORS['BLACK'])
                    text_rect = item_text.get_rect(center=rect.center)
                    self.screen.blit(item_text, text_rect)
                elif cell_value == self.env.game.ITEM_SHIELD: # 绘制护盾物品
                    pygame.draw.rect(self.screen, COLORS['GOLD'], rect)
                    item_text = self.font_small.render("S", True, COLORS['BLACK'])
                    text_rect = item_text.get_rect(center=rect.center)
                    self.screen.blit(item_text, text_rect)
                elif cell_value == self.env.game.ITEM_SQUARE_RANGE_UP: # 绘制方形爆炸物品
                    pygame.draw.rect(self.screen, COLORS['DARK_GREEN'], rect)
                    item_text = self.font_small.render("SQ", True, COLORS['BLACK'])
                    text_rect = item_text.get_rect(center=rect.center)
                    self.screen.blit(item_text, text_rect)
                else:
                    pygame.draw.rect(self.screen, COLORS['LIGHT_GRAY'], rect, 1) # 网格线

                # 绘制玩家（在其他背景之上，但炸弹和爆炸有自己的绘制逻辑）
                # 玩家绘制在顶层
                # 绘制玩家
                if self.observation['alive1']:
                    # 调整玩家矩形的Y坐标
                    p1_rect = pygame.Rect(self.observation['player1_pos'][1] * self.cell_size, self.observation['player1_pos'][0] * self.cell_size + self.header_height, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, COLORS['BLUE'], p1_rect) # 玩家1
                    # 绘制眼睛或小标记以区分
                    pygame.draw.circle(self.screen, COLORS['BLACK'], (p1_rect.centerx - self.cell_size // 6, p1_rect.centery - self.cell_size // 6), self.cell_size // 10)
                    pygame.draw.circle(self.screen, COLORS['BLACK'], (p1_rect.centerx + self.cell_size // 6, p1_rect.centery - self.cell_size // 6), self.cell_size // 10)
                    # 绘制护盾效果
                    if self.observation['player1_shield_active']:
                        pygame.draw.circle(self.screen, COLORS['GOLD'], p1_rect.center, self.cell_size // 2, 3) # 护盾光环

                if self.observation['alive2']:
                    # 调整玩家矩形的Y坐标
                    p2_rect = pygame.Rect(self.observation['player2_pos'][1] * self.cell_size, self.observation['player2_pos'][0] * self.cell_size + self.header_height, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, COLORS['RED'], p2_rect) # 玩家2
                    # 绘制眼睛或小标记以区分
                    pygame.draw.circle(self.screen, COLORS['WHITE'], (p2_rect.centerx - self.cell_size // 6, p2_rect.centery - self.cell_size // 6), self.cell_size // 10)
                    pygame.draw.circle(self.screen, COLORS['WHITE'], (p2_rect.centerx + self.cell_size // 6, p2_rect.centery - self.cell_size // 6), self.cell_size // 10)
                    # 绘制护盾效果
                    if self.observation['player2_shield_active']:
                        pygame.draw.circle(self.screen, COLORS['GOLD'], p2_rect.center, self.cell_size // 2, 3) # 护盾光环

                
                # pos = (r, c)
                # if self.player2_agent.escape_path != None:
                #     if pos in self.player2_agent.escape_path:
                #         pygame.draw.circle(self.screen, COLORS['DARK_GREEN'], rect.center, self.cell_size // 2 - 2)

        # 绘制信息面板 (修改Y坐标以适应顶部信息栏)
        info_y = self.header_height + self.board_size * self.cell_size + 20
        start_x = 20

        # 泡泡堂特定信息
        bombs1 = self.observation['player1_bombs_max']
        range1 = self.observation['player1_range']
        shield1 = f"护盾: {'是' if self.observation['player1_shield_active'] else '否'}"
        range_type1 = f"范围: {'方形' if self.observation['player1_range_type'] == 'square' else '十字'}"
        score1 = self.observation['player1_score'] # 获取玩家1分数

        bombs2 = self.observation['player2_bombs_max']
        range2 = self.observation['player2_range']
        shield2 = f"护盾: {'是' if self.observation['player2_shield_active'] else '否'}"
        range_type2 = f"范围: {'方形' if self.observation['player2_range_type'] == 'square' else '十字'}"
        score2 = self.observation['player2_score'] # 获取玩家2分数

        player_info = f"  Player1        :  炸弹数量: {bombs1}  范围: {range1}  类型：{range_type1} {shield1} 分数: {score1} "
        ai_info = f"  Player2(AI):  炸弹数量: {bombs2}  范围: {range2}  类型：{range_type2} {shield2} 分数: {score2} "

        info_surface = self.font_medium.render(player_info, True, COLORS['BLUE'])
        self.screen.blit(info_surface, (start_x, info_y))
        info_surface2 = self.font_medium.render(ai_info, True, COLORS['RED'])
        self.screen.blit(info_surface2, (start_x, info_y +30))


        # 游戏结束信息弹窗
        if not self.game_active and self.env and self.env.is_terminal():
            winner = self.env.get_winner()
            game_over_messages = []
            if winner == 1:
                game_over_messages.append("玩家 1 获胜！")
            elif winner == 2:
                game_over_messages.append("玩家 2 获胜！")
            else:
                game_over_messages.append("平局！")
            
            game_over_messages.append(f"最终分数:")
            game_over_messages.append(f"玩家1: {self.observation['player1_score']}")
            game_over_messages.append(f"玩家2: {self.observation['player2_score']}")
            game_over_messages.append("按 'R' 重玩")
            game_over_messages.append("按 'ESC' 退出") # 添加退出提示
            
            self._draw_message_box(game_over_messages)


        pygame.display.flip()

    def run(self):
        """运行游戏主循环"""
        running = True
        
        while running:
            # 处理事件
            running = self.handle_events()
            
            # 只有当游戏活跃时才更新游戏状态
            if self.game_active:
                self.update_game()
            
            # 绘制界面
            self.draw()
            
            # 控制帧率
            self.clock.tick(10) # 泡泡堂逻辑更复杂，所以帧率稍低

        pygame.quit()
        sys.exit()

def main():
    """主函数"""
    print("启动泡泡堂 AI 对战...")
    print("控制说明:")
    print("- 使用方向键或 WASD 控制你的玩家")
    print("- 按 空格键 放置炸弹")
    print("- 按 'R' 键重置当前游戏")
    print("- 按 'ESC' 键退出游戏")
    
    gui = GameGUI()
    gui.run()

if __name__ == '__main__':
    # 要运行此文件，需要config.py，假设它包含
    # GAME_CONFIGS = {
    #     'bomb': {'timeout': 300, 'max_moves': 1000}
    # }
    # REWARDS = {
    #     'win': 100, 'lose': -100, 'draw': 0, 'alive_step': 0.1, 'death': -50
    # }
    # 或者直接在GUI中定义这些常量
    # 这里假设config存在，否则会引发ModuleNotFoundError
    try:
        import config
    except ImportError:
        print("警告: config.py 未找到。使用默认游戏配置。")
        class Config:
            GAME_CONFIGS = {
                'bomb': {'timeout': 300, 'max_moves': 1000}
            }
            REWARDS = {
                'win': 100, 'lose': -100, 'draw': 0, 'alive_step': 0.1, 'death': -50
            }
        config = Config()

    main()
