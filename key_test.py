import pygame
import sys

# 初始化 Pygame
pygame.init() #需要开大写

# 设置窗口
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Pygame 按键测试")

# 设置字体
font = pygame.font.Font(None, 36)  # 默认字体，字号36

# 主循环
running = True
last_key_pressed = "暂无按键"

while running:
    screen.fill((0, 0, 0))  # 清屏（黑色背景）
    
    # 显示提示文字
    text = font.render(f"按下的键: {last_key_pressed}", True, (255, 255, 255))
    screen.blit(text, (50, 50))
    
    # 显示退出提示
    exit_text = font.render("按 ESC 键退出", True, (255, 255, 255))
    screen.blit(exit_text, (50, 100))
    
    # 事件处理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            key_name = pygame.key.name(event.key)
            last_key_pressed = key_name  # 更新最后按下的键
            print(f"按下键: {key_name}")  # 控制台输出
            
            if event.key == pygame.K_ESCAPE:  # ESC 退出
                running = False
    
    pygame.display.flip()  # 更新屏幕显示

# 退出 Pygame
pygame.quit()
sys.exit()