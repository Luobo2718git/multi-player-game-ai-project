"""
项目测试文件
验证双人游戏AI框架的基本功能
"""

import sys
import traceback
from typing import Dict, List, Any


def test_imports():
    """测试模块导入"""
    print("=== 测试模块导入 ===")
    
    try:
        # 测试游戏模块
        from games.gomoku import GomokuGame, GomokuEnv
        print("✓ 游戏模块导入成功")
        
        # 测试智能体模块
        from agents import HumanAgent, RandomBot, MinimaxBot, MCTSBot, RLBot, BehaviorTreeBot
        print("✓ 智能体模块导入成功")
        
        # 测试工具模块
        from utils.game_utils import evaluate_agents
        print("✓ 工具模块导入成功")
        
        # 测试配置模块
        import config
        print("✓ 配置模块导入成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        traceback.print_exc()
        return False


def test_gomoku_game():
    """测试五子棋游戏"""
    print("\n=== 测试五子棋游戏 ===")
    
    try:
        from games.gomoku import GomokuGame
        
        # 创建游戏
        game = GomokuGame(board_size=9, win_length=5)
        print("✓ 游戏创建成功")
        
        # 测试重置
        state = game.reset()
        print("✓ 游戏重置成功")
        
        # 测试有效动作
        valid_actions = game.get_valid_actions()
        print(f"✓ 有效动作数量: {len(valid_actions)}")
        
        # 测试动作执行
        action = valid_actions[0]
        observation, reward, done, info = game.step(action)
        print("✓ 动作执行成功")
        
        # 测试状态获取
        state = game.get_state()
        print("✓ 状态获取成功")
        
        # 测试克隆
        cloned_game = game.clone()
        print("✓ 游戏克隆成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 五子棋游戏测试失败: {e}")
        traceback.print_exc()
        return False


def test_gomoku_env():
    """测试五子棋环境"""
    print("\n=== 测试五子棋环境 ===")
    
    try:
        from games.gomoku import GomokuEnv
        
        # 创建环境
        env = GomokuEnv(board_size=9, win_length=5)
        print("✓ 环境创建成功")
        
        # 测试重置
        observation, info = env.reset()
        print("✓ 环境重置成功")
        
        # 测试有效动作
        valid_actions = env.get_valid_actions()
        print(f"✓ 有效动作数量: {len(valid_actions)}")
        
        # 测试动作执行
        action = valid_actions[0]
        observation, reward, terminated, truncated, info = env.step(action)
        print("✓ 环境动作执行成功")
        
        # 测试渲染
        env.render(mode='human')
        print("✓ 环境渲染成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 五子棋环境测试失败: {e}")
        traceback.print_exc()
        return False


def test_agents():
    """测试智能体"""
    print("\n=== 测试智能体 ===")
    
    try:
        from agents import RandomBot, MinimaxBot, MCTSBot
        from games.gomoku import GomokuEnv
        
        # 创建环境
        env = GomokuEnv(board_size=9, win_length=5)
        observation, info = env.reset()
        
        # 测试随机Bot
        random_bot = RandomBot(name="测试随机Bot", player_id=1)
        action = random_bot.get_action(observation, env)
        print("✓ 随机Bot测试成功")
        
        # 测试Minimax Bot
        minimax_bot = MinimaxBot(name="测试MinimaxBot", player_id=2, max_depth=2)
        action = minimax_bot.get_action(env.get_board_state(), env.game.current_player)
        print("✓ Minimax Bot测试成功")
        
        # 测试MCTS Bot
        mcts_bot = MCTSBot(name="测试MCTSBot", player_id=3, simulation_count=100)
        action = mcts_bot.get_action(observation, env)
        print("✓ MCTS Bot测试成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 智能体测试失败: {e}")
        traceback.print_exc()
        return False


def test_game_play():
    """测试游戏对战"""
    print("\n=== 测试游戏对战 ===")
    
    try:
        from games.gomoku import GomokuEnv
        from agents import RandomBot, MinimaxBot
        
        # 创建环境和智能体
        env = GomokuEnv(board_size=9, win_length=5)
        agent1 = RandomBot(name="随机Bot", player_id=1)
        agent2 = MinimaxBot(name="MinimaxBot", player_id=2, max_depth=2)
        
        # 进行游戏
        observation, info = env.reset()
        agents = {1: agent1, 2: agent2}
        step_count = 0
        
        while not env.is_terminal() and step_count < 20:
            current_agent = agents[env.game.current_player]
            if current_agent.name == 'MinimaxBot':
                action = current_agent.get_action(env.get_board_state(), env.game.current_player)
            else:
                action = current_agent.get_action(observation, env)
            
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            if terminated or truncated:
                break
        
        winner = env.get_winner()
        print(f"✓ 游戏对战完成，获胜者: {winner}，步数: {step_count}")
        
        return True
        
    except Exception as e:
        print(f"✗ 游戏对战测试失败: {e}")
        traceback.print_exc()
        return False


def test_evaluation():
    """测试智能体评估"""
    print("\n=== 测试智能体评估 ===")
    
    try:
        from games.gomoku import GomokuEnv
        from agents import RandomBot, MinimaxBot
        from utils.game_utils import evaluate_agents
        
        # 创建环境和智能体
        env = GomokuEnv(board_size=9, win_length=5)
        agent1 = RandomBot(name="随机Bot", player_id=1)
        agent2 = MinimaxBot(name="MinimaxBot", player_id=2, max_depth=2)
        
        # 评估智能体
        results = evaluate_agents(env, agent1, agent2, num_games=5, save_results=False)
        
        print(f"✓ 智能体评估完成")
        print(f"  {agent1.name} 胜率: {results['summary']['agent1_win_rate']:.2%}")
        print(f"  {agent2.name} 胜率: {results['summary']['agent2_win_rate']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"✗ 智能体评估测试失败: {e}")
        traceback.print_exc()
        return False


def test_custom_agents():
    """测试自定义智能体"""
    print("\n=== 测试自定义智能体 ===")
    
    try:
        from examples.custom_agent import GreedyBot, DefensiveBot, PatternBot
        from games.gomoku import GomokuEnv
        
        # 创建环境
        env = GomokuEnv(board_size=9, win_length=5)
        observation, info = env.reset()
        
        # 测试自定义智能体
        greedy_bot = GreedyBot(name="贪心Bot", player_id=1)
        defensive_bot = DefensiveBot(name="防守Bot", player_id=2)
        pattern_bot = PatternBot(name="模式Bot", player_id=3)
        
        # 测试动作选择
        action1 = greedy_bot.get_action(observation, env)
        action2 = defensive_bot.get_action(observation, env)
        action3 = pattern_bot.get_action(observation, env)
        
        print("✓ 自定义智能体测试成功")
        print(f"  贪心Bot动作: {action1}")
        print(f"  防守Bot动作: {action2}")
        print(f"  模式Bot动作: {action3}")
        
        return True
        
    except Exception as e:
        print(f"✗ 自定义智能体测试失败: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("双人游戏AI框架 - 项目测试")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_gomoku_game,
        test_gomoku_env,
        test_agents,
        test_game_play,
        test_evaluation,
        test_custom_agents
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ 测试 {test.__name__} 出现异常: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！项目运行正常。")
        return True
    else:
        print("❌ 部分测试失败，请检查错误信息。")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 