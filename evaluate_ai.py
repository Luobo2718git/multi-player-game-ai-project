#!/usr/bin/env python3
"""
AI性能评估脚本
用于测试和比较不同AI算法的性能
"""

import argparse
import time
import json
import os
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

# 导入游戏和智能体
from games.gomoku import GomokuEnv
from games.snake import SnakeEnv
from agents import RandomBot, MinimaxBot, MCTSBot, RLBot, BehaviorTreeBot
from utils.game_utils import evaluate_agents, tournament


def create_agent(agent_type: str, player_id: int, name: str = None, **kwargs):
    """创建智能体"""
    if name is None:
        name = f"{agent_type}_{player_id}"
    
    agent_map = {
        'random': RandomBot,
        'minimax': MinimaxBot,
        'mcts': MCTSBot,
        'rl': RLBot,
        'behavior_tree': BehaviorTreeBot,
        # 添加基础AI类型
        'improved_random': lambda **k: __import__('examples.simple_ai_examples', fromlist=['ImprovedRandomBot']).ImprovedRandomBot(**k),
        'rule_based': lambda **k: __import__('examples.simple_ai_examples', fromlist=['RuleBasedGomokuBot']).RuleBasedGomokuBot(**k),
        'greedy_snake': lambda **k: __import__('examples.simple_ai_examples', fromlist=['GreedySnakeBot']).GreedySnakeBot(**k),
        'search_based': lambda **k: __import__('examples.simple_ai_examples', fromlist=['SearchBasedBot']).SearchBasedBot(**k),
        # 进阶AI类型（需要学生实现）
        # 'q_learning': lambda **k: __import__('examples.advanced_ai_examples', fromlist=['QLearningBot']).QLearningBot(**k),
        # 'llm_bot': lambda **k: __import__('examples.advanced_ai_examples', fromlist=['LLMBot']).LLMBot(**k)
    }
    
    if agent_type not in agent_map:
        raise ValueError(f"不支持的智能体类型: {agent_type}")
    
    return agent_map[agent_type](name=name, player_id=player_id, **kwargs)


def create_environment(game_type: str, **kwargs):
    """创建游戏环境"""
    env_map = {
        'gomoku': GomokuEnv,
        'snake': SnakeEnv
    }
    
    if game_type not in env_map:
        raise ValueError(f"不支持的游戏类型: {game_type}")
    
    return env_map[game_type](**kwargs)


def benchmark_single_agent(env, agent, num_games=100, opponent_type='random'):
    """对单个智能体进行基准测试"""
    print(f"\n=== {agent.name} 基准测试 ===")
    
    # 创建对手
    opponent = create_agent(opponent_type, 2, f"{opponent_type}_opponent")
    
    # 记录统计信息
    stats = {
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'total_time': 0,
        'total_moves': 0,
        'move_times': [],
        'game_lengths': []
    }
    
    for game_num in range(num_games):
        # 重置环境
        observation, info = env.reset()
        
        # 随机选择先手
        if game_num % 2 == 0:
            players = {1: agent, 2: opponent}
        else:
            players = {1: opponent, 2: agent}
        
        game_start_time = time.time()
        move_count = 0
        agent_move_times = []
        
        # 游戏循环
        while not env.is_terminal() and move_count < 1000:
            current_player = env.game.current_player
            current_agent = players[current_player]
            
            # 记录智能体思考时间
            if current_agent == agent:
                move_start_time = time.time()
                # print(current_agent.name)
                if 'minimax' in current_agent.name:
                    action = current_agent.get_action(env.get_board_state(), env.game.current_player)
                else:
                    action = current_agent.get_action(observation, env)
                move_time = time.time() - move_start_time
                agent_move_times.append(move_time)
            else:
                if 'minimax' in current_agent.name:
                    action = current_agent.get_action(env.get_board_state(), env.game.current_player)
                else:
                    action = current_agent.get_action(observation, env)
            
            if action is None:
                break
            
            observation, reward, terminated, truncated, step_info = env.step(action)
            move_count += 1
            
            if terminated or truncated:
                break
        
        # 统计结果
        game_time = time.time() - game_start_time
        winner = env.get_winner()
        
        if winner == (1 if game_num % 2 == 0 else 2):
            stats['wins'] += 1
        elif winner == (2 if game_num % 2 == 0 else 1):
            stats['losses'] += 1
        else:
            stats['draws'] += 1
        
        stats['total_time'] += game_time
        stats['total_moves'] += len(agent_move_times)
        stats['move_times'].extend(agent_move_times)
        stats['game_lengths'].append(move_count)
        
        # 显示进度
        if (game_num + 1) % (num_games // 10) == 0:
            print(f"进度: {game_num + 1}/{num_games}")
    
    # 计算最终统计
    stats['win_rate'] = stats['wins'] / num_games
    stats['loss_rate'] = stats['losses'] / num_games
    stats['draw_rate'] = stats['draws'] / num_games
    stats['avg_move_time'] = np.mean(stats['move_times']) if stats['move_times'] else 0
    stats['avg_game_length'] = np.mean(stats['game_lengths'])
    stats['move_time_std'] = np.std(stats['move_times']) if stats['move_times'] else 0
    
    return stats


def compare_agents(env, agent_types, num_games=50, **agent_kwargs):
    """比较多个智能体的性能"""
    print(f"\n=== 智能体比较 (每对 {num_games} 局) ===")
    
    # 创建智能体
    agents = []
    for i, agent_type in enumerate(agent_types):
        kwargs = agent_kwargs.get(agent_type, {})
        agent = create_agent(agent_type, i + 1, **kwargs)
        agents.append(agent)
        print(f"创建智能体: {agent.name}")
    
    # 运行锦标赛
    results = tournament(env, agents, num_games)
    
    return results


def analyze_performance(stats_list, agent_names):
    """分析性能统计"""
    print("\n=== 性能分析 ===")
    
    # 创建比较表格
    metrics = ['win_rate', 'avg_move_time', 'avg_game_length', 'move_time_std']
    
    print(f"{'智能体':<15} {'胜率':<8} {'平均思考时间':<12} {'平均游戏长度':<12} {'时间标准差':<10}")
    print("-" * 70)
    
    for i, (stats, name) in enumerate(zip(stats_list, agent_names)):
        print(f"{name:<15} {stats['win_rate']:<8.2%} "
              f"{stats['avg_move_time']:<12.3f} "
              f"{stats['avg_game_length']:<12.1f} "
              f"{stats['move_time_std']:<10.3f}")


def plot_performance(stats_list, agent_names, save_path=None):
    """绘制性能图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('AI智能体性能比较', fontsize=16)
    
    # 胜率比较
    win_rates = [stats['win_rate'] for stats in stats_list]
    axes[0, 0].bar(agent_names, win_rates)
    axes[0, 0].set_title('胜率比较')
    axes[0, 0].set_ylabel('胜率')
    axes[0, 0].set_ylim(0, 1)
    
    # 思考时间比较
    avg_times = [stats['avg_move_time'] for stats in stats_list]
    axes[0, 1].bar(agent_names, avg_times)
    axes[0, 1].set_title('平均思考时间')
    axes[0, 1].set_ylabel('时间 (秒)')
    
    # 游戏长度比较
    avg_lengths = [stats['avg_game_length'] for stats in stats_list]
    axes[1, 0].bar(agent_names, avg_lengths)
    axes[1, 0].set_title('平均游戏长度')
    axes[1, 0].set_ylabel('回合数')
    
    # 时间分布箱线图
    move_times_data = [stats['move_times'] for stats in stats_list]
    axes[1, 1].boxplot(move_times_data, labels=agent_names)
    axes[1, 1].set_title('思考时间分布')
    axes[1, 1].set_ylabel('时间 (秒)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()


def save_results(results, filename):
    """保存评估结果"""
    os.makedirs('results', exist_ok=True)
    filepath = os.path.join('results', filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"结果已保存到: {filepath}")


def load_results(filename):
    """加载评估结果"""
    filepath = os.path.join('results', filename)
    
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"结果已从 {filepath} 加载")
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI性能评估工具")
    
    # 基本参数
    parser.add_argument('--game', type=str, default='gomoku',
                       choices=['gomoku', 'snake'],
                       help='游戏类型')
    parser.add_argument('--agents', type=str, nargs='+',
                       default=['random', 'minimax'],
                       choices=['random', 'minimax', 'mcts', 'rl', 'behavior_tree', 
                               'improved_random', 'rule_based', 'greedy_snake', 'search_based'],
                       help='要评估的智能体类型')
    parser.add_argument('--games', type=int, default=100,
                       help='每个测试的游戏数量')
    parser.add_argument('--compare', action='store_true',
                       help='比较模式：智能体两两对战')
    parser.add_argument('--benchmark', action='store_true',
                       help='基准测试模式：与随机AI对战')
    
    # 游戏参数
    parser.add_argument('--board-size', type=int, default=15,
                       help='棋盘大小')
    parser.add_argument('--win-length', type=int, default=5,
                       help='获胜长度（五子棋）')
    
    # AI参数
    parser.add_argument('--minimax-depth', type=int, default=3,
                       help='Minimax搜索深度')
    parser.add_argument('--mcts-simulations', type=int, default=1000,
                       help='MCTS模拟次数')
    
    # 输出参数
    parser.add_argument('--save', type=str,
                       help='保存结果到文件')
    parser.add_argument('--load', type=str,
                       help='从文件加载结果')
    parser.add_argument('--plot', action='store_true',
                       help='绘制性能图表')
    parser.add_argument('--no-plot', action='store_true',
                       help='不显示图表')
    
    args = parser.parse_args()
    
    # 如果指定加载文件，直接加载并显示
    if args.load:
        results = load_results(args.load)
        if results and args.plot:
            # TODO: 从加载的结果重新绘制图表
            print("从文件绘制图表功能待实现")
        return
    
    # 创建游戏环境
    if args.game == 'gomoku':
        env = create_environment(args.game, 
                               board_size=args.board_size, 
                               win_length=args.win_length)
    else:
        env = create_environment(args.game, board_size=args.board_size)
    
    # 准备AI参数
    agent_kwargs = {
        'minimax': {'max_depth': args.minimax_depth},
        'mcts': {'simulation_count': args.mcts_simulations}
    }
    
    print(f"游戏类型: {args.game}")
    print(f"评估智能体: {args.agents}")
    print(f"每个测试游戏数: {args.games}")
    
    if args.compare:
        # 比较模式
        results = compare_agents(env, args.agents, args.games, **agent_kwargs)
        
        if args.save:
            save_results(results, args.save)
        
    elif args.benchmark:
        # 基准测试模式
        stats_list = []
        agent_names = []
        
        for agent_type in args.agents:
            kwargs = agent_kwargs.get(agent_type, {})
            agent = create_agent(agent_type, 1, **kwargs)
            stats = benchmark_single_agent(env, agent, args.games)
            
            stats_list.append(stats)
            agent_names.append(agent.name)
            
            print(f"\n{agent.name} 基准测试结果:")
            print(f"  胜率: {stats['win_rate']:.2%}")
            print(f"  平均思考时间: {stats['avg_move_time']:.3f}秒")
            print(f"  平均游戏长度: {stats['avg_game_length']:.1f}回合")
        
        # 分析性能
        analyze_performance(stats_list, agent_names)
        
        # 绘制图表
        if args.plot and not args.no_plot:
            save_path = f"results/benchmark_{args.game}_{int(time.time())}.png" if args.save else None
            plot_performance(stats_list, agent_names, save_path)
        
        # 保存结果
        if args.save:
            results = {
                'benchmark_results': {name: stats for name, stats in zip(agent_names, stats_list)},
                'config': vars(args),
                'timestamp': time.time()
            }
            save_results(results, args.save)
    
    else:
        print("请指定 --compare 或 --benchmark 模式")


if __name__ == "__main__":
    main() 