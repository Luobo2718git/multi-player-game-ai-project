"""
游戏工具函数
"""

import time
from typing import Dict, Any, List

def evaluate_agents(env, agent1, agent2, num_games=10, save_results=False):
    """
    评估两个智能体的对战结果
    
    Args:
        env: 游戏环境
        agent1: 智能体1
        agent2: 智能体2
        num_games: 游戏局数
        save_results: 是否保存结果
    
    Returns:
        dict: 评估结果
    """
    results = {
        'games': [],
        'summary': {
            'total_games': num_games,
            'agent1_wins': 0,
            'agent2_wins': 0,
            'draws': 0,
            'agent1_win_rate': 0.0,
            'agent2_win_rate': 0.0,
            'draw_rate': 0.0
        }
    }
    
    for game_num in range(num_games):
        # 重置环境
        observation, info = env.reset()
        
        # 交替玩家顺序
        if game_num % 2 == 0:
            players = {1: agent1, 2: agent2}
        else:
            players = {1: agent2, 2: agent1}
        
        game_result = {
            'game_num': game_num + 1,
            'moves': [],
            'winner': None,
            'total_moves': 0,
            'game_time': 0
        }
        
        start_time = time.time()
        move_count = 0
        max_moves = 1000  # 防止无限循环
        
        # 游戏循环
        while not env.is_terminal() and move_count < max_moves:
            current_player = env.game.current_player
            current_agent = players[current_player]
            
            # 获取动作
            try:
                if current_agent.name == 'MinimaxBot':
                    action = current_agent.get_action(env.get_board_state(), env.game.current_player)
                else:
                    action = current_agent.get_action(observation, env)
                if action is None:
                    break
                
                # 执行动作
                observation, reward, terminated, truncated, step_info = env.step(action)
                
                # 记录移动
                game_result['moves'].append({
                    'player': current_player,
                    'agent': current_agent.name,
                    'action': action,
                    'reward': reward
                })
                
                move_count += 1
                
                if terminated or truncated:
                    break
                    
            except Exception as e:
                print(f"游戏 {game_num + 1} 中发生错误: {e}")
                break
        
        # 记录游戏结果
        game_result['total_moves'] = move_count
        game_result['game_time'] = time.time() - start_time
        game_result['winner'] = env.get_winner()
        
        # 更新统计
        winner = game_result['winner']
        if winner == 1:
            if game_num % 2 == 0:
                results['summary']['agent1_wins'] += 1
            else:
                results['summary']['agent2_wins'] += 1
        elif winner == 2:
            if game_num % 2 == 0:
                results['summary']['agent2_wins'] += 1
            else:
                results['summary']['agent1_wins'] += 1
        else:
            results['summary']['draws'] += 1
        
        results['games'].append(game_result)
        
        # 打印进度
        if (game_num + 1) % max(1, num_games // 10) == 0:
            print(f"已完成 {game_num + 1}/{num_games} 局游戏")
    
    # 计算胜率
    total = results['summary']['total_games']
    results['summary']['agent1_win_rate'] = results['summary']['agent1_wins'] / total
    results['summary']['agent2_win_rate'] = results['summary']['agent2_wins'] / total
    results['summary']['draw_rate'] = results['summary']['draws'] / total
    
    # 保存结果（如果需要）
    if save_results:
        import json
        import os
        
        os.makedirs('results', exist_ok=True)
        timestamp = int(time.time())
        filename = f'results/evaluation_{agent1.name}_vs_{agent2.name}_{timestamp}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {filename}")
    
    return results


def play_human_vs_ai(env, human_agent, ai_agent):
    """
    人机对战函数
    
    Args:
        env: 游戏环境
        human_agent: 人类智能体
        ai_agent: AI智能体
    """
    print("=== 人机对战开始 ===")
    print(f"人类玩家: {human_agent.name}")
    print(f"AI玩家: {ai_agent.name}")
    
    # 重置环境
    observation, info = env.reset()
    
    # 设置玩家
    players = {1: human_agent, 2: ai_agent}
    
    move_count = 0
    max_moves = 1000
    
    # 游戏循环
    while not env.is_terminal() and move_count < max_moves:
        current_player = env.game.current_player
        current_agent = players[current_player]
        
        print(f"\n=== 第 {move_count + 1} 回合 ===")
        print(f"当前玩家: {current_agent.name}")
        
        try:
            # 获取动作
            if current_agent.name == 'MinimaxBot':
                action = current_agent.get_action(env.get_board_state(), env.game.current_player)
            else:
                action = current_agent.get_action(observation, env)
            if action is None:
                print("无效动作，游戏结束")
                break
            
            # 执行动作
            observation, reward, terminated, truncated, step_info = env.step(action)
            
            print(f"执行动作: {action}")
            
            move_count += 1
            
            if terminated or truncated:
                break
                
        except Exception as e:
            print(f"游戏中发生错误: {e}")
            break
    
    # 显示最终结果
    print("\n=== 游戏结束 ===")
    env.render()
    
    winner = env.get_winner()
    if winner:
        winner_agent = players[winner]
        print(f"获胜者: {winner_agent.name}")
    else:
        print("平局")
    
    print(f"总回合数: {move_count}")


def tournament(env, agents, num_games_per_pair=10):
    """
    锦标赛模式，让多个智能体互相对战
    
    Args:
        env: 游戏环境
        agents: 智能体列表
        num_games_per_pair: 每对智能体的对战局数
    
    Returns:
        dict: 锦标赛结果
    """
    results = {
        'agents': [agent.name for agent in agents],
        'matches': [],
        'leaderboard': []
    }
    
    # 两两对战
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            agent1, agent2 = agents[i], agents[j]
            
            print(f"\n=== {agent1.name} vs {agent2.name} ===")
            
            match_result = evaluate_agents(
                env, agent1, agent2, 
                num_games=num_games_per_pair, 
                save_results=False
            )
            
            match_result['agent1_name'] = agent1.name
            match_result['agent2_name'] = agent2.name
            
            results['matches'].append(match_result)
            
            print(f"{agent1.name} 胜率: {match_result['summary']['agent1_win_rate']:.2%}")
            print(f"{agent2.name} 胜率: {match_result['summary']['agent2_win_rate']:.2%}")
            print(f"平局率: {match_result['summary']['draw_rate']:.2%}")
    
    # 计算排行榜
    agent_stats = {agent.name: {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0} for agent in agents}
    
    for match in results['matches']:
        agent1_name = match['agent1_name']
        agent2_name = match['agent2_name']
        
        agent1_wins = match['summary']['agent1_wins']
        agent2_wins = match['summary']['agent2_wins']
        draws = match['summary']['draws']
        
        agent_stats[agent1_name]['wins'] += agent1_wins
        agent_stats[agent1_name]['losses'] += agent2_wins
        agent_stats[agent1_name]['draws'] += draws
        agent_stats[agent1_name]['games'] += num_games_per_pair
        
        agent_stats[agent2_name]['wins'] += agent2_wins
        agent_stats[agent2_name]['losses'] += agent1_wins
        agent_stats[agent2_name]['draws'] += draws
        agent_stats[agent2_name]['games'] += num_games_per_pair
    
    # 计算胜率并排序
    for agent_name, stats in agent_stats.items():
        if stats['games'] > 0:
            stats['win_rate'] = stats['wins'] / stats['games']
        else:
            stats['win_rate'] = 0
    
    # 按胜率排序
    leaderboard = sorted(agent_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)
    results['leaderboard'] = leaderboard
    
    # 显示排行榜
    print("\n=== 锦标赛排行榜 ===")
    for rank, (agent_name, stats) in enumerate(leaderboard, 1):
        print(f"{rank}. {agent_name}: 胜率 {stats['win_rate']:.2%} "
              f"({stats['wins']}胜 {stats['losses']}负 {stats['draws']}平)")
    
    return results 