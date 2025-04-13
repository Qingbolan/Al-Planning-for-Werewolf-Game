#!/usr/bin/env python
"""
测试狼人杀游戏的胜利条件平衡性
"""
import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from werewolf_env import WerewolfEnv
from agents.base_agent import RandomAgent, HeuristicAgent
from config.default_config import DEFAULT_GAME_CONFIG

def run_game(agent_type, num_players=6):
    """运行一局游戏，返回获胜方"""
    env = WerewolfEnv(DEFAULT_GAME_CONFIG)
    obs, info = env.reset()
    
    # 创建智能体
    agents = []
    for i in range(num_players):
        if agent_type == 'random':
            agents.append(RandomAgent(i))
        elif agent_type == 'heuristic':
            agents.append(HeuristicAgent(i))
        else:
            agents.append(RandomAgent(i))
    
    # 初始化智能体
    for agent in agents:
        agent.initialize(env.game_state)
    
    # 游戏主循环
    done = False
    while not done:
        player_idx = env.current_player_id
        
        if player_idx < 0:
            break
            
        agent = agents[player_idx]
        obs = env.game_state.get_observation(player_idx)
        action = agent.act(obs)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated
    
    # 返回游戏结果
    return env.game_state.game_result

def test_win_condition(num_games=1000, agent_types=['random', 'heuristic']):
    """测试胜利条件的平衡性"""
    results = {}
    
    for agent_type in agent_types:
        print(f"\n测试 {agent_type} 智能体...")
        
        werewolf_wins = 0
        villager_wins = 0
        
        for i in range(num_games):
            if i % 100 == 0:
                print(f"已完成: {i}/{num_games} 局")
                
            result = run_game(agent_type)
            
            if result == 'werewolf':
                werewolf_wins += 1
            elif result == 'villager':
                villager_wins += 1
        
        werewolf_rate = werewolf_wins / num_games
        villager_rate = villager_wins / num_games
        
        results[agent_type] = {
            'werewolf_win_rate': werewolf_rate,
            'villager_win_rate': villager_rate,
            'werewolf_wins': werewolf_wins,
            'villager_wins': villager_wins
        }
        
        print(f"{agent_type} 智能体统计:")
        print(f"  狼人胜率: {werewolf_rate:.2f} ({werewolf_wins}/{num_games})")
        print(f"  村民胜率: {villager_rate:.2f} ({villager_wins}/{num_games})")
    
    # 可视化结果
    visualize_results(results)
    
    return results

def visualize_results(results):
    """可视化胜率分布"""
    try:
        plt.figure(figsize=(10, 6))
        
        agent_types = list(results.keys())
        werewolf_rates = [results[agent]['werewolf_win_rate'] for agent in agent_types]
        villager_rates = [results[agent]['villager_win_rate'] for agent in agent_types]
        
        x = np.arange(len(agent_types))
        width = 0.35
        
        plt.bar(x - width/2, werewolf_rates, width, label='狼人胜率')
        plt.bar(x + width/2, villager_rates, width, label='村民胜率')
        
        plt.ylabel('胜率')
        plt.title('不同智能体类型下的胜率分布')
        plt.xticks(x, agent_types)
        plt.legend()
        
        # 添加水平线表示理想平衡点
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('win_rate_distribution.png')
        print("胜率分布图已保存为 'win_rate_distribution.png'")
    except Exception as e:
        print(f"可视化失败: {e}")

if __name__ == "__main__":
    # 设置随机种子，确保结果可重现
    random.seed(42)
    np.random.seed(42)
    
    # 运行测试，使用较少的游戏数来加速测试
    test_win_condition(num_games=100) 