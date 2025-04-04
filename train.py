"""
狼人杀智能体训练脚本
"""
import os
import numpy as np
import random
import time
from typing import List, Dict, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import json

from werewolf_env import WerewolfEnv
from agents.base_agent import BaseAgent, RandomAgent, HeuristicAgent, create_agent
from config import DEFAULT_GAME_CONFIG, ROLE_TEAMS
from utils.visualizer import BeliefVisualizer


class Trainer:
    """智能体训练器"""
    
    def __init__(self, 
                 env_config: Dict[str, Any] = None, 
                 num_players: int = 6,
                 log_dir: str = './logs',
                 save_dir: str = './models',
                 visualize_dir: str = './visualizations'):
        """
        初始化训练器
        
        Args:
            env_config: 环境配置
            num_players: 玩家数量
            log_dir: 日志目录
            save_dir: 模型保存目录
            visualize_dir: 可视化保存目录
        """
        self.num_players = num_players
        self.env_config = env_config or DEFAULT_GAME_CONFIG
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.visualize_dir = visualize_dir
        
        # 创建目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(visualize_dir, exist_ok=True)
        
        # 创建环境
        self.env = WerewolfEnv(self.env_config)
        
        # 统计数据
        self.stats = {
            'werewolf_wins': 0,
            'villager_wins': 0,
            'total_games': 0,
            'game_lengths': [],
            'rewards': defaultdict(list)
        }
    
    def create_agents(self, agent_types: List[str]) -> List[BaseAgent]:
        """
        创建智能体
        
        Args:
            agent_types: 智能体类型列表
            
        Returns:
            智能体列表
        """
        agents = []
        for i in range(self.num_players):
            agent_type = agent_types[i % len(agent_types)]
            agents.append(create_agent(agent_type, i))
        return agents
    
    def run_episode(self, agents: List[BaseAgent], render: bool = False, visualize: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        运行一局游戏
        
        Args:
            agents: 智能体列表
            render: 是否渲染
            visualize: 是否可视化信念状态
            
        Returns:
            (游戏结果, 动作历史)
        """
        # 重置环境
        observations = self.env.reset()
        
        # 初始化智能体
        for i, agent in enumerate(agents):
            agent.initialize(self.env.game_state)
        
        done = False
        action_history = []
        total_rewards = [0] * self.num_players
        
        # 游戏主循环
        while not done:
            # 渲染
            if render:
                self.env.render()
            
            # 可视化信念状态
            if visualize and hasattr(agents[0], 'belief_updater') and agents[0].belief_updater:
                believer_id = 0
                belief_visualizer = BeliefVisualizer()
                timestamp = int(time.time())
                belief_visualizer.generate_belief_report(
                    agents[believer_id].belief_updater.belief_state,
                    self.env.game_state,
                    believer_id,
                    f"{self.visualize_dir}/belief_{timestamp}.png"
                )
            
            current_player = self.env.game_state.current_player
            
            # 如果当前阶段没有特定玩家行动，则跳过
            if current_player is None:
                # 游戏阶段转换
                observations, rewards, done, infos = self.env.step(None)
                continue
            
            # 获取当前玩家的智能体
            agent = agents[current_player]
            
            # 智能体决策
            action = agent.act(observations[current_player])
            
            # 记录行动
            action_info = {
                'player_id': current_player,
                'phase': self.env.game_state.phase,
                'action': action
            }
            action_history.append(action_info)
            
            # 执行行动
            observations, rewards, done, infos = self.env.step(action)
            
            # 累计奖励
            for i in range(self.num_players):
                total_rewards[i] += rewards[i]
        
        # 游戏结果
        result = {
            'winner': self.env.game_state.winner,
            'game_length': len(action_history),
            'total_rewards': total_rewards
        }
        
        if render:
            print(f"游戏结束! 胜利方: {result['winner']}")
            print(f"总奖励: {result['total_rewards']}")
        
        return result, action_history
    
    def evaluate(self, 
                agent_types: List[str], 
                num_episodes: int = 100, 
                render: bool = False,
                visualize: bool = False) -> Dict[str, Any]:
        """
        评估智能体
        
        Args:
            agent_types: 智能体类型列表
            num_episodes: 评估局数
            render: 是否渲染
            visualize: 是否可视化
            
        Returns:
            评估结果
        """
        results = []
        werewolf_wins = 0
        villager_wins = 0
        
        # 运行多局游戏
        for episode in range(num_episodes):
            agents = self.create_agents(agent_types)
            result, _ = self.run_episode(agents, render=(render and episode < 5), visualize=visualize)
            results.append(result)
            
            # 统计胜率
            if result['winner'] == 'werewolves':
                werewolf_wins += 1
            elif result['winner'] == 'villagers':
                villager_wins += 1
        
        # 计算胜率
        werewolf_win_rate = werewolf_wins / num_episodes
        villager_win_rate = villager_wins / num_episodes
        
        # 计算平均奖励
        avg_rewards = []
        for i in range(self.num_players):
            rewards = [r['total_rewards'][i] for r in results]
            avg_rewards.append(sum(rewards) / len(rewards))
        
        return {
            'num_episodes': num_episodes,
            'agent_types': agent_types,
            'werewolf_win_rate': werewolf_win_rate,
            'villager_win_rate': villager_win_rate,
            'avg_rewards': avg_rewards,
            'avg_game_length': sum(r['game_length'] for r in results) / len(results)
        }
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """
        保存评估结果
        
        Args:
            results: 评估结果
            filename: 文件名
        """
        with open(f"{self.log_dir}/{filename}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # 创建胜率可视化
        plt.figure(figsize=(10, 6))
        labels = ['狼人', '村民']
        values = [results['werewolf_win_rate'], results['villager_win_rate']]
        plt.bar(labels, values, color=['red', 'blue'])
        plt.ylim(0, 1)
        plt.title('阵营胜率')
        plt.ylabel('胜率')
        plt.savefig(f"{self.log_dir}/{filename}_win_rate.png")
        plt.close()
        
        # 创建平均奖励可视化
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(results['avg_rewards'])), results['avg_rewards'])
        plt.title('玩家平均奖励')
        plt.xlabel('玩家ID')
        plt.ylabel('平均奖励')
        plt.savefig(f"{self.log_dir}/{filename}_rewards.png")
        plt.close()
    
    def train_self_play(self, 
                       initial_agent_types: List[str], 
                       num_generations: int = 10,
                       episodes_per_generation: int = 100,
                       render: bool = False,
                       visualize: bool = False):
        """
        自我对弈训练
        
        Args:
            initial_agent_types: 初始智能体类型列表
            num_generations: 世代数量
            episodes_per_generation: 每世代局数
            render: 是否渲染
            visualize: 是否可视化
        """
        current_agent_types = initial_agent_types
        
        for generation in range(num_generations):
            print(f"训练世代 {generation + 1}/{num_generations}")
            
            # 评估当前智能体
            eval_results = self.evaluate(
                current_agent_types, 
                num_episodes=episodes_per_generation,
                render=render,
                visualize=visualize
            )
            
            # 保存结果
            self.save_results(eval_results, f"generation_{generation}")
            
            # 更新智能体策略
            # 在实际实现中，这里应该根据评估结果改进智能体
            # 例如：更新策略网络参数、调整探索率等
            
            # 目前仅为演示，我们不做实际更新
            current_agent_types = initial_agent_types
        
        print("训练完成!")


def main():
    """主函数"""
    # 创建训练器
    trainer = Trainer(num_players=6)
    
    # 随机智能体评估
    print("评估随机智能体...")
    random_results = trainer.evaluate(
        agent_types=['random'],
        num_episodes=100,
        render=True
    )
    trainer.save_results(random_results, "random_agents")
    
    # 启发式智能体评估
    print("评估启发式智能体...")
    heuristic_results = trainer.evaluate(
        agent_types=['heuristic'],
        num_episodes=100,
        render=True
    )
    trainer.save_results(heuristic_results, "heuristic_agents")
    
    # 混合智能体评估
    print("评估混合智能体...")
    mixed_results = trainer.evaluate(
        agent_types=['random', 'heuristic'],
        num_episodes=100,
        render=True
    )
    trainer.save_results(mixed_results, "mixed_agents")
    
    # 简单的自我对弈训练演示
    print("开始自我对弈训练...")
    trainer.train_self_play(
        initial_agent_types=['heuristic'],
        num_generations=5,
        episodes_per_generation=50,
        render=False
    )


if __name__ == "__main__":
    main() 