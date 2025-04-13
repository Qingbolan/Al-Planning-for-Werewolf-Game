"""
基础训练器类
"""
from typing import Dict, List, Any, Optional, Tuple
import os
import torch
import numpy as np
from abc import ABC, abstractmethod
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from werewolf_env import WerewolfEnv
from models.rl_agent import WerewolfNetwork
from config.default_config import DEFAULT_GAME_CONFIG


class BaseTrainer(ABC):
    """基础训练器类"""
    
    def __init__(self,
                 env_config: Dict[str, Any],
                 num_players: int = 6,
                 log_dir: str = './logs',
                 save_dir: str = './models',
                 visualize_dir: str = './visualizations',
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 num_workers: int = 4):
        """
        初始化训练器
        
        Args:
            env_config: 环境配置
            num_players: 玩家数量
            log_dir: 日志目录
            save_dir: 模型保存目录
            visualize_dir: 可视化保存目录
            device: 计算设备
            num_workers: 并行工作进程数量
        """
        self.env_config = env_config or DEFAULT_GAME_CONFIG
        self.num_players = num_players
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.visualize_dir = visualize_dir
        self.device = device
        self.num_workers = num_workers if num_workers > 0 else mp.cpu_count()
        
        # 使用多少并行进程
        print(f"初始化训练器，使用设备: {self.device}, 并行进程数: {self.num_workers}")
        
        # 创建目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(visualize_dir, exist_ok=True)
        
        # 创建环境
        self.env = WerewolfEnv(self.env_config)
        
        # 创建模型
        self.model = WerewolfNetwork(
            observation_dim=128,
            action_dim=100,
            num_players=num_players,
            hidden_dim=256
        ).to(device)
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        
        # 统计数据
        self.stats = {
            'werewolf_wins': 0,
            'villager_wins': 0,
            'total_games': 0,
            'game_lengths': [],
            'rewards': [],
            'losses': []
        }
    
    @abstractmethod
    def train(self, **kwargs) -> WerewolfNetwork:
        """
        训练模型
        
        Returns:
            训练好的模型
        """
        pass
    
    def run_episode(self, agents: List[Any], render: bool = False, training: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        运行一局游戏
        
        Args:
            agents: 智能体列表
            render: 是否渲染
            training: 是否在训练模式
            
        Returns:
            (游戏结果, 动作历史)
        """
        # 重置环境
        obs = self.env.reset()
        
        # 初始化智能体
        for agent in agents:
            agent.initialize(self.env.game_state)
        
        # 游戏循环
        done = False
        action_history = []
        
        while not done:
            # 获取当前玩家
            current_player = self.env.game_state.get_current_player()
            if current_player < 0:
                break
            
            # 获取当前玩家的动作
            action = agents[current_player].get_action(self.env.game_state)
            
            # 执行动作
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # 记录动作历史
            action_history.append({
                'player': current_player,
                'action': action,
                'reward': reward,
                'info': info
            })
            
            # 更新智能体
            if training and hasattr(agents[current_player], 'update_beliefs'):
                agents[current_player].update_beliefs(self.env.game_state)
            
            # 检查游戏是否结束
            done = terminated or truncated
        
        # 游戏结果
        result = {
            'winner': self.env.game_state.game_result,
            'game_length': len(action_history),
            'total_rewards': {i: self.env.rewards[i] for i in range(self.num_players)}
        }
        
        return result, action_history
    
    def run_parallel_episodes(self, create_agents_fn, num_episodes: int, training: bool = False) -> List[Dict[str, Any]]:
        """
        并行运行多局游戏
        
        Args:
            create_agents_fn: 创建智能体的函数
            num_episodes: 游戏局数
            training: 是否在训练模式
            
        Returns:
            游戏结果列表
        """
        results = []
        
        # 定义单个工作函数
        def worker_fn(episode_idx: int):
            # 为每个线程创建独立的环境和智能体
            env = WerewolfEnv(self.env_config)
            agents = create_agents_fn()
            
            # 运行游戏
            obs = env.reset()
            for agent in agents:
                agent.initialize(env.game_state)
            
            done = False
            action_history = []
            
            while not done:
                current_player = env.game_state.get_current_player()
                if current_player < 0:
                    break
                
                action = agents[current_player].get_action(env.game_state)
                obs, reward, terminated, truncated, info = env.step(action)
                
                action_history.append({
                    'player': current_player,
                    'action': action,
                    'reward': reward,
                    'info': info
                })
                
                if training and hasattr(agents[current_player], 'update_beliefs'):
                    agents[current_player].update_beliefs(env.game_state)
                
                done = terminated or truncated
            
            # 构建游戏结果
            result = {
                'winner': env.game_state.game_result,
                'game_length': len(action_history),
                'total_rewards': {i: env.rewards[i] for i in range(self.num_players)},
                'action_history': action_history
            }
            
            if training:
                # 获取训练数据
                training_data = {}
                for i, agent in enumerate(agents):
                    if hasattr(agent, 'get_training_data'):
                        training_data[i] = agent.get_training_data()
                result['training_data'] = training_data
            
            return result
        
        # 使用线程池并行执行
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(worker_fn, i) for i in range(num_episodes)]
            for future in tqdm(futures, desc="并行游戏执行"):
                results.append(future.result())
        
        return results
    
    @abstractmethod
    def evaluate(self, num_episodes: int = 100, render: bool = False) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            num_episodes: 评估局数
            render: 是否渲染
            
        Returns:
            评估结果
        """
        pass
    
    def parallel_evaluate(self, create_agents_fn, num_episodes: int) -> Dict[str, Any]:
        """
        并行评估模型
        
        Args:
            create_agents_fn: 创建智能体的函数
            num_episodes: 评估局数
            
        Returns:
            评估结果
        """
        results = self.run_parallel_episodes(create_agents_fn, num_episodes, training=False)
        
        # 统计结果
        werewolf_wins = sum(1 for r in results if r['winner'] == 'werewolf')
        villager_wins = sum(1 for r in results if r['winner'] == 'villager')
        game_lengths = [r['game_length'] for r in results]
        avg_rewards = []
        
        for result in results:
            rewards = list(result['total_rewards'].values())
            if rewards:
                avg_rewards.append(sum(rewards) / len(rewards))
        
        return {
            'werewolf_win_rate': werewolf_wins / num_episodes if num_episodes > 0 else 0,
            'villager_win_rate': villager_wins / num_episodes if num_episodes > 0 else 0,
            'avg_game_length': sum(game_lengths) / len(game_lengths) if game_lengths else 0,
            'avg_reward': sum(avg_rewards) / len(avg_rewards) if avg_rewards else 0,
            'detailed_results': results
        }
    
    def save_model(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存到 {path}")
    
    def load_model(self, path: str) -> None:
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"已从 {path} 加载模型")
    
    def save_stats(self, path: str) -> None:
        """
        保存统计数据
        
        Args:
            path: 保存路径
        """
        import json
        with open(path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"统计数据已保存到 {path}")
    
    def load_stats(self, path: str) -> None:
        """
        加载统计数据
        
        Args:
            path: 统计数据路径
        """
        import json
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.stats = json.load(f)
            print(f"已从 {path} 加载统计数据") 