"""
第二阶段训练器：混合训练
"""
from typing import Dict, List, Any, Optional, Tuple
import os
import torch
import numpy as np
from tqdm import tqdm

from train.base.base_trainer import BaseTrainer
from werewolf_env import WerewolfEnv
from agents.base_agent import RandomAgent, HeuristicAgent
from models.rl_agent import RLAgent, WerewolfNetwork
from config.default_config import DEFAULT_GAME_CONFIG


class Stage2Trainer(BaseTrainer):
    """第二阶段训练器：混合训练"""
    
    def train(self,
              num_episodes: int = 2000,
              pretrained_model: Optional[WerewolfNetwork] = None,
              evaluate_every: int = 100,
              save_every: int = 500,
              render_every: int = 200) -> WerewolfNetwork:
        """
        训练模型
        
        Args:
            num_episodes: 训练局数
            pretrained_model: 预训练模型
            evaluate_every: 每多少局评估一次
            save_every: 每多少局保存一次模型
            render_every: 每多少局渲染一次
            
        Returns:
            训练好的模型
        """
        # 如果有预训练模型，加载其参数
        if pretrained_model is not None:
            self.model.load_state_dict(pretrained_model.state_dict())
            print("已加载预训练模型")
        
        # 创建预训练模型实例（用于其他玩家）
        pretrained_agents = []
        for i in range(self.num_players):
            pretrained_agents.append(RLAgent(i, model=pretrained_model, device=self.device))
        
        # 进度条
        pbar = tqdm(range(num_episodes))
        
        for episode in pbar:
            # 创建智能体，一个RL智能体和多个预训练模型
            agents = []
            rl_agent_indices = []
            
            # 随机选择一个位置放置RL智能体
            rl_index = np.random.randint(0, self.num_players)
            rl_agent_indices.append(rl_index)
            
            for i in range(self.num_players):
                if i == rl_index:
                    # RL智能体
                    agents.append(RLAgent(i, model=self.model, device=self.device))
                else:
                    # 预训练模型
                    agents.append(pretrained_agents[i])
            
            # 运行一局游戏
            render = (episode % render_every == 0)
            result, _ = self.run_episode(agents, render=render, training=True)
            
            # 更新统计数据
            self.stats['total_games'] += 1
            self.stats['game_lengths'].append(result['game_length'])
            self.stats['rewards'].append(result['total_rewards'][rl_index])
            
            if result['winner'] == 'werewolf':
                self.stats['werewolf_wins'] += 1
            elif result['winner'] == 'villager':
                self.stats['villager_wins'] += 1
            
            # 更新模型
            if hasattr(agents[rl_index], 'update_model'):
                loss = agents[rl_index].update_model()
                self.stats['losses'].append(loss)
                pbar.set_description(f"Loss: {loss:.4f}, Reward: {result['total_rewards'][rl_index]:.4f}")
            
            # 定期评估
            if episode % evaluate_every == evaluate_every - 1:
                eval_result = self.evaluate(num_episodes=10, render=False)
                print(f"\n评估结果 (第{episode+1}局):")
                print(f"狼人胜率: {eval_result['werewolf_win_rate']:.2f}")
                print(f"村民胜率: {eval_result['villager_win_rate']:.2f}")
                print(f"平均奖励: {eval_result['avg_reward']:.2f}")
            
            # 定期保存模型
            if episode % save_every == save_every - 1:
                self.save_model(f"{self.save_dir}/model_episode_{episode+1}.pt")
        
        # 保存最终模型
        self.save_model(f"{self.save_dir}/model_final.pt")
        
        return self.model
    
    def evaluate(self, num_episodes: int = 100, render: bool = False) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            num_episodes: 评估局数
            render: 是否渲染
            
        Returns:
            评估结果
        """
        results = []
        werewolf_wins = 0
        villager_wins = 0
        total_rewards = []
        
        for episode in range(num_episodes):
            # 创建智能体
            agents = []
            rl_agent_indices = []
            
            # 随机分配角色，确保RL智能体平等地扮演不同角色
            rl_index = episode % self.num_players
            rl_agent_indices.append(rl_index)
            
            for i in range(self.num_players):
                if i == rl_index:
                    # RL智能体
                    agents.append(RLAgent(i, model=self.model, device=self.device))
                else:
                    # 预训练模型
                    agents.append(RLAgent(i, model=self.model, device=self.device))
            
            # 运行一局游戏
            result, _ = self.run_episode(agents, render=render)
            results.append(result)
            
            # 统计胜率
            if result['winner'] == 'werewolf':
                werewolf_wins += 1
            elif result['winner'] == 'villager':
                villager_wins += 1
            
            # 记录RL智能体的奖励
            total_rewards.append(result['total_rewards'][rl_index])
        
        # 计算胜率和平均奖励
        werewolf_win_rate = werewolf_wins / num_episodes
        villager_win_rate = villager_wins / num_episodes
        avg_reward = sum(total_rewards) / len(total_rewards)
        
        return {
            'werewolf_win_rate': werewolf_win_rate,
            'villager_win_rate': villager_win_rate,
            'avg_reward': avg_reward,
            'detailed_results': results
        } 