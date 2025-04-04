"""
基于强化学习的狼人杀智能体训练器
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import wandb
from tqdm import tqdm

from werewolf_env import WerewolfEnv
from agents.base_agent import BaseAgent, RandomAgent, HeuristicAgent
from models.rl_agent import RLAgent, WerewolfNetwork
from utils.belief_updater import BeliefState
from config.default_config import DEFAULT_GAME_CONFIG, ROLE_TEAMS


class RLTrainer:
    """强化学习训练器"""
    
    def __init__(self, 
                 env_config: Dict[str, Any] = None, 
                 num_players: int = 6,
                 obs_dim: int = 128,
                 action_dim: int = 100,
                 learning_rate: float = 0.0003,
                 gamma: float = 0.99,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 log_dir: str = './logs',
                 save_dir: str = './models/saved',
                 visualize_dir: str = './visualizations',
                 use_wandb: bool = False):
        """
        初始化训练器
        
        Args:
            env_config: 环境配置
            num_players: 玩家数量
            obs_dim: 观察空间维度
            action_dim: 动作空间维度
            learning_rate: 学习率
            gamma: 折扣因子
            entropy_coef: 熵系数
            value_coef: 价值函数系数
            max_grad_norm: 梯度裁剪范数
            device: 计算设备
            log_dir: 日志目录
            save_dir: 模型保存目录
            visualize_dir: 可视化保存目录
            use_wandb: 是否使用wandb记录实验
        """
        self.num_players = num_players
        self.env_config = env_config or DEFAULT_GAME_CONFIG
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.visualize_dir = visualize_dir
        self.device = device
        
        # RL参数
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # 创建目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(visualize_dir, exist_ok=True)
        
        # 创建环境
        self.env = WerewolfEnv(self.env_config)
        
        # 创建模型
        self.model = WerewolfNetwork(
            observation_dim=obs_dim,
            action_dim=action_dim,
            num_players=num_players,
            hidden_dim=256
        ).to(device)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 使用wandb
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project="werewolf-rl",
                config={
                    "num_players": num_players,
                    "learning_rate": learning_rate,
                    "gamma": gamma,
                    "entropy_coef": entropy_coef,
                    "value_coef": value_coef,
                    "env_config": env_config,
                }
            )
        
        # 统计数据
        self.stats = {
            'werewolf_wins': 0,
            'villager_wins': 0,
            'total_games': 0,
            'game_lengths': [],
            'rewards': defaultdict(list),
            'losses': []
        }
    
    def create_agents(self, agent_types: List[str], model=None) -> List[BaseAgent]:
        """
        创建智能体
        
        Args:
            agent_types: 智能体类型列表，可以是'random', 'heuristic', 'rl'
            model: RL智能体使用的模型
            
        Returns:
            智能体列表
        """
        agents = []
        for i in range(self.num_players):
            agent_type = agent_types[i % len(agent_types)]
            
            if agent_type == 'random':
                agents.append(RandomAgent(i))
            elif agent_type == 'heuristic':
                agents.append(HeuristicAgent(i))
            elif agent_type == 'rl':
                agents.append(RLAgent(i, model=model, device=self.device))
            else:
                # 默认随机智能体
                agents.append(RandomAgent(i))
        
        return agents
    
    def compute_returns(self, rewards: List[float], values: List[float], gamma: float) -> List[float]:
        """
        计算回报值
        
        Args:
            rewards: 奖励序列
            values: 价值序列
            gamma: 折扣因子
            
        Returns:
            回报值序列
        """
        returns = []
        R = 0
        
        for r, v in zip(reversed(rewards), reversed(values)):
            R = r + gamma * R
            returns.insert(0, R)
            
        return returns
    
    def update_model(self, 
                    state_history: List[Dict[str, torch.Tensor]], 
                    action_history: List[int], 
                    reward_history: List[float], 
                    action_log_prob_history: List[float], 
                    value_history: List[float]) -> float:
        """
        更新模型参数
        
        Args:
            state_history: 状态历史
            action_history: 动作历史
            reward_history: 奖励历史
            action_log_prob_history: 动作对数概率历史
            value_history: 价值历史
            
        Returns:
            损失值
        """
        # 计算回报值
        returns = self.compute_returns(reward_history, value_history, self.gamma)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 转换为张量
        action_log_probs = torch.FloatTensor(action_log_prob_history).to(self.device)
        values = torch.FloatTensor(value_history).unsqueeze(1).to(self.device)
        
        # 计算优势
        advantages = returns.unsqueeze(1) - values
        
        # 策略损失
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        
        # 价值损失
        value_loss = 0.5 * advantages.pow(2).mean()
        
        # 熵损失（促进探索）
        entropy_loss = torch.FloatTensor([-self.entropy_coef * ent for ent in action_log_prob_history]).mean()
        
        # 总损失
        loss = policy_loss + self.value_coef * value_loss + entropy_loss
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        return loss.item()
    
    def run_episode(self, agents: List[BaseAgent], render: bool = False, training: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        运行一局游戏
        
        Args:
            agents: 智能体列表
            render: 是否渲染
            training: 是否处于训练模式
            
        Returns:
            (游戏结果, 动作历史)
        """
        # 重置环境
        observations, _ = self.env.reset()
        
        # 初始化智能体
        for i, agent in enumerate(agents):
            agent.initialize(self.env.game_state)
        
        done = False
        truncated = False
        action_history = []
        total_rewards = [0] * self.num_players
        
        # 游戏主循环
        while not done and not truncated:
            # 渲染
            if render:
                self.env.render()
            
            current_player = self.env.current_player_id
            
            # 如果当前阶段没有特定玩家行动，则跳过
            if current_player is None or current_player < 0:
                # 游戏阶段转换
                observations, rewards, done, truncated, infos = self.env.step(None)
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
            observations, rewards, done, truncated, infos = self.env.step(action)
            
            # 累计奖励并存储到智能体中（用于训练）
            for i, reward in enumerate(rewards):
                total_rewards[i] += reward
                if training and isinstance(agents[i], RLAgent):
                    agents[i].store_reward(reward)
        
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
    
    def collect_training_data(self, agents: List[RLAgent]) -> Tuple[List, List, List, List, List]:
        """
        从智能体收集训练数据
        
        Args:
            agents: RL智能体列表
            
        Returns:
            合并的训练数据
        """
        all_states = []
        all_actions = []
        all_rewards = []
        all_log_probs = []
        all_values = []
        
        # 从每个RL智能体获取训练数据
        for agent in agents:
            if isinstance(agent, RLAgent):
                states, actions, rewards, log_probs, values = agent.get_training_data()
                
                if states:  # 只有当智能体有动作时才添加
                    all_states.extend(states)
                    all_actions.extend(actions)
                    all_rewards.extend(rewards)
                    all_log_probs.extend(log_probs)
                    all_values.extend(values)
        
        return all_states, all_actions, all_rewards, all_log_probs, all_values
    
    def train(self, 
             num_episodes: int = 1000, 
             batch_size: int = 4,
             opponent_types: List[str] = ['random', 'heuristic'],
             evaluate_every: int = 100,
             save_every: int = 500,
             render_every: int = 200):
        """
        训练RL智能体
        
        Args:
            num_episodes: 总训练局数
            batch_size: 批次大小，每批次更新一次模型
            opponent_types: 对手类型
            evaluate_every: 每多少局评估一次
            save_every: 每多少局保存一次模型
            render_every: 每多少局渲染一次
        """
        # 统计初始化
        episode_rewards = []
        episode_lengths = []
        episode_losses = []
        werewolf_wins = 0
        villager_wins = 0
        
        # 进度条
        pbar = tqdm(range(num_episodes))
        
        for episode in pbar:
            # 创建智能体，一个RL智能体和多个其他类型智能体
            agents = []
            rl_agent_indices = []
            
            # 随机选择一个位置放置RL智能体
            rl_index = random.randint(0, self.num_players - 1)
            rl_agent_indices.append(rl_index)
            
            for i in range(self.num_players):
                if i == rl_index:
                    # RL智能体
                    agents.append(RLAgent(i, model=self.model, device=self.device))
                else:
                    # 随机选择其他类型的智能体
                    agent_type = random.choice(opponent_types)
                    if agent_type == 'random':
                        agents.append(RandomAgent(i))
                    elif agent_type == 'heuristic':
                        agents.append(HeuristicAgent(i))
                    else:
                        agents.append(RandomAgent(i))
            
            # 运行一局游戏
            render = (episode % render_every == 0)
            result, _ = self.run_episode(agents, render=render, training=True)
            
            # 收集统计数据
            episode_rewards.append(sum(result['total_rewards']) / len(result['total_rewards']))
            episode_lengths.append(result['game_length'])
            
            if result['winner'] == 'werewolves':
                werewolf_wins += 1
            elif result['winner'] == 'villagers':
                villager_wins += 1
            
            # 收集训练数据
            states, actions, rewards, log_probs, values = self.collect_training_data(agents)
            
            # 如果收集到足够的数据，或者到达批次大小，就更新模型
            if states and (episode % batch_size == batch_size - 1 or episode == num_episodes - 1):
                loss = self.update_model(states, actions, rewards, log_probs, values)
                episode_losses.append(loss)
                
                # 更新进度条
                pbar.set_description(f"Loss: {loss:.4f}, Reward: {episode_rewards[-1]:.4f}")
                
                # 记录wandb
                if self.use_wandb:
                    wandb.log({
                        "loss": loss,
                        "reward": episode_rewards[-1],
                        "episode_length": episode_lengths[-1],
                        "werewolf_win_rate": werewolf_wins / (episode + 1),
                        "villager_win_rate": villager_wins / (episode + 1)
                    })
            
            # 定期评估
            if episode % evaluate_every == evaluate_every - 1:
                eval_result = self.evaluate(num_episodes=10, render=False)
                
                print(f"\nEvaluation after {episode+1} episodes:")
                print(f"Werewolf Win Rate: {eval_result['werewolf_win_rate']:.2f}")
                print(f"Villager Win Rate: {eval_result['villager_win_rate']:.2f}")
                print(f"Average Reward: {eval_result['avg_reward']:.2f}")
                
                # 记录wandb
                if self.use_wandb:
                    wandb.log({
                        "eval_werewolf_win_rate": eval_result['werewolf_win_rate'],
                        "eval_villager_win_rate": eval_result['villager_win_rate'],
                        "eval_avg_reward": eval_result['avg_reward']
                    })
            
            # 定期保存模型
            if episode % save_every == save_every - 1:
                self.save_model(f"{self.save_dir}/model_episode_{episode+1}.pt")
        
        # 训练结束后保存最终模型
        self.save_model(f"{self.save_dir}/model_final.pt")
        
        # 保存统计数据
        self.stats['werewolf_wins'] = werewolf_wins
        self.stats['villager_wins'] = villager_wins
        self.stats['total_games'] = num_episodes
        self.stats['game_lengths'] = episode_lengths
        self.stats['rewards']['all'] = episode_rewards
        self.stats['losses'] = episode_losses
        
        self.save_results(self.stats, f"{self.log_dir}/training_results.json")
        
        # 可视化训练过程
        self.visualize_training(episode_rewards, episode_lengths, episode_losses)
        
        return self.stats
    
    def evaluate(self, num_episodes: int = 100, render: bool = False) -> Dict[str, Any]:
        """
        评估RL智能体
        
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
            # 创建智能体，RL智能体与随机和启发式智能体对抗
            agents = []
            
            # 随机分配角色，确保RL智能体平等地扮演不同角色
            rl_index = episode % self.num_players
            
            for i in range(self.num_players):
                if i == rl_index:
                    # RL智能体
                    agents.append(RLAgent(i, model=self.model, device=self.device))
                else:
                    # 其他智能体，随机选择
                    if random.random() < 0.5:
                        agents.append(RandomAgent(i))
                    else:
                        agents.append(HeuristicAgent(i))
            
            # 运行一局游戏
            result, _ = self.run_episode(agents, render=render)
            results.append(result)
            
            # 统计胜率
            if result['winner'] == 'werewolves':
                werewolf_wins += 1
            elif result['winner'] == 'villagers':
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
        print(f"从 {path} 加载模型")
    
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """
        保存结果
        
        Args:
            results: 结果字典
            filename: 文件名
        """
        # 转换numpy数组为列表，以便JSON序列化
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results[key] = value.tolist()
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        results[key][k] = v.tolist()
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
    
    def visualize_training(self, rewards: List[float], lengths: List[int], losses: List[float]) -> None:
        """
        可视化训练过程
        
        Args:
            rewards: 奖励序列
            lengths: 游戏长度序列
            losses: 损失序列
        """
        plt.figure(figsize=(15, 10))
        
        # 绘制奖励曲线
        plt.subplot(3, 1, 1)
        plt.plot(rewards)
        plt.title('Average Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # 绘制游戏长度曲线
        plt.subplot(3, 1, 2)
        plt.plot(lengths)
        plt.title('Game Length per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        
        # 绘制损失曲线
        plt.subplot(3, 1, 3)
        plt.plot(losses)
        plt.title('Loss per Batch Update')
        plt.xlabel('Batch Update')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(f"{self.visualize_dir}/training_curves.png")
        plt.close()


def main():
    """主函数"""
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建训练器
    trainer = RLTrainer(
        num_players=6,
        obs_dim=128,
        action_dim=100,
        learning_rate=0.0003,
        gamma=0.99,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_wandb=False  # 设置为True以使用wandb
    )
    
    # 训练
    trainer.train(
        num_episodes=5000,
        batch_size=4,
        opponent_types=['random', 'heuristic'],
        evaluate_every=100,
        save_every=500,
        render_every=200
    )
    
    # 最终评估
    eval_result = trainer.evaluate(num_episodes=100, render=False)
    print("\nFinal Evaluation:")
    print(f"Werewolf Win Rate: {eval_result['werewolf_win_rate']:.2f}")
    print(f"Villager Win Rate: {eval_result['villager_win_rate']:.2f}")
    print(f"Average Reward: {eval_result['avg_reward']:.2f}")


if __name__ == "__main__":
    main() 