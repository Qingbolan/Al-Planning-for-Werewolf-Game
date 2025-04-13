"""
强化学习智能体
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
import os

from agents.base_agent import BaseAgent
from werewolf_env.state import GameState
from werewolf_env.actions import (
    ActionType, Action, NightAction, DaySpeech, VoteAction, NoAction,
    create_night_action, create_speech, create_vote, create_no_action,
    SpeechType
)

class WerewolfNetwork(nn.Module):
    """
    狼人杀游戏的神经网络模型
    
    特点:
    1. 处理多种输入特征（玩家状态、游戏状态、历史信息）
    2. 支持不同角色的策略
    3. 输出动作概率和状态价值
    """
    
    def __init__(self, 
                 observation_dim: int, 
                 action_dim: int,
                 num_players: int = 6,
                 num_roles: int = 6,
                 hidden_dim: int = 256):
        """
        初始化网络
        
        Args:
            observation_dim: 观察空间维度
            action_dim: 动作空间维度
            num_players: 玩家数量
            num_roles: 角色数量
            hidden_dim: 隐藏层维度
        """
        super(WerewolfNetwork, self).__init__()
        
        # 特征维度
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 角色嵌入
        self.role_embedding = nn.Embedding(num_roles, hidden_dim // 4)
        
        # 玩家身份嵌入
        self.player_embedding = nn.Embedding(num_players, hidden_dim // 4)
        
        # LSTM用于处理历史信息
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True
        )
        
        # 计算组合特征的维度
        combined_dim = hidden_dim + (hidden_dim // 4) * 2  # 特征提取器输出 + 角色嵌入 + 玩家嵌入
        lstm_dim = hidden_dim // 2  # LSTM隐藏层输出
        total_dim = combined_dim + lstm_dim  # 最终输入到策略头和价值头的维度
        
        # 策略头（动作概率）
        self.policy_head = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 价值头（状态价值）
        self.value_head = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, 
               state_dict: Dict[str, torch.Tensor], 
               hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            state_dict: 状态字典，包含各种输入特征
            hidden_state: LSTM隐藏状态
            
        Returns:
            action_logits: 动作对数概率
            state_value: 状态价值
            new_hidden_state: 新的LSTM隐藏状态
        """
        # 提取状态特征
        state_features = state_dict.get('features')
        
        # 确保特征维度正确
        if state_features.shape[-1] != self.observation_dim:
            print(f"警告: 输入特征维度 {state_features.shape[-1]} 与预期维度 {self.observation_dim} 不匹配")
            # 如果维度不匹配，重塑特征向量
            if state_features.shape[-1] < self.observation_dim:
                # 填充
                padding = torch.zeros(state_features.shape[0], self.observation_dim - state_features.shape[-1], 
                                    device=state_features.device)
                state_features = torch.cat([state_features, padding], dim=-1)
            else:
                # 截断
                state_features = state_features[:, :self.observation_dim]
        
        # 获取角色和玩家ID
        role_id = state_dict.get('role_id', torch.zeros(state_features.shape[0], dtype=torch.long, device=state_features.device))
        player_id = state_dict.get('player_id', torch.zeros(state_features.shape[0], dtype=torch.long, device=state_features.device))
        
        # 获取历史信息
        history = state_dict.get('history', None)
        
        # 特征提取
        x = self.feature_extractor(state_features)
        
        # 嵌入角色和玩家ID
        role_emb = self.role_embedding(role_id)
        player_emb = self.player_embedding(player_id)
        
        # 结合特征和嵌入
        combined_features = torch.cat([x, role_emb, player_emb], dim=-1)
        
        # 打印维度（调试用）
        print(f"组合特征维度: {combined_features.shape}")
        
        # 处理历史信息
        if history is not None:
            batch_size = history.shape[0]
            seq_len = history.shape[1]
            
            # 重塑历史信息以适应LSTM输入
            history_reshaped = history.view(batch_size, seq_len, -1)
            
            # 应用LSTM处理历史信息
            lstm_out, new_hidden_state = self.lstm(history_reshaped, hidden_state)
            
            # 获取最后一个时间步的输出
            lstm_features = lstm_out[:, -1, :]
        else:
            # 如果没有历史信息，初始化LSTM特征为0
            lstm_features = torch.zeros(combined_features.shape[0], self.lstm.hidden_size, device=combined_features.device)
            
            if hidden_state is None:
                # 初始化隐藏状态
                h0 = torch.zeros(1, combined_features.shape[0], self.lstm.hidden_size, device=combined_features.device)
                c0 = torch.zeros(1, combined_features.shape[0], self.lstm.hidden_size, device=combined_features.device)
                new_hidden_state = (h0, c0)
            else:
                new_hidden_state = hidden_state
        
        # 打印维度（调试用）
        print(f"LSTM特征维度: {lstm_features.shape}")
        
        # 合并所有特征
        all_features = torch.cat([combined_features, lstm_features], dim=-1)
        
        # 打印维度（调试用）
        print(f"所有特征维度: {all_features.shape}")
        
        # 计算动作概率和状态价值
        action_logits = self.policy_head(all_features)
        state_value = self.value_head(all_features)
        
        return action_logits, state_value, new_hidden_state

class MaskedCategorical:
    """支持动作掩码的分类分布"""
    
    def __init__(self, logits, mask):
        """
        初始化分布
        
        Args:
            logits: 未归一化的对数概率
            mask: 合法动作掩码 (1表示合法，0表示非法)
        """
        self.logits = logits
        self.mask = mask
        
        # 应用掩码，将非法动作的概率设为极小值
        masked_logits = logits.clone()
        masked_logits[mask == 0] = -1e9
        
        self.probs = F.softmax(masked_logits, dim=-1)
    
    def sample(self):
        """采样动作"""
        return torch.multinomial(self.probs, 1).squeeze(-1)
    
    def log_prob(self, actions):
        """计算对数概率"""
        # 确保actions的维度正确
        if len(actions.shape) == 0:
            actions = actions.unsqueeze(0)  # 转换为[batch_size]形状
        
        # 确保actions的批次维度与概率分布匹配
        if len(actions.shape) == 1 and len(self.probs.shape) == 2:
            # 转换为[batch_size, 1]形状
            actions = actions.unsqueeze(-1)
        
        # 根据维度情况选择正确的gather方法
        if len(self.probs.shape) == len(actions.shape) + 1:
            # 如果probs比actions多一个维度
            # 例如probs: [batch_size, num_actions], actions: [batch_size]
            # 需要扩展actions的维度
            gathered = self.probs.gather(1, actions.unsqueeze(-1))
            return torch.log(gathered).squeeze(-1)
        else:
            # 如果维度相同
            # 例如probs: [batch_size, num_actions], actions: [batch_size, 1]
            gathered = self.probs.gather(-1, actions)
            return torch.log(gathered)
    
    def entropy(self):
        """计算熵"""
        return -torch.sum(self.probs * torch.log(self.probs + 1e-9), dim=-1)


class RLAgent(BaseAgent):
    """基于强化学习的智能体"""
    
    def __init__(self, player_id: int, model: WerewolfNetwork = None, device: str = "cpu"):
        """
        初始化RL智能体
        
        Args:
            player_id: 玩家ID
            model: 预训练模型
            device: 计算设备
        """
        super().__init__(player_id)
        
        self.device = torch.device(device)
        self.model = model
        self.hidden_state = None
        
        # 检查并确保模型在正确的设备上
        if model is not None and next(model.parameters()).device != self.device:
            self.model = model.to(self.device)
            print(f"已将模型转移到设备: {self.device}")
        
        # 动作历史，用于训练
        self.action_history = []
        self.state_history = []
        self.reward_history = []
        self.entropy_history = []
        self.action_log_prob_history = []
        self.value_history = []
        
        # 探索参数
        self.epsilon = 0.1  # 探索率
        
        # 角色到ID的映射
        self.role_to_id = {
            'villager': 0,
            'werewolf': 1,
            'seer': 2,
            'robber': 3,
            'troublemaker': 4,
            'insomniac': 5,
            'minion': 6,
        }
        
        # 当前可用动作掩码
        self.action_mask = None
        
        # GPU内存管理
        self.use_cuda = self.device.type == 'cuda'
        if self.use_cuda:
            # 启用CUDA内存优化
            torch.cuda.empty_cache()
            # 设置较低的初始内存分配以减少碎片
            torch.cuda.set_per_process_memory_fraction(0.7)
    
    def initialize(self, game_state):
        """初始化智能体状态"""
        super().initialize(game_state)
        
        # 重置LSTM隐藏状态
        self.hidden_state = None
        
        # 重置历史记录
        self.action_history = []
        self.state_history = []
        self.reward_history = []
        self.entropy_history = []
        self.action_log_prob_history = []
        self.value_history = []
        
        # 清理GPU内存
        if self.use_cuda:
            torch.cuda.empty_cache()
    
    def _preprocess_observation(self, observation):
        """
        预处理观察结果为NN输入
        
        Args:
            observation: 观察结果
            
        Returns:
            处理后的状态字典
        """
        # 获取当前状态信息
        phase = observation.get('phase', self.current_phase)
        role = observation.get('original_role', self.current_role)
        role_id = self.role_to_id.get(role, 0)
        
        # 检查游戏状态是否存在
        if not self.game_state or not hasattr(self.game_state, 'players'):
            # 返回默认特征
            features = np.zeros(128)  # 使用默认128维特征
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            role_id_tensor = torch.LongTensor([role_id]).to(self.device)
            player_id_tensor = torch.LongTensor([self.player_id]).to(self.device)
            
            return {
                'features': features_tensor,
                'role_id': role_id_tensor,
                'player_id': player_id_tensor,
                'history': None
            }
        
        # 制作特征向量
        features = []
        
        # 加入玩家ID的one-hot编码
        player_one_hot = np.zeros(len(self.game_state.players))
        player_one_hot[self.player_id] = 1
        features.extend(player_one_hot)
        
        # 加入玩家的存活状态 - 默认所有玩家都存活
        alive_status = np.ones(len(self.game_state.players))  # 默认所有玩家都是存活的
        # 如果游戏状态中有存活信息，使用它
        if self.game_state and hasattr(self.game_state, 'players'):
            for i, player in enumerate(self.game_state.players):
                if 'alive' in player and not player['alive']:
                    alive_status[i] = 0
        features.extend(alive_status)
        
        # 加入阶段的one-hot编码
        game_phases = ['night', 'day', 'vote', 'end']  # 定义游戏阶段
        phase_one_hot = np.zeros(len(game_phases))
        phase_index = game_phases.index(phase) if phase in game_phases else 0
        phase_one_hot[phase_index] = 1
        features.extend(phase_one_hot)
        
        # 如果有信念状态，加入信念状态的信息
        if self.belief_updater:
            for player_id in range(len(self.game_state.players)):
                if player_id == self.player_id:
                    # 自己的角色是确定的
                    belief = np.zeros(len(self.role_to_id))
                    belief[role_id] = 1
                else:
                    # 其他玩家的角色信念
                    belief = np.zeros(len(self.role_to_id))
                    
                    if player_id in self.belief_updater.belief_state.beliefs:
                        for role, prob in self.belief_updater.belief_state.beliefs[player_id].items():
                            if role in self.role_to_id:
                                belief[self.role_to_id[role]] = prob
                
                features.extend(belief)
        
        # 如果有历史信息，加入历史信息
        history_features = []
        if hasattr(self, 'action_history') and len(self.action_history) > 0:
            # 简单地加入最近的几个动作的ID
            recent_actions = self.action_history[-5:]  # 最近5个动作
            for action in recent_actions:
                if isinstance(action, dict):
                    action_type = action.get('type', '')
                    player = action.get('player_id', -1)
                    
                    # 简单编码动作类型和玩家
                    action_feature = np.zeros(len(ActionType) + len(self.game_state.players))
                    
                    # 设置动作类型
                    if action_type in [t.name for t in ActionType]:
                        action_feature[ActionType[action_type].value] = 1
                    
                    # 设置玩家ID
                    if 0 <= player < len(self.game_state.players):
                        action_feature[len(ActionType) + player] = 1
                    
                    history_features.extend(action_feature)
        
        # 补全特征向量到128维
        current_dim = len(features)
        if current_dim < 128:
            features.extend([0.0] * (128 - current_dim))
        elif current_dim > 128:
            features = features[:128]  # 截断到128维
        
        # 将特征转换为张量并移至正确设备
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        role_id_tensor = torch.LongTensor([role_id]).to(self.device)
        player_id_tensor = torch.LongTensor([self.player_id]).to(self.device)
        
        return {
            'features': features_tensor,
            'role_id': role_id_tensor,
            'player_id': player_id_tensor,
            'history': None
        }
        
    def _update_action_mask(self, observation):
        """
        更新动作掩码
        
        Args:
            observation: 观察结果
        """
        # 检查动作空间
        action_space = observation.get('action_space', [])
        if not action_space:
            # 如果动作空间为空，默认所有动作有效
            self.action_mask = torch.ones(100, dtype=torch.float32, device=self.device)
            return
        
        # 否则，根据有效动作创建掩码
        mask = torch.zeros(100, dtype=torch.float32, device=self.device)
        
        for action in action_space:
            action_id = self._action_to_id(action)
            if 0 <= action_id < 100:  # 确保ID在有效范围内
                mask[action_id] = 1.0
        
        self.action_mask = mask
    
    def get_action(self, game_state):
        """
        获取动作
        
        Args:
            game_state: 游戏状态
            
        Returns:
            选择的动作
        """
        # 更新游戏状态
        self.game_state = game_state
        
        # 获取观察结果
        observation = game_state.get_observation(self.player_id)
        
        # 更新动作掩码
        self._update_action_mask(observation)
        
        # 获取状态特征
        state_dict = self._preprocess_observation(observation)
        self.state_history.append(state_dict)
        
        with torch.no_grad():
            # 将一切移到GPU（如果可用）
            if self.action_mask is not None:
                self.action_mask = self.action_mask.to(self.device)
            
            # 获取策略和价值
            policy_logits, value, self.hidden_state = self.model(state_dict, self.hidden_state)
            
            # 使用掩码创建分类分布
            masked_policy = MaskedCategorical(policy_logits, self.action_mask)
            
            # 以概率epsilon进行随机探索
            if self.training and np.random.random() < self.epsilon:
                # 随机选择有效动作
                valid_actions = torch.nonzero(self.action_mask).squeeze(-1)
                if len(valid_actions) > 0:
                    action_id = int(valid_actions[np.random.randint(0, len(valid_actions))].cpu().numpy())
                else:
                    action_id = np.random.randint(0, 100)  # 如果没有有效动作，随机选择
            else:
                # 根据策略采样动作
                action_id = int(masked_policy.sample().cpu().numpy())
                
            # 记录动作和策略信息（如果在训练中）
            if self.training:
                # 将action_id转换为张量以计算对数概率
                action_tensor = torch.tensor([action_id], device=self.device)
                log_prob = masked_policy.log_prob(action_tensor)
                entropy = masked_policy.entropy()
                
                self.action_log_prob_history.append(log_prob)
                self.entropy_history.append(entropy)
                self.value_history.append(value)
        
        # 将动作ID转换为实际动作
        action = self._action_id_to_action(action_id, observation)
        
        # 记录动作历史
        self.action_history.append(action)
        
        return action
        
    def batch_get_actions(self, observations, max_batch_size=16):
        """
        批量获取动作
        
        Args:
            observations: 观察结果列表
            max_batch_size: 最大批量大小
            
        Returns:
            动作ID列表
        """
        num_observations = len(observations)
        action_ids = []
        
        # 分批处理
        for i in range(0, num_observations, max_batch_size):
            batch_observations = observations[i:i + max_batch_size]
            batch_size = len(batch_observations)
            
            # 准备批量输入
            features_batch = []
            role_ids_batch = []
            player_ids_batch = []
            action_masks_batch = []
            
            for obs in batch_observations:
                # 预处理观察
                state_dict = self._preprocess_observation(obs)
                features_batch.append(state_dict['features'])
                role_ids_batch.append(state_dict['role_id'])
                player_ids_batch.append(state_dict['player_id'])
                
                # 更新动作掩码
                self._update_action_mask(obs)
                action_masks_batch.append(self.action_mask)
            
            # 连接批次
            features_tensor = torch.cat(features_batch, dim=0)
            role_ids_tensor = torch.cat(role_ids_batch, dim=0)
            player_ids_tensor = torch.cat(player_ids_batch, dim=0)
            action_masks_tensor = torch.stack(action_masks_batch)
            
            batch_state_dict = {
                'features': features_tensor,
                'role_id': role_ids_tensor,
                'player_id': player_ids_tensor,
                'history': None
            }
            
            with torch.no_grad():
                # 批量推理
                policy_logits, _, _ = self.model(batch_state_dict)
                
                # 对每个样本使用其掩码
                batch_action_ids = []
                for j in range(batch_size):
                    masked_policy = MaskedCategorical(policy_logits[j:j+1], action_masks_tensor[j:j+1])
                    action_id = int(masked_policy.sample().cpu().numpy())
                    batch_action_ids.append(action_id)
                
                action_ids.extend(batch_action_ids)
        
        return action_ids
        
    def get_training_data_batch(self, max_batch_size=32):
        """
        以批量方式获取训练数据
        
        Args:
            max_batch_size: 最大批量大小
            
        Returns:
            训练数据批次
        """
        # 收集所有经验
        if not self.state_history or not self.action_log_prob_history:
            return None
        
        # 准备训练数据
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        
        # 添加到批次数据
        states = self.state_history
        log_probs = self.action_log_prob_history
        values = self.value_history
        
        # 处理奖励
        if not self.reward_history:
            # 如果没有奖励记录，使用默认值
            rewards = [torch.tensor([0.0], device=self.device) for _ in range(len(states))]
        else:
            rewards = [torch.tensor([r], device=self.device) for r in self.reward_history]
        
        return states, log_probs, values, rewards
    
    def update_model(self, optimizer, criterion=None, batch_size=4):
        """
        更新模型参数
        
        Args:
            optimizer: 优化器
            criterion: 损失函数
            batch_size: 批量大小
            
        Returns:
            损失值
        """
        if not self.state_history or not self.action_log_prob_history:
            return 0.0
        
        # 获取训练数据
        states, log_probs, values, rewards = self.get_training_data_batch()
        
        # 使用批量处理进行更新
        num_samples = len(states)
        total_loss = 0.0
        
        for i in range(0, num_samples, batch_size):
            batch_states = states[i:i + batch_size]
            batch_log_probs = log_probs[i:i + batch_size]
            batch_values = values[i:i + batch_size]
            batch_rewards = rewards[i:i + batch_size]
            
            # 计算优势
            advantages = []
            for j in range(len(batch_rewards)):
                advantage = batch_rewards[j] - batch_values[j].detach()
                advantages.append(advantage)
            
            # 将优势转换为张量
            advantages_tensor = torch.cat(advantages)
            
            # 计算策略损失
            policy_loss = -torch.cat(batch_log_probs) * advantages_tensor
            
            # 计算价值损失
            value_loss = F.mse_loss(
                torch.cat(batch_values), 
                torch.cat(batch_rewards)
            )
            
            # 计算熵奖励（鼓励探索）
            entropy = torch.cat([e.mean() for e in self.entropy_history[i:i + batch_size]])
            
            # 总损失
            loss = policy_loss.mean() + 0.5 * value_loss - 0.01 * entropy.mean()
            
            # 更新模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 清除历史
        self.state_history = []
        self.action_log_prob_history = []
        self.entropy_history = []
        self.value_history = []
        self.reward_history = []
        
        # 平均损失
        return total_loss / (num_samples // batch_size + 1)
    
    def store_reward(self, reward: float) -> None:
        """
        存储奖励，用于训练
        
        Args:
            reward: 奖励值
        """
        self.reward_history.append(reward)
    
    def get_training_data(self) -> Tuple[List, List, List, List, List]:
        """
        获取训练数据
        
        Returns:
            (状态历史, 动作历史, 奖励历史, 动作对数概率历史, 价值历史)
        """
        return (
            self.state_history,
            self.action_history,
            self.reward_history,
            self.action_log_prob_history,
            self.value_history
        )
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if self.model:
            torch.save(self.model.state_dict(), path)
    
    def load(self, path: str) -> None:
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        if self.model:
            self.model.load_state_dict(torch.load(path, map_location=self.device)) 