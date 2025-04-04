"""
狼人杀游戏强化学习模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import random

from agents.base_agent import BaseAgent
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
        
        # 策略头（动作概率）
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 价值头（状态价值）
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
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
        
        # 合并所有特征
        all_features = torch.cat([combined_features, lstm_features], dim=-1)
        
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
        return torch.log(self.probs.gather(1, actions.unsqueeze(-1))).squeeze(-1)
    
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
    
    def _preprocess_observation(self, observation):
        """
        预处理观察结果为模型可用的张量
        
        Args:
            observation: 观察结果
            
        Returns:
            预处理后的状态字典
        """
        if not observation:
            return None
            
        # 提取观察中的关键信息
        phase = observation.get('phase', self.current_phase)
        role = observation.get('original_role', self.current_role)
        
        # 将角色转换为ID
        role_id = self.role_to_id.get(role, 0)
        
        # 制作特征向量
        features = []
        
        # 加入玩家ID的one-hot编码
        player_one_hot = np.zeros(len(self.game_state.players))
        player_one_hot[self.player_id] = 1
        features.extend(player_one_hot)
        
        # 加入玩家的存活状态
        alive_status = np.array([1 if player['alive'] else 0 for player in self.game_state.players])
        features.extend(alive_status)
        
        # 加入阶段的one-hot编码
        phase_one_hot = np.zeros(len(GameState.GAME_PHASES))
        phase_index = GameState.GAME_PHASES.index(phase) if phase in GameState.GAME_PHASES else 0
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
        
        # 转换为PyTorch张量
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        role_id_tensor = torch.LongTensor([role_id]).to(self.device)
        player_id_tensor = torch.LongTensor([self.player_id]).to(self.device)
        
        history_tensor = None
        if history_features:
            history_tensor = torch.FloatTensor(history_features).unsqueeze(0).unsqueeze(0).to(self.device)
        
        return {
            'features': features_tensor,
            'role_id': role_id_tensor,
            'player_id': player_id_tensor,
            'history': history_tensor
        }
    
    def _update_action_mask(self, observation):
        """
        更新当前可用动作掩码
        
        Args:
            observation: 观察结果
        """
        phase = observation.get('phase', self.current_phase)
        role = observation.get('original_role', self.current_role)
        
        # 创建一个全零掩码（默认所有动作不可用）
        mask = np.zeros(self.model.policy_head[-1].out_features)
        
        if phase == 'night':
            # 夜晚行动可用性取决于角色
            if role == 'werewolf':
                # 狼人可以查看其他狼人或中央牌堆
                mask[0:2] = 1  # 假设前两个动作是狼人的夜晚行动
            elif role == 'seer':
                # 预言家可以查验玩家或中央牌堆
                mask[2:4] = 1  # 假设接下来两个动作是预言家的夜晚行动
                # 对于查验玩家动作，还需要设置可查验的玩家
                for i in range(len(self.game_state.players)):
                    if i != self.player_id:
                        mask[4 + i] = 1  # 可以查验的其他玩家
            # 其他角色的夜晚行动...
            
        elif phase == 'day':
            # 白天发言掩码
            speech_start_idx = 20  # 假设前20个动作是夜晚行动
            
            # 所有玩家都可以声称角色
            mask[speech_start_idx:speech_start_idx+7] = 1  # 可以声称的7种角色
            
            # 指控其他玩家
            for i in range(len(self.game_state.players)):
                if i != self.player_id:
                    mask[speech_start_idx+7+i] = 1  # 可以指控的其他玩家
            
            # 其他发言类型...
            
        elif phase == 'vote':
            # 投票掩码
            vote_start_idx = 50  # 假设前50个动作是夜晚行动和白天发言
            
            # 可以投票给任何其他玩家
            for i in range(len(self.game_state.players)):
                if i != self.player_id:
                    mask[vote_start_idx+i] = 1
        
        # 如果没有任何可用动作，允许空动作
        if np.sum(mask) == 0:
            mask[-1] = 1  # 最后一个动作是空动作
        
        self.action_mask = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
    
    def _action_id_to_action(self, action_id, observation):
        """
        将动作ID转换为实际动作
        
        Args:
            action_id: 动作ID
            observation: 观察结果
            
        Returns:
            实际动作
        """
        phase = observation.get('phase', self.current_phase)
        role = observation.get('original_role', self.current_role)
        
        if phase == 'night':
            # 解析夜晚行动
            if role == 'werewolf':
                if action_id == 0:
                    return create_night_action(self.player_id, role, 'check_other_werewolves')
                elif action_id == 1:
                    card_index = random.randint(0, 2)
                    return create_night_action(self.player_id, role, 'check_center_card', card_index=card_index)
            
            elif role == 'seer':
                if action_id == 2:
                    # 查验玩家
                    target_id = (action_id - 2) % len(self.game_state.players)
                    if target_id == self.player_id:  # 避免查验自己
                        target_id = (target_id + 1) % len(self.game_state.players)
                    return create_night_action(self.player_id, role, 'check_player', target_id=target_id)
                elif action_id == 3:
                    # 查看中央牌堆
                    card_indices = random.sample(range(3), 2)
                    return create_night_action(self.player_id, role, 'check_center_cards', card_indices=card_indices)
            
            # 其他角色的夜晚行动...
            
        elif phase == 'day':
            # 解析白天发言
            speech_start_idx = 20  # 假设前20个动作是夜晚行动
            
            if speech_start_idx <= action_id < speech_start_idx + 7:
                # 声称角色
                claimed_role = list(self.role_to_id.keys())[action_id - speech_start_idx]
                return create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role=claimed_role)
            
            elif speech_start_idx + 7 <= action_id < speech_start_idx + 7 + len(self.game_state.players):
                # 指控其他玩家
                target_id = action_id - (speech_start_idx + 7)
                return create_speech(self.player_id, SpeechType.ACCUSE.name, 
                                    target_id=target_id, accused_role='werewolf')
            
            # 其他发言类型...
            
        elif phase == 'vote':
            # 解析投票动作
            vote_start_idx = 50  # 假设前50个动作是夜晚行动和白天发言
            
            if vote_start_idx <= action_id < vote_start_idx + len(self.game_state.players):
                # 投票给玩家
                target_id = action_id - vote_start_idx
                if target_id == self.player_id:  # 避免投票给自己
                    target_id = (target_id + 1) % len(self.game_state.players)
                return create_vote(self.player_id, target_id)
        
        # 默认返回空动作
        return create_no_action(self.player_id)
    
    def _night_action(self, observation: Dict[str, Any]) -> Action:
        """基于模型选择夜晚行动"""
        # 预处理观察结果
        state_dict = self._preprocess_observation(observation)
        if state_dict is None:
            return create_no_action(self.player_id)
        
        # 更新动作掩码
        self._update_action_mask(observation)
        
        # 使用模型预测动作
        with torch.no_grad():
            action_logits, state_value, new_hidden_state = self.model(state_dict, self.hidden_state)
            
            # 使用带掩码的分类分布采样动作
            dist = MaskedCategorical(action_logits, self.action_mask)
            
            # 以一定概率随机探索
            if random.random() < self.epsilon:
                # 随机选择一个合法动作
                valid_actions = torch.nonzero(self.action_mask.squeeze()).squeeze()
                if valid_actions.dim() == 0:  # 只有一个合法动作
                    action = valid_actions.unsqueeze(0)
                else:
                    action = valid_actions[random.randint(0, len(valid_actions) - 1)].unsqueeze(0)
            else:
                # 使用模型策略采样动作
                action = dist.sample().unsqueeze(0)
            
            # 计算对数概率和熵，用于训练
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        
        # 更新LSTM隐藏状态
        self.hidden_state = new_hidden_state
        
        # 记录状态和动作，用于训练
        self.state_history.append(state_dict)
        action_id = action.item()
        self.action_history.append(action_id)
        self.action_log_prob_history.append(log_prob.item())
        self.entropy_history.append(entropy.item())
        self.value_history.append(state_value.item())
        
        # 将动作ID转换为实际动作
        return self._action_id_to_action(action_id, observation)
    
    def _day_action(self, observation: Dict[str, Any]) -> Action:
        """基于模型选择白天发言"""
        # 使用与夜晚行动相同的逻辑，但处理不同的动作空间
        return self._night_action(observation)  # 复用相同的逻辑
    
    def _vote_action(self, observation: Dict[str, Any]) -> Action:
        """基于模型选择投票目标"""
        # 使用与夜晚行动相同的逻辑，但处理不同的动作空间
        return self._night_action(observation)  # 复用相同的逻辑
    
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