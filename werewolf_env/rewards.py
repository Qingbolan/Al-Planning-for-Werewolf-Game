from typing import Dict, Any, List, Tuple
import numpy as np
from collections import defaultdict

# 导入角色信息
from config.default_config import ROLE_TEAMS


class RewardFunction:
    """狼人杀游戏的奖励函数系统
    
    该系统负责计算玩家在不同游戏阶段的奖励值。包括:
    1. 基础奖励 - 游戏胜负和行动成功/失败
    2. 阶段性奖励 - 夜晚行动、白天讨论和投票阶段的特定奖励
    3. 角色特定奖励 - 针对不同角色的特殊奖励机制
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化奖励函数系统
        
        Args:
            config: 奖励函数配置，包含各种奖励的权重
        """
        self.config = config or {}
        
        # 设置基础奖励权重
        self._setup_reward_weights()
        
        # 累计奖励跟踪
        self.cumulative_rewards = defaultdict(float)
    
    def _setup_reward_weights(self):
        """设置各种奖励的权重"""
        # 基础奖励权重
        self.weights = {
            # 游戏结果奖励
            'win': self.config.get('win_reward', 1.0),
            'lose': self.config.get('lose_reward', -0.5),
            'draw': self.config.get('draw_reward', 0.0),
            
            # 行动奖励
            'action_success': self.config.get('action_success_reward', 0.1),
            'action_failure': self.config.get('action_failure_reward', -0.05),
            
            # 阶段性奖励
            'correct_accusation': self.config.get('correct_accusation_reward', 0.2),  # 正确指认狼人
            'incorrect_accusation': self.config.get('incorrect_accusation_reward', -0.1),  # 错误指认平民
            'werewolf_misdirection': self.config.get('werewolf_misdirection_reward', 0.3),  # 狼人成功误导
            'survival_per_round': self.config.get('survival_per_round', 0.05),  # 每轮存活奖励
            
            # 角色特定奖励
            'werewolf_team_kill': self.config.get('werewolf_team_kill', 0.2),  # 狼人成功击杀
            'seer_correct_check': self.config.get('seer_correct_check', 0.2),  # 预言家正确查验
            'robber_strategic_steal': self.config.get('robber_strategic_steal', 0.15),  # 强盗有战略性的偷窃
        }
    
    def reset(self):
        """重置累计奖励"""
        self.cumulative_rewards = defaultdict(float)
    
    def compute_reward(self, player_id: int, action: Any, result: Dict[str, Any], 
                       game_state: Any) -> float:
        """计算当前动作的奖励值
        
        Args:
            player_id: 执行动作的玩家ID
            action: 执行的动作
            result: 动作执行结果
            game_state: 当前游戏状态
            
        Returns:
            计算得到的奖励值
        """
        reward = 0.0
        
        # 1. 基础奖励：动作成功或失败
        reward += self._compute_action_reward(result)
        
        # 2. 游戏结束奖励
        if game_state.phase == 'end':
            end_reward = self._compute_end_game_reward(player_id, game_state)
            reward += end_reward
        
        # 3. 阶段性奖励
        phase_reward = self._compute_phase_reward(player_id, action, result, game_state)
        reward += phase_reward
        
        # 4. 角色特定奖励
        role_reward = self._compute_role_specific_reward(player_id, action, result, game_state)
        reward += role_reward
        
        # 更新累计奖励
        self.cumulative_rewards[player_id] += reward
        
        return reward
    
    def _compute_action_reward(self, result: Dict[str, Any]) -> float:
        """计算动作执行的基础奖励
        
        Args:
            result: 动作执行结果
            
        Returns:
            动作奖励值
        """
        if result.get('success', False):
            return self.weights['action_success']
        else:
            return self.weights['action_failure']
    
    def _compute_end_game_reward(self, player_id: int, game_state: Any) -> float:
        """计算游戏结束时的奖励
        
        Args:
            player_id: 玩家ID
            game_state: 游戏状态
            
        Returns:
            游戏结束奖励值
        """
        # 获取玩家角色和阵营
        player_role = game_state.players[player_id]['current_role']
        player_team = ROLE_TEAMS.get(player_role, 'villager')
        
        # 游戏结果
        game_result = game_state.game_result
        
        # 根据阵营分配奖励
        if game_result == player_team:
            # 胜利
            return self.weights['win']
        elif game_result == 'draw':
            # 平局
            return self.weights['draw']
        else:
            # 失败
            return self.weights['lose']
    
    def _compute_phase_reward(self, player_id: int, action: Any, 
                             result: Dict[str, Any], game_state: Any) -> float:
        """计算特定游戏阶段的奖励
        
        Args:
            player_id: 玩家ID
            action: 执行的动作
            result: 动作执行结果
            game_state: 游戏状态
            
        Returns:
            阶段性奖励值
        """
        reward = 0.0
        phase = game_state.phase
        
        # 玩家角色和阵营
        player_role = game_state.players[player_id]['current_role']
        player_team = ROLE_TEAMS.get(player_role, 'villager')
        
        # 1. 夜晚阶段奖励
        if phase == 'night':
            # 待实现更复杂的夜晚奖励逻辑
            pass
            
        # 2. 白天讨论阶段奖励
        elif phase == 'day' and game_state.sub_phase == 'discussion':
            # 预言家正确指认狼人
            if player_role == 'seer' and action.action_type == 'speak':
                # 分析发言内容，检查是否正确指认了狼人
                # 这需要NLP分析或结构化的发言数据
                pass
                
            # 狼人成功误导
            elif player_team == 'werewolf' and action.action_type == 'speak':
                # 分析发言内容，检查是否成功误导
                pass
        
        # 3. 投票阶段奖励
        elif phase == 'day' and game_state.sub_phase == 'vote':
            if action.action_type == 'vote':
                target_id = action.target
                
                # 获取目标玩家的角色和阵营
                target_role = game_state.players[target_id]['current_role']
                target_team = ROLE_TEAMS.get(target_role, 'villager')
                
                # 村民正确投票狼人
                if player_team == 'villager' and target_team == 'werewolf':
                    reward += self.weights['correct_accusation']
                
                # 村民错误投票村民
                elif player_team == 'villager' and target_team == 'villager':
                    reward += self.weights['incorrect_accusation']
                
                # 狼人成功误导村民投票村民
                elif player_team == 'werewolf' and target_team == 'villager':
                    # 检查是否有村民跟随投票
                    for voter_id, vote_target in game_state.votes.items():
                        voter_role = game_state.players[voter_id]['current_role']
                        voter_team = ROLE_TEAMS.get(voter_role, 'villager')
                        
                        if voter_team == 'villager' and vote_target == target_id:
                            reward += self.weights['werewolf_misdirection']
                            break
        
        # 4. 每轮存活奖励
        if game_state.round > 1:  # 从第二轮开始奖励存活
            reward += self.weights['survival_per_round']
            
        return reward
    
    def _compute_role_specific_reward(self, player_id: int, action: Any, 
                                     result: Dict[str, Any], game_state: Any) -> float:
        """计算角色特定的奖励
        
        Args:
            player_id: 玩家ID
            action: 执行的动作
            result: 动作执行结果
            game_state: 游戏状态
            
        Returns:
            角色特定奖励值
        """
        reward = 0.0
        
        # 获取玩家角色
        player_role = game_state.players[player_id]['current_role']
        
        # 1. 狼人特定奖励
        if player_role == 'werewolf':
            if action.action_type == 'kill' and result.get('success', False):
                # 成功击杀奖励
                reward += self.weights['werewolf_team_kill']
        
        # 2. 预言家特定奖励
        elif player_role == 'seer':
            if action.action_type == 'check' and result.get('success', False):
                target_id = action.target
                target_role = game_state.players[target_id]['current_role']
                target_team = ROLE_TEAMS.get(target_role, 'villager')
                
                # 预言家发现狼人
                if target_team == 'werewolf':
                    reward += self.weights['seer_correct_check']
        
        # 3. 强盗特定奖励
        elif player_role == 'robber':
            if action.action_type == 'rob' and result.get('success', False):
                target_id = action.target
                target_role = game_state.players[target_id]['current_role']
                
                # 强盗偷取了特殊角色
                if target_role != 'villager':
                    reward += self.weights['robber_strategic_steal']
        
        return reward
    
    def get_cumulative_rewards(self) -> Dict[int, float]:
        """获取所有玩家的累积奖励
        
        Returns:
            玩家ID到累积奖励的映射
        """
        return dict(self.cumulative_rewards) 