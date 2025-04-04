"""
狼人杀游戏角色观察空间定义
为不同角色实现特定的观察空间
"""
from typing import Dict, List, Any, Tuple, Optional, Union
import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod

from werewolf_env.state import GameState


class RoleObservation(ABC):
    """角色观察基类"""
    
    def __init__(self, player_id: int, game_state: GameState):
        self.player_id = player_id
        self.game_state = game_state
        self.observation = game_state.get_observation(player_id)
        
        # 角色特定的观察信息
        self.role_specific_obs = self._get_role_specific_observation()
        
    @abstractmethod
    def _get_role_specific_observation(self) -> Dict[str, Any]:
        """获取角色特定的观察信息"""
        pass
    
    def get_combined_observation(self) -> Dict[str, Any]:
        """获取组合观察信息（基础观察 + 角色特定观察）"""
        combined_obs = self.observation.copy()
        combined_obs.update(self.role_specific_obs)
        return combined_obs
    
    def to_vector(self) -> np.ndarray:
        """将观察转换为向量表示，用于神经网络输入"""
        # 基本向量化逻辑，子类可以重写此方法提供更特定的向量化
        
        # 阶段编码
        phase_map = {phase: i for i, phase in enumerate(GameState.GAME_PHASES)}
        phase_encoding = np.zeros(len(GameState.GAME_PHASES))
        phase_idx = phase_map.get(self.observation.get('phase', 'init'), 0)
        phase_encoding[phase_idx] = 1
        
        # 角色编码
        all_roles = self.game_state.config.get('roles', [])
        unique_roles = list(set(all_roles))
        role_map = {role: i for i, role in enumerate(unique_roles)}
        
        original_role = self.observation.get('original_role', '')
        role_encoding = np.zeros(len(unique_roles))
        if original_role in role_map:
            role_encoding[role_map[original_role]] = 1
        
        # 发言历史编码
        speech_history = self.observation.get('speech_history', [])
        speech_encoding = np.zeros((10, 5))  # 假设最多记录10条发言，每条5个特征
        
        for i, speech in enumerate(speech_history[-10:]):  # 只取最近10条
            if i >= 10:
                break
                
            speaker_id = speech.get('player_id', 0)
            content = speech.get('content', {})
            speech_type = content.get('type', '')
            
            # 简单编码：发言者ID，发言类型，目标ID（如果有）
            speech_encoding[i, 0] = speaker_id / self.game_state.num_players
            
            # 发言类型编码
            speech_type_map = {
                'CLAIM_ROLE': 1,
                'CLAIM_ACTION_RESULT': 2,
                'ACCUSE': 3,
                'DEFEND': 4,
                'VOTE_INTENTION': 5
            }
            speech_encoding[i, 1] = speech_type_map.get(speech_type, 0) / 5
            
            # 目标ID编码
            target_id = -1
            if 'target_id' in content:
                target_id = content['target_id']
            elif 'target' in content:
                target_id = content['target']
                
            if 0 <= target_id < self.game_state.num_players:
                speech_encoding[i, 2] = target_id / self.game_state.num_players
            
            # 角色声明编码
            claimed_role = ''
            if 'role' in content:
                claimed_role = content['role']
            elif 'accused_role' in content:
                claimed_role = content['accused_role']
                
            if claimed_role in role_map:
                speech_encoding[i, 3] = role_map[claimed_role] / len(unique_roles)
        
        # 投票编码
        votes = self.observation.get('votes', {})
        vote_encoding = np.zeros(self.game_state.num_players)
        
        for voter_id, target_id in votes.items():
            if 0 <= target_id < self.game_state.num_players:
                vote_encoding[target_id] += 1
                
        vote_encoding = vote_encoding / max(1, np.sum(vote_encoding))  # 归一化
        
        # 组合所有特征
        features = [
            phase_encoding,
            role_encoding,
            speech_encoding.flatten(),
            vote_encoding
        ]
        
        return np.concatenate([f for f in features])


class VillagerObservation(RoleObservation):
    """村民观察类"""
    
    def _get_role_specific_observation(self) -> Dict[str, Any]:
        """获取村民特定的观察信息"""
        # 村民没有特殊的观察能力
        return {
            'role_type': 'villager',
            'special_info': {}
        }


class WerewolfObservation(RoleObservation):
    """狼人观察类"""
    
    def _get_role_specific_observation(self) -> Dict[str, Any]:
        """获取狼人特定的观察信息"""
        other_werewolves = []
        
        # 获取其他狼人信息
        for player_id, player_info in enumerate(self.game_state.players):
            if (player_id != self.player_id and 
                (player_info['original_role'] == 'werewolf' or player_info['current_role'] == 'werewolf')):
                other_werewolves.append(player_id)
        
        return {
            'role_type': 'werewolf',
            'special_info': {
                'other_werewolves': other_werewolves
            }
        }
    
    def to_vector(self) -> np.ndarray:
        """将狼人观察转换为向量表示"""
        base_vector = super().to_vector()
        
        # 添加其他狼人信息
        other_werewolves = self.role_specific_obs['special_info'].get('other_werewolves', [])
        werewolf_encoding = np.zeros(self.game_state.num_players)
        
        for werewolf_id in other_werewolves:
            if 0 <= werewolf_id < self.game_state.num_players:
                werewolf_encoding[werewolf_id] = 1
        
        return np.concatenate([base_vector, werewolf_encoding])


class SeerObservation(RoleObservation):
    """预言家观察类"""
    
    def _get_role_specific_observation(self) -> Dict[str, Any]:
        """获取预言家特定的观察信息"""
        checked_players = {}
        checked_center_cards = {}
        
        # 从行动历史中提取预言家的查验结果
        for action in self.game_state.action_history:
            if action['player_id'] == self.player_id and action['role'] == 'seer':
                if action['action'] == 'check_player' and 'target' in action and 'result' in action:
                    checked_players[action['target']] = action['result']
                    
                elif action['action'] == 'check_center_cards' and 'targets' in action and 'result' in action:
                    for i, card_idx in enumerate(action['targets']):
                        if i < len(action['result']):
                            checked_center_cards[card_idx] = action['result'][i]
        
        return {
            'role_type': 'seer',
            'special_info': {
                'checked_players': checked_players,
                'checked_center_cards': checked_center_cards
            }
        }
    
    def to_vector(self) -> np.ndarray:
        """将预言家观察转换为向量表示"""
        base_vector = super().to_vector()
        
        # 添加查验结果信息
        special_info = self.role_specific_obs['special_info']
        checked_players = special_info.get('checked_players', {})
        
        # 角色映射
        all_roles = self.game_state.config.get('roles', [])
        unique_roles = list(set(all_roles))
        role_map = {role: i for i, role in enumerate(unique_roles)}
        
        # 编码查验过的玩家
        player_check_encoding = np.zeros((self.game_state.num_players, len(unique_roles)))
        
        for player_id, role in checked_players.items():
            if 0 <= player_id < self.game_state.num_players and role in role_map:
                player_check_encoding[player_id, role_map[role]] = 1
        
        return np.concatenate([base_vector, player_check_encoding.flatten()])


class RobberObservation(RoleObservation):
    """强盗观察类"""
    
    def _get_role_specific_observation(self) -> Dict[str, Any]:
        """获取强盗特定的观察信息"""
        stolen_from = None
        new_role = None
        
        # 从行动历史中提取强盗的行动结果
        for action in self.game_state.action_history:
            if action['player_id'] == self.player_id and action['role'] == 'robber':
                if action['action'] == 'swap_role' and 'target' in action and 'result' in action:
                    stolen_from = action['target']
                    new_role = action['result']
                    break
        
        return {
            'role_type': 'robber',
            'special_info': {
                'stolen_from': stolen_from,
                'new_role': new_role
            }
        }
    
    def to_vector(self) -> np.ndarray:
        """将强盗观察转换为向量表示"""
        base_vector = super().to_vector()
        
        # 添加偷取角色信息
        special_info = self.role_specific_obs['special_info']
        stolen_from = special_info.get('stolen_from')
        new_role = special_info.get('new_role')
        
        # 角色映射
        all_roles = self.game_state.config.get('roles', [])
        unique_roles = list(set(all_roles))
        role_map = {role: i for i, role in enumerate(unique_roles)}
        
        # 编码被偷玩家
        stolen_encoding = np.zeros(self.game_state.num_players)
        if stolen_from is not None and 0 <= stolen_from < self.game_state.num_players:
            stolen_encoding[stolen_from] = 1
        
        # 编码新角色
        new_role_encoding = np.zeros(len(unique_roles))
        if new_role in role_map:
            new_role_encoding[role_map[new_role]] = 1
        
        return np.concatenate([base_vector, stolen_encoding, new_role_encoding])


class TroublemakerObservation(RoleObservation):
    """捣蛋鬼观察类"""
    
    def _get_role_specific_observation(self) -> Dict[str, Any]:
        """获取捣蛋鬼特定的观察信息"""
        swapped_players = []
        
        # 从行动历史中提取捣蛋鬼的行动结果
        for action in self.game_state.action_history:
            if action['player_id'] == self.player_id and action['role'] == 'troublemaker':
                if action['action'] == 'swap_roles' and 'targets' in action and action.get('result') == True:
                    swapped_players = action['targets']
                    break
        
        return {
            'role_type': 'troublemaker',
            'special_info': {
                'swapped_players': swapped_players
            }
        }
    
    def to_vector(self) -> np.ndarray:
        """将捣蛋鬼观察转换为向量表示"""
        base_vector = super().to_vector()
        
        # 添加交换玩家信息
        special_info = self.role_specific_obs['special_info']
        swapped_players = special_info.get('swapped_players', [])
        
        # 编码交换的玩家
        swapped_encoding = np.zeros(self.game_state.num_players)
        for player_id in swapped_players:
            if 0 <= player_id < self.game_state.num_players:
                swapped_encoding[player_id] = 1
        
        return np.concatenate([base_vector, swapped_encoding])


class MinionObservation(RoleObservation):
    """爪牙观察类"""
    
    def _get_role_specific_observation(self) -> Dict[str, Any]:
        """获取爪牙特定的观察信息"""
        werewolves = []
        
        # 从行动历史中提取爪牙的查看结果
        for action in self.game_state.action_history:
            if action['player_id'] == self.player_id and action['role'] == 'minion':
                if action['action'] == 'check_werewolves' and 'result' in action:
                    werewolves = action['result']
                    break
        
        return {
            'role_type': 'minion',
            'special_info': {
                'werewolves': werewolves
            }
        }
    
    def to_vector(self) -> np.ndarray:
        """将爪牙观察转换为向量表示"""
        base_vector = super().to_vector()
        
        # 添加狼人信息
        special_info = self.role_specific_obs['special_info']
        werewolves = special_info.get('werewolves', [])
        
        # 编码狼人
        werewolf_encoding = np.zeros(self.game_state.num_players)
        for werewolf_id in werewolves:
            if 0 <= werewolf_id < self.game_state.num_players:
                werewolf_encoding[werewolf_id] = 1
        
        return np.concatenate([base_vector, werewolf_encoding])


class InsomniacObservation(RoleObservation):
    """失眠者观察类"""
    
    def _get_role_specific_observation(self) -> Dict[str, Any]:
        """获取失眠者特定的观察信息"""
        final_role = None
        
        # 从行动历史中提取失眠者的查看结果
        for action in self.game_state.action_history:
            if action['player_id'] == self.player_id and action['role'] == 'insomniac':
                if action['action'] == 'check_final_role' and 'result' in action:
                    final_role = action['result']
                    break
        
        return {
            'role_type': 'insomniac',
            'special_info': {
                'final_role': final_role
            }
        }
    
    def to_vector(self) -> np.ndarray:
        """将失眠者观察转换为向量表示"""
        base_vector = super().to_vector()
        
        # 添加最终角色信息
        special_info = self.role_specific_obs['special_info']
        final_role = special_info.get('final_role')
        
        # 角色映射
        all_roles = self.game_state.config.get('roles', [])
        unique_roles = list(set(all_roles))
        role_map = {role: i for i, role in enumerate(unique_roles)}
        
        # 编码最终角色
        final_role_encoding = np.zeros(len(unique_roles))
        if final_role in role_map:
            final_role_encoding[role_map[final_role]] = 1
        
        return np.concatenate([base_vector, final_role_encoding])


# 角色观察类映射表
ROLE_OBSERVATION_MAP = {
    'villager': VillagerObservation,
    'werewolf': WerewolfObservation,
    'seer': SeerObservation,
    'robber': RobberObservation,
    'troublemaker': TroublemakerObservation,
    'minion': MinionObservation,
    'insomniac': InsomniacObservation
}


def create_role_observation(player_id: int, game_state: GameState) -> RoleObservation:
    """
    根据玩家ID和游戏状态创建相应角色的观察对象
    
    Args:
        player_id: 玩家ID
        game_state: 游戏状态
        
    Returns:
        角色观察对象
    """
    if player_id < 0 or player_id >= len(game_state.players):
        # 默认为村民观察
        return VillagerObservation(player_id, game_state)
    
    role = game_state.players[player_id]['original_role']
    observation_class = ROLE_OBSERVATION_MAP.get(role, VillagerObservation)
    
    return observation_class(player_id, game_state) 