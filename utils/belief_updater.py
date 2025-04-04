"""
狼人杀游戏信念状态更新器
用于跟踪和更新玩家对其他玩家角色的信念
"""
from typing import Dict, List, Any, Set, Tuple, Optional
import numpy as np
from collections import defaultdict
import copy

from werewolf_env.state import GameState
from werewolf_env.actions import SpeechType


class BeliefState:
    """信念状态类，表示对游戏中角色分布的信念"""
    
    def __init__(self, game_state: GameState, player_id: int):
        """
        初始化信念状态
        
        Args:
            game_state: 游戏状态
            player_id: 玩家ID
        """
        self.player_id = player_id
        self.num_players = game_state.num_players
        self.possible_roles = list(set(game_state.roles))
        self.original_role = game_state.players[player_id]['original_role']
        
        # 为每个玩家初始化角色概率分布（均匀分布）
        self.beliefs = {}
        for p_id in range(self.num_players):
            if p_id == player_id:
                # 对自己的角色有确定信念
                self.beliefs[p_id] = {role: 1.0 if role == self.original_role else 0.0 
                                     for role in self.possible_roles}
            else:
                # 对其他玩家的角色有均匀分布的信念
                self.beliefs[p_id] = {role: 1.0 / len(self.possible_roles) 
                                     for role in self.possible_roles}
        
        # 确定的角色（通过夜晚行动或其他确定方式得知）
        self.certain_roles = {player_id: self.original_role}
        
        # 可能的中央牌堆角色
        self.center_card_beliefs = {}
        for i in range(game_state.num_center_cards):
            self.center_card_beliefs[i] = {role: 1.0 / len(self.possible_roles) 
                                         for role in self.possible_roles}
        
        # 已知的信息
        self.known_info = []
        
        # 记录声称的角色
        self.claimed_roles = {}
        
        # 记录特殊角色行动的声称
        self.claimed_actions = defaultdict(list)
        
        # 记录投票历史
        self.vote_history = {}
    
    def normalize_beliefs(self) -> None:
        """归一化所有信念概率使其总和为1"""
        for player_id in self.beliefs:
            total = sum(self.beliefs[player_id].values())
            if total > 0:
                for role in self.beliefs[player_id]:
                    self.beliefs[player_id][role] /= total
        
        # 归一化中央牌堆信念
        for card_idx in self.center_card_beliefs:
            total = sum(self.center_card_beliefs[card_idx].values())
            if total > 0:
                for role in self.center_card_beliefs[card_idx]:
                    self.center_card_beliefs[card_idx][role] /= total
    
    def update_with_certain_role(self, player_id: int, role: str) -> None:
        """
        使用确定的角色信息更新信念
        
        Args:
            player_id: 玩家ID
            role: 确定的角色
        """
        if player_id < 0 or player_id >= self.num_players:
            return
            
        # 更新确定角色字典
        self.certain_roles[player_id] = role
        
        # 更新该玩家的信念分布
        for r in self.beliefs[player_id]:
            self.beliefs[player_id][r] = 1.0 if r == role else 0.0
        
        # 更新已知信息
        self.known_info.append({
            'type': 'certain_role',
            'player_id': player_id,
            'role': role
        })
        
        # 更新其他玩家的信念（同一角色不能由多个玩家担任）
        for p_id in self.beliefs:
            if p_id != player_id:
                if role in self.beliefs[p_id]:
                    # 降低该玩家为该角色的概率（不完全排除以处理可能的不确定性）
                    self.beliefs[p_id][role] *= 0.1
        
        # 归一化信念
        self.normalize_beliefs()
    
    def update_with_center_card(self, card_idx: int, role: str) -> None:
        """
        更新中央牌堆的信念
        
        Args:
            card_idx: 牌堆索引
            role: 确定的角色
        """
        if card_idx not in self.center_card_beliefs:
            return
            
        # 更新该卡牌的信念
        for r in self.center_card_beliefs[card_idx]:
            self.center_card_beliefs[card_idx][r] = 1.0 if r == role else 0.0
        
        # 更新已知信息
        self.known_info.append({
            'type': 'center_card',
            'card_idx': card_idx,
            'role': role
        })
        
        # 更新玩家信念（该角色在中央牌堆，不太可能由玩家担任）
        for p_id in self.beliefs:
            if role in self.beliefs[p_id]:
                # 降低该玩家为该角色的概率
                self.beliefs[p_id][role] *= 0.5
        
        # 归一化信念
        self.normalize_beliefs()
    
    def update_with_role_swap(self, player_id1: int, player_id2: int) -> None:
        """
        更新角色交换后的信念
        
        Args:
            player_id1: 第一个玩家ID
            player_id2: 第二个玩家ID
        """
        if player_id1 < 0 or player_id1 >= self.num_players or player_id2 < 0 or player_id2 >= self.num_players:
            return
            
        # 交换信念
        self.beliefs[player_id1], self.beliefs[player_id2] = self.beliefs[player_id2], self.beliefs[player_id1]
        
        # 更新确定角色（如果有的话）
        if player_id1 in self.certain_roles and player_id2 in self.certain_roles:
            self.certain_roles[player_id1], self.certain_roles[player_id2] = self.certain_roles[player_id2], self.certain_roles[player_id1]
        elif player_id1 in self.certain_roles:
            self.certain_roles[player_id2] = self.certain_roles.pop(player_id1)
        elif player_id2 in self.certain_roles:
            self.certain_roles[player_id1] = self.certain_roles.pop(player_id2)
        
        # 更新已知信息
        self.known_info.append({
            'type': 'role_swap',
            'player_id1': player_id1,
            'player_id2': player_id2
        })


class RoleSpecificBeliefUpdater:
    """角色特定的信念更新器基类"""
    
    def __init__(self, player_id: int, game_state: GameState):
        """
        初始化信念更新器
        
        Args:
            player_id: 玩家ID
            game_state: 游戏状态
        """
        self.player_id = player_id
        self.game_state = game_state
        self.belief_state = BeliefState(game_state, player_id)
        self.role = game_state.players[player_id]['original_role']
    
    def update_with_night_action(self, action_result: Dict[str, Any]) -> None:
        """
        根据夜晚行动结果更新信念
        
        Args:
            action_result: 行动结果
        """
        # 基类方法，子类可以重写
        pass
    
    def update_with_speech(self, speaker_id: int, speech_content: Dict[str, Any]) -> None:
        """
        根据发言更新信念
        
        Args:
            speaker_id: 发言者ID
            speech_content: 发言内容
        """
        # 记录玩家声称的角色
        if speech_content.get('type') == SpeechType.CLAIM_ROLE.name and 'role' in speech_content:
            self.belief_state.claimed_roles[speaker_id] = speech_content['role']
            
            # 更新信念（轻微提高该角色的概率）
            role = speech_content['role']
            if role in self.belief_state.beliefs[speaker_id]:
                # 增加该角色的概率
                self.belief_state.beliefs[speaker_id][role] *= 1.2
                
                # 如果玩家声称自己是狼人（不太可能），降低相信度
                if role == 'werewolf':
                    self.belief_state.beliefs[speaker_id][role] *= 0.5
        
        # 处理行动结果声称
        elif speech_content.get('type') == SpeechType.CLAIM_ACTION_RESULT.name:
            if 'role' in speech_content and 'action' in speech_content and 'target' in speech_content and 'result' in speech_content:
                self.belief_state.claimed_actions[speaker_id].append({
                    'role': speech_content['role'],
                    'action': speech_content['action'],
                    'target': speech_content['target'],
                    'result': speech_content['result']
                })
                
                # 如果声称查验结果，更新信念
                if speech_content['role'] == 'seer' and speech_content['action'] in ['查验', '查看']:
                    # 尝试解析目标玩家ID
                    target_id = -1
                    target = speech_content['target']
                    if isinstance(target, int):
                        target_id = target
                    elif isinstance(target, str) and target.startswith('玩家') and target[2:].isdigit():
                        target_id = int(target[2:])
                    
                    if 0 <= target_id < self.game_state.num_players:
                        claimed_role = speech_content['result']
                        
                        # 根据自己的角色和信息判断这个声称的可信度
                        if self.role == 'seer':
                            # 如果自己是预言家，知道这个声称是否真实
                            # 这里简化处理，实际上需要根据自己的查验结果来判断
                            credibility = 0.1  # 假设可信度低
                        else:
                            # 如果自己不是预言家，根据其他信息来判断可信度
                            credibility = 0.5  # 中等可信度
                        
                        # 更新目标玩家的角色信念
                        if claimed_role in self.belief_state.beliefs[target_id]:
                            # 按可信度调整概率
                            self.belief_state.beliefs[target_id][claimed_role] *= (1.0 + credibility)
        
        # 处理指控
        elif speech_content.get('type') == SpeechType.ACCUSE.name:
            if 'target_id' in speech_content and 'accused_role' in speech_content:
                target_id = speech_content['target_id']
                accused_role = speech_content['accused_role']
                
                if 0 <= target_id < self.game_state.num_players and accused_role in self.belief_state.beliefs[target_id]:
                    # 增加该玩家为被指控角色的概率
                    self.belief_state.beliefs[target_id][accused_role] *= 1.1
        
        # 归一化信念
        self.belief_state.normalize_beliefs()
    
    def update_with_votes(self, votes: Dict[int, int]) -> None:
        """
        根据投票更新信念
        
        Args:
            votes: 投票结果，键为投票者ID，值为目标ID
        """
        # 记录投票历史
        self.belief_state.vote_history.update(votes)
        
        # 分析投票模式
        vote_counts = defaultdict(int)
        for target_id in votes.values():
            vote_counts[target_id] += 1
        
        # 玩家投票给谁可能反映他们的阵营
        for voter_id, target_id in votes.items():
            if voter_id == self.player_id:
                continue  # 跳过自己的投票
                
            # 根据投票目标调整信念
            # 如果投票给可能的狼人，增加该玩家是好人的概率
            werewolf_prob = self.belief_state.beliefs[target_id].get('werewolf', 0.0)
            if werewolf_prob > 0.5:
                # 增加投票者是村民阵营的概率
                for role in ['villager', 'seer', 'robber', 'troublemaker', 'insomniac']:
                    if role in self.belief_state.beliefs[voter_id]:
                        self.belief_state.beliefs[voter_id][role] *= 1.1
                        
                # 降低投票者是狼人阵营的概率
                for role in ['werewolf', 'minion']:
                    if role in self.belief_state.beliefs[voter_id]:
                        self.belief_state.beliefs[voter_id][role] *= 0.9
            
            # 如果投票给可能的预言家，增加该玩家是狼人的概率
            seer_prob = self.belief_state.beliefs[target_id].get('seer', 0.0)
            if seer_prob > 0.5:
                # 增加投票者是狼人阵营的概率
                for role in ['werewolf', 'minion']:
                    if role in self.belief_state.beliefs[voter_id]:
                        self.belief_state.beliefs[voter_id][role] *= 1.1
                        
                # 降低投票者是村民阵营的概率
                for role in ['villager', 'seer', 'robber', 'troublemaker', 'insomniac']:
                    if role in self.belief_state.beliefs[voter_id]:
                        self.belief_state.beliefs[voter_id][role] *= 0.9
        
        # 归一化信念
        self.belief_state.normalize_beliefs()
    
    def get_action_probabilities(self, action_space: List[Any]) -> Dict[Any, float]:
        """
        获取行动概率分布
        
        Args:
            action_space: 可用行动空间
            
        Returns:
            行动概率分布
        """
        # 基类方法，子类可以重写
        return {action: 1.0 / len(action_space) for action in action_space}


class VillagerBeliefUpdater(RoleSpecificBeliefUpdater):
    """村民信念更新器"""
    
    def update_with_night_action(self, action_result: Dict[str, Any]) -> None:
        """村民没有夜晚行动"""
        pass


class WerewolfBeliefUpdater(RoleSpecificBeliefUpdater):
    """狼人信念更新器"""
    
    def update_with_night_action(self, action_result: Dict[str, Any]) -> None:
        """根据狼人夜晚行动结果更新信念"""
        action = action_result.get('action', '')
        
        if action == 'check_other_werewolves':
            # 更新其他狼人的信息
            werewolves = action_result.get('result', [])
            for werewolf_id in werewolves:
                if 0 <= werewolf_id < self.game_state.num_players:
                    self.belief_state.update_with_certain_role(werewolf_id, 'werewolf')
        
        elif action == 'check_center_card':
            # 更新中央牌堆信息
            card_index = action_result.get('card_index', -1)
            role = action_result.get('result', '')
            if card_index >= 0 and role:
                self.belief_state.update_with_center_card(card_index, role)


class SeerBeliefUpdater(RoleSpecificBeliefUpdater):
    """预言家信念更新器"""
    
    def update_with_night_action(self, action_result: Dict[str, Any]) -> None:
        """根据预言家夜晚行动结果更新信念"""
        action = action_result.get('action', '')
        
        if action == 'check_player':
            # 更新玩家角色信息
            target_id = action_result.get('target', -1)
            result = action_result.get('result', '')
            if target_id >= 0 and result:
                self.belief_state.update_with_certain_role(target_id, result)
        
        elif action == 'check_center_cards':
            # 更新中央牌堆信息
            targets = action_result.get('targets', [])
            results = action_result.get('result', [])
            for i, card_idx in enumerate(targets):
                if i < len(results):
                    self.belief_state.update_with_center_card(card_idx, results[i])


class RobberBeliefUpdater(RoleSpecificBeliefUpdater):
    """强盗信念更新器"""
    
    def update_with_night_action(self, action_result: Dict[str, Any]) -> None:
        """根据强盗夜晚行动结果更新信念"""
        action = action_result.get('action', '')
        
        if action == 'swap_role':
            # 更新角色交换信息
            target_id = action_result.get('target', -1)
            result = action_result.get('result', '')
            if target_id >= 0 and result:
                # 知道目标玩家原来的角色
                self.belief_state.update_with_certain_role(target_id, result)
                
                # 自己的角色变成了目标玩家的角色
                self.belief_state.update_with_certain_role(self.player_id, result)
                
                # 目标玩家的角色变成了强盗
                self.belief_state.update_with_certain_role(target_id, 'robber')


class TroublemakerBeliefUpdater(RoleSpecificBeliefUpdater):
    """捣蛋鬼信念更新器"""
    
    def update_with_night_action(self, action_result: Dict[str, Any]) -> None:
        """根据捣蛋鬼夜晚行动结果更新信念"""
        action = action_result.get('action', '')
        
        if action == 'swap_roles':
            # 更新角色交换信息
            targets = action_result.get('targets', [])
            result = action_result.get('result', False)
            
            if len(targets) == 2 and result:
                target_id1, target_id2 = targets
                
                # 知道两个玩家交换了角色，但不知道他们的具体角色
                # 只能更新信念状态来反映这一点
                self.belief_state.update_with_role_swap(target_id1, target_id2)


class MinionBeliefUpdater(RoleSpecificBeliefUpdater):
    """爪牙信念更新器"""
    
    def update_with_night_action(self, action_result: Dict[str, Any]) -> None:
        """根据爪牙夜晚行动结果更新信念"""
        action = action_result.get('action', '')
        
        if action == 'check_werewolves':
            # 更新狼人信息
            werewolves = action_result.get('result', [])
            for werewolf_id in werewolves:
                if 0 <= werewolf_id < self.game_state.num_players:
                    self.belief_state.update_with_certain_role(werewolf_id, 'werewolf')


class InsomniacBeliefUpdater(RoleSpecificBeliefUpdater):
    """失眠者信念更新器"""
    
    def update_with_night_action(self, action_result: Dict[str, Any]) -> None:
        """根据失眠者夜晚行动结果更新信念"""
        action = action_result.get('action', '')
        
        if action == 'check_final_role':
            # 更新自己的最终角色
            result = action_result.get('result', '')
            if result:
                self.belief_state.update_with_certain_role(self.player_id, result)


# 角色信念更新器映射表
BELIEF_UPDATER_MAP = {
    'villager': VillagerBeliefUpdater,
    'werewolf': WerewolfBeliefUpdater,
    'minion': MinionBeliefUpdater,
    'seer': SeerBeliefUpdater,
    'robber': RobberBeliefUpdater,
    'troublemaker': TroublemakerBeliefUpdater,
    'insomniac': InsomniacBeliefUpdater
}


def create_belief_updater(player_id: int, game_state: GameState) -> RoleSpecificBeliefUpdater:
    """
    根据玩家角色创建对应的信念更新器
    
    Args:
        player_id: 玩家ID
        game_state: 游戏状态
        
    Returns:
        信念更新器实例
    """
    if player_id < 0 or player_id >= len(game_state.players):
        # 默认为村民信念更新器
        return VillagerBeliefUpdater(player_id, game_state)
    
    role = game_state.players[player_id]['original_role']
    updater_class = BELIEF_UPDATER_MAP.get(role, VillagerBeliefUpdater)
    
    return updater_class(player_id, game_state)