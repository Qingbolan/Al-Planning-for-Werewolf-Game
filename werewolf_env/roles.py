"""
狼人杀游戏角色定义
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

class Role(ABC):
    """角色基类"""
    
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.original_role_name = self.__class__.__name__.lower()
        self.current_role_name = self.original_role_name
        self.team = self._get_team()
        
    def _get_team(self) -> str:
        """获取角色所属阵营"""
        if self.current_role_name in ['werewolf', 'minion']:
            return 'werewolf_team'
        return 'villager_team'
    
    @abstractmethod
    def night_action(self, game_state: Dict[str, Any], action_params: Dict[str, Any]) -> Dict[str, Any]:
        """夜晚行动，返回行动结果"""
        pass
    
    @property
    def has_night_action(self) -> bool:
        """是否有夜间行动"""
        return True


class Villager(Role):
    """村民角色"""
    
    def night_action(self, game_state: Dict[str, Any], action_params: Dict[str, Any]) -> Dict[str, Any]:
        """村民没有夜间行动"""
        return {'action': 'no_action', 'result': None}
    
    @property
    def has_night_action(self) -> bool:
        """村民没有夜间行动"""
        return False


class Werewolf(Role):
    """狼人角色"""
    
    def night_action(self, game_state: Dict[str, Any], action_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        狼人夜晚行动：
        1. 查看其他狼人
        2. 如果只有一个狼人，可以查看中央牌堆一张牌
        """
        action = action_params.get('action', 'check_other_werewolves')
        
        if action == 'check_other_werewolves':
            # 获取其他狼人玩家ID
            werewolves = []
            for player_id, player_info in enumerate(game_state['players']):
                if player_id != self.player_id and player_info['current_role'] == 'werewolf':
                    werewolves.append(player_id)
            
            return {
                'action': 'check_other_werewolves',
                'result': werewolves
            }
            
        elif action == 'check_center_card':
            # 查看中央牌堆一张牌
            card_index = action_params.get('card_index', 0)
            if 0 <= card_index < len(game_state['center_cards']):
                return {
                    'action': 'check_center_card',
                    'result': game_state['center_cards'][card_index]
                }
            
        return {'action': action, 'result': None}


class Minion(Role):
    """爪牙角色"""
    
    def night_action(self, game_state: Dict[str, Any], action_params: Dict[str, Any]) -> Dict[str, Any]:
        """爪牙夜晚行动：查看所有狼人"""
        werewolves = []
        for player_id, player_info in enumerate(game_state['players']):
            if player_info['current_role'] == 'werewolf':
                werewolves.append(player_id)
                
        return {
            'action': 'check_werewolves',
            'result': werewolves
        }


class Seer(Role):
    """预言家角色"""
    
    def night_action(self, game_state: Dict[str, Any], action_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        预言家夜晚行动：
        1. 查看一名玩家的角色
        2. 查看中央牌堆两张牌
        """
        action = action_params.get('action', 'check_player')
        
        if action == 'check_player':
            # 查看一名玩家的角色
            target_id = action_params.get('target_id', 0)
            if 0 <= target_id < len(game_state['players']) and target_id != self.player_id:
                return {
                    'action': 'check_player',
                    'target': target_id,
                    'result': game_state['players'][target_id]['current_role']
                }
                
        elif action == 'check_center_cards':
            # 查看中央牌堆两张牌
            card_indices = action_params.get('card_indices', [0, 1])
            cards = []
            for idx in card_indices[:2]:  # 最多查看两张
                if 0 <= idx < len(game_state['center_cards']):
                    cards.append(game_state['center_cards'][idx])
            
            return {
                'action': 'check_center_cards',
                'targets': card_indices[:2],
                'result': cards
            }
            
        return {'action': action, 'result': None}


class Robber(Role):
    """强盗角色"""
    
    def night_action(self, game_state: Dict[str, Any], action_params: Dict[str, Any]) -> Dict[str, Any]:
        """强盗夜晚行动：与另一名玩家交换角色"""
        target_id = action_params.get('target_id', 0)
        
        if 0 <= target_id < len(game_state['players']) and target_id != self.player_id:
            # 获取目标玩家当前角色
            target_role = game_state['players'][target_id]['current_role']
            
            # 更新角色（实际交换在环境中进行）
            return {
                'action': 'swap_role',
                'target': target_id,
                'result': target_role
            }
            
        return {'action': 'swap_role', 'result': None}


class Troublemaker(Role):
    """捣蛋鬼角色"""
    
    def night_action(self, game_state: Dict[str, Any], action_params: Dict[str, Any]) -> Dict[str, Any]:
        """捣蛋鬼夜晚行动：交换两名其他玩家的角色"""
        target_id1 = action_params.get('target_id1', 0)
        target_id2 = action_params.get('target_id2', 1)
        
        if (0 <= target_id1 < len(game_state['players']) and 
            0 <= target_id2 < len(game_state['players']) and 
            target_id1 != target_id2 and
            target_id1 != self.player_id and
            target_id2 != self.player_id):
            
            # 更新角色（实际交换在环境中进行）
            return {
                'action': 'swap_roles',
                'targets': [target_id1, target_id2],
                'result': True
            }
            
        return {'action': 'swap_roles', 'result': False}


class Insomniac(Role):
    """失眠者角色"""
    
    def night_action(self, game_state: Dict[str, Any], action_params: Dict[str, Any]) -> Dict[str, Any]:
        """失眠者夜晚行动：查看自己的最终角色"""
        return {
            'action': 'check_final_role',
            'result': game_state['players'][self.player_id]['current_role']
        }


# 角色类映射表
ROLE_MAP = {
    'villager': Villager,
    'werewolf': Werewolf,
    'minion': Minion,
    'seer': Seer,
    'robber': Robber,
    'troublemaker': Troublemaker,
    'insomniac': Insomniac
}


def create_role(role_name: str, player_id: int) -> Role:
    """根据角色名称创建角色实例"""
    if role_name in ROLE_MAP:
        return ROLE_MAP[role_name](player_id)
    else:
        # 默认为村民
        return Villager(player_id) 