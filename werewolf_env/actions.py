"""
狼人杀游戏行动定义
"""
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum, auto
from utils.common import validate_action


class ActionType(Enum):
    """行动类型枚举"""
    # 夜晚行动
    NIGHT_ACTION = auto()
    # 白天发言
    DAY_SPEECH = auto()
    # 投票
    VOTE = auto()
    # 无行动
    NO_ACTION = auto()


class Action:
    """行动基类"""
    
    def __init__(self, action_type: ActionType, player_id: int):
        self.action_type = action_type
        self.player_id = player_id
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            'action_type': self.action_type.name,
            'player_id': self.player_id
        }


class NightAction(Action):
    """夜晚行动"""
    
    def __init__(self, player_id: int, action_name: str, action_params: Dict[str, Any]):
        super().__init__(ActionType.NIGHT_ACTION, player_id)
        self.action_name = action_name
        self.action_params = action_params
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        result = super().to_dict()
        result.update({
            'action_name': self.action_name,
            'action_params': self.action_params
        })
        return result


class DaySpeech(Action):
    """白天发言"""
    
    def __init__(
        self, 
        player_id: int, 
        speech_type: str,
        content: Dict[str, Any]
    ):
        super().__init__(ActionType.DAY_SPEECH, player_id)
        self.speech_type = speech_type
        self.content = content
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        result = super().to_dict()
        result.update({
            'speech_type': self.speech_type,
            'content': self.content
        })
        return result


class VoteAction(Action):
    """投票行动"""
    
    def __init__(self, player_id: int, target_id: int):
        super().__init__(ActionType.VOTE, player_id)
        self.target_id = target_id
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        result = super().to_dict()
        result.update({
            'target_id': self.target_id
        })
        return result


class NoAction(Action):
    """无行动"""
    
    def __init__(self, player_id: int):
        super().__init__(ActionType.NO_ACTION, player_id)


class SpeechType(Enum):
    """发言类型枚举"""
    # 角色声明
    CLAIM_ROLE = auto()
    # 行动结果声明
    CLAIM_ACTION_RESULT = auto()
    # 指控
    ACCUSE = auto()
    # 辩解
    DEFEND = auto()
    # 投票意向
    VOTE_INTENTION = auto()


def create_night_action(
    player_id: int, 
    role: str, 
    action_name: str, 
    **kwargs
) -> NightAction:
    """创建夜晚行动"""
    
    action_params = {}
    
    # 不同角色的不同行动所需参数
    if role == 'werewolf':
        if action_name == 'check_other_werewolves':
            # 不需要额外参数
            pass
        elif action_name == 'check_center_card':
            action_params['card_index'] = kwargs.get('card_index', 0)
            
    elif role == 'minion':
        # 爪牙不需要额外参数
        pass
        
    elif role == 'seer':
        if action_name == 'check_player':
            action_params['target_id'] = kwargs.get('target_id', 0)
        elif action_name == 'check_center_cards':
            action_params['card_indices'] = kwargs.get('card_indices', [0, 1])
            
    elif role == 'robber':
        action_params['target_id'] = kwargs.get('target_id', 0)
        
    elif role == 'troublemaker':
        action_params['target_id1'] = kwargs.get('target_id1', 0)
        action_params['target_id2'] = kwargs.get('target_id2', 1)
    
    # 添加通用参数
    action_params.update({k: v for k, v in kwargs.items() if k not in action_params})
    
    return NightAction(player_id, action_name, action_params)


def create_speech(
    player_id: int,
    speech_type: str,
    **kwargs
) -> DaySpeech:
    """创建发言"""
    
    content = {}
    
    # 根据不同发言类型设置内容
    if speech_type == SpeechType.CLAIM_ROLE.name:
        content['role'] = kwargs.get('role', 'villager')
        
    elif speech_type == SpeechType.CLAIM_ACTION_RESULT.name:
        content['role'] = kwargs.get('role', 'villager')
        content['action'] = kwargs.get('action', '')
        content['target'] = kwargs.get('target', '')
        content['result'] = kwargs.get('result', '')
        
    elif speech_type == SpeechType.ACCUSE.name:
        content['target_id'] = kwargs.get('target_id', 0)
        content['accused_role'] = kwargs.get('accused_role', 'werewolf')
        
    elif speech_type == SpeechType.DEFEND.name:
        content['not_role'] = kwargs.get('not_role', 'werewolf')
        content['reason'] = kwargs.get('reason', '')
        
    elif speech_type == SpeechType.VOTE_INTENTION.name:
        content['target_id'] = kwargs.get('target_id', 0)
    
    # 添加其他可能的内容
    content.update({k: v for k, v in kwargs.items() if k not in content})
    
    return DaySpeech(player_id, speech_type, content)


def create_vote(player_id: int, target_id: int) -> VoteAction:
    """创建投票行动"""
    return VoteAction(player_id, target_id)


def create_no_action(player_id: int) -> NoAction:
    """创建无行动"""
    return NoAction(player_id)


def process_action(action):
    validate_action(action)
    # 处理动作的其他逻辑 