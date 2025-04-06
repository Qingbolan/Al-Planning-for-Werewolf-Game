"""
Werewolf Game Action Definitions
"""
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum, auto
from utils.common import validate_action


class ActionType(Enum):
    """Action type enumeration"""
    # Night action
    NIGHT_ACTION = auto()
    # Day speech
    DAY_SPEECH = auto()
    # Vote
    VOTE = auto()
    # No action
    NO_ACTION = auto()


class Action:
    """Base action class"""
    
    def __init__(self, action_type: ActionType, player_id: int):
        self.action_type = action_type
        self.player_id = player_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'action_type': self.action_type.name,
            'player_id': self.player_id
        }


class NightAction(Action):
    """Night action"""
    
    def __init__(self, player_id: int, action_name: str, action_params: Dict[str, Any]):
        super().__init__(ActionType.NIGHT_ACTION, player_id)
        self.action_name = action_name
        self.action_params = action_params
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = super().to_dict()
        result.update({
            'action_name': self.action_name,
            'action_params': self.action_params
        })
        return result


class DaySpeech(Action):
    """Day speech"""
    
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
        """Convert to dictionary representation"""
        result = super().to_dict()
        result.update({
            'speech_type': self.speech_type,
            'content': self.content
        })
        return result


class VoteAction(Action):
    """Vote action"""
    
    def __init__(self, player_id: int, target_id: int):
        super().__init__(ActionType.VOTE, player_id)
        self.target_id = target_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = super().to_dict()
        result.update({
            'target_id': self.target_id
        })
        return result


class NoAction(Action):
    """No action"""
    
    def __init__(self, player_id: int):
        super().__init__(ActionType.NO_ACTION, player_id)


class SpeechType(Enum):
    """Speech type enumeration"""
    # Role claim
    CLAIM_ROLE = auto()
    # Action result claim
    CLAIM_ACTION_RESULT = auto()
    # Accusation
    ACCUSE = auto()
    # Defense
    DEFEND = auto()
    # Vote intention
    VOTE_INTENTION = auto()


def create_night_action(
    player_id: int, 
    role: str, 
    action_name: str, 
    **kwargs
) -> NightAction:
    """Create night action"""
    
    action_params = {}
    
    # Parameters required for different actions of different roles
    if role == 'werewolf':
        if action_name == 'check_other_werewolves':
            # No additional parameters needed
            pass
        elif action_name == 'check_center_card':
            action_params['card_index'] = kwargs.get('card_index', 0)
            
    elif role == 'minion':
        # Minion does not need additional parameters
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
    
    # Add common parameters
    action_params.update({k: v for k, v in kwargs.items() if k not in action_params})
    
    return NightAction(player_id, action_name, action_params)


def create_speech(
    player_id: int,
    speech_type: str,
    **kwargs
) -> DaySpeech:
    """Create speech"""
    
    content = {}
    
    # Set content based on different speech types
    if speech_type == SpeechType.CLAIM_ROLE.name:
        role = kwargs.get('role', 'villager')
        content['text'] = f"I am a {role}"  # Complete sentence
        content['role'] = role
        
    elif speech_type == SpeechType.CLAIM_ACTION_RESULT.name:
        role = kwargs.get('role', 'villager')
        action = kwargs.get('action', '')
        target = kwargs.get('target', '')
        result = kwargs.get('result', '')
        content['text'] = f"As a {role}, I {action} {target}, and the result is {result}"  # Complete sentence
        content['role'] = role
        content['action'] = action
        content['target'] = target
        content['result'] = result
        
    elif speech_type == SpeechType.ACCUSE.name:
        target_id = kwargs.get('target_id', 0)
        accused_role = kwargs.get('accused_role', 'werewolf')
        content['text'] = f"I think player {target_id} is a {accused_role}"  # Complete sentence
        content['target_id'] = target_id
        content['accused_role'] = accused_role
        
    elif speech_type == SpeechType.DEFEND.name:
        not_role = kwargs.get('not_role', 'werewolf')
        reason = kwargs.get('reason', '')
        content['text'] = f"I am not a {not_role} because {reason}"  # Complete sentence
        content['not_role'] = not_role
        content['reason'] = reason
        
    elif speech_type == SpeechType.VOTE_INTENTION.name:
        target_id = kwargs.get('target_id', 0)
        content['text'] = f"I plan to vote for player {target_id}"  # Complete sentence
        content['target_id'] = target_id
    
    # Add other possible content
    content.update({k: v for k, v in kwargs.items() if k not in content})
    
    return DaySpeech(player_id, speech_type, content)


def create_vote(player_id: int, target_id: int) -> VoteAction:
    """Create vote action"""
    return VoteAction(player_id, target_id)


def create_no_action(player_id: int) -> NoAction:
    """Create no action"""
    return NoAction(player_id)


def process_action(action):
    validate_action(action)
    # Other action processing logic 