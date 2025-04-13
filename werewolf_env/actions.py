"""
Werewolf Game Action Definitions
"""
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum, auto
from utils.common import validate_action
import random


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
        content['type'] = speech_type
        
    elif speech_type == SpeechType.CLAIM_ACTION_RESULT.name:
        role = kwargs.get('role', 'villager')
        action = kwargs.get('action', '')
        target = kwargs.get('target', '')
        result = kwargs.get('result', '')
        
        # 根据不同角色和行动提供更具体的表述
        if role == 'seer':
            if 'player' in str(target):
                player_num = ''.join(filter(str.isdigit, str(target)))
                content['text'] = f"As the Seer, I checked player {player_num} and saw they were a {result}"
            else:
                content['text'] = f"As the Seer, I looked at {target} and discovered {result}"
        elif role == 'robber':
            if 'player' in str(target):
                player_num = ''.join(filter(str.isdigit, str(target)))
                content['text'] = f"As the Robber, I stole from player {player_num} and got the {result} role"
            else:
                content['text'] = f"As the Robber, I stole a card and received the {result} role"
        elif role == 'troublemaker':
            if 'players' in str(target):
                content['text'] = f"As the Troublemaker, I swapped the roles of {target}"
            else:
                content['text'] = f"As the Troublemaker, I switched the roles of {target}"
        elif role == 'insomniac':
            content['text'] = f"As the Insomniac, I woke up at the end of the night and saw my role was {result}"
        else:
            content['text'] = f"As a {role}, I {action} {target}, and found that {result}"
            
        content['role'] = role
        content['action'] = action
        content['target'] = target
        content['result'] = result
        content['type'] = speech_type
        
    elif speech_type == SpeechType.ACCUSE.name:
        target_id = kwargs.get('target_id', 0)
        accused_role = kwargs.get('accused_role', 'werewolf')
        
        # 更多变化的指控方式
        accuse_phrases = [
            f"I suspect player {target_id} is a {accused_role}",
            f"I think player {target_id} is definitely a {accused_role}",
            f"Based on their behavior, player {target_id} seems to be a {accused_role}",
            f"Player {target_id} is acting suspicious, likely a {accused_role}"
        ]
        content['text'] = random.choice(accuse_phrases)
        content['target_id'] = target_id
        content['accused_role'] = accused_role
        content['type'] = speech_type
        
    elif speech_type == SpeechType.DEFEND.name:
        not_role = kwargs.get('not_role', 'werewolf')
        reason = kwargs.get('reason', '')
        
        # 更多变化的辩护方式
        if reason:
            defend_phrases = [
                f"I am not a {not_role} because {reason}",
                f"There's no way I'm a {not_role}. {reason}",
                f"I can assure everyone I'm not a {not_role}. {reason}",
                f"Don't suspect me of being a {not_role}. {reason}"
            ]
            content['text'] = random.choice(defend_phrases)
        else:
            content['text'] = f"I am definitely not a {not_role}"
            
        content['not_role'] = not_role
        content['reason'] = reason
        content['type'] = speech_type
        
    elif speech_type == SpeechType.VOTE_INTENTION.name:
        target_id = kwargs.get('target_id', 0)
        
        # 更多变化的投票意向表达
        vote_phrases = [
            f"I plan to vote for player {target_id}",
            f"I'm voting for player {target_id}",
            f"My vote goes to player {target_id}",
            f"I think we should eliminate player {target_id}"
        ]
        content['text'] = random.choice(vote_phrases)
        content['target_id'] = target_id
        content['type'] = speech_type
    
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