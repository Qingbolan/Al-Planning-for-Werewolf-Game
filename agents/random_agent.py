"""
随机行动的狼人杀智能体
"""
from typing import Dict, List, Any, Tuple
import random

from werewolf_env.actions import (
    Action, 
    create_night_action, create_speech, create_vote, create_no_action,
    SpeechType
)
from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Random action agent"""
    
    def _night_action(self, observation: Dict[str, Any]) -> Action:
        """Random night action"""
        role = self.current_role
        
        # Get available night actions for the role
        if role == 'werewolf':
            actions = ['check_other_werewolves', 'check_center_card']
            action_name = random.choice(actions)
            
            if action_name == 'check_center_card':
                card_index = random.randint(0, 2)  # Randomly select a center card
                action = create_night_action(self.player_id, role, action_name, card_index=card_index)
            else:
                action = create_night_action(self.player_id, role, action_name)
                
        elif role == 'seer':
            actions = ['check_player', 'check_center_cards']
            action_name = random.choice(actions)
            
            if action_name == 'check_player':
                target_id = self.get_random_player_except_self()
                action = create_night_action(self.player_id, role, action_name, target_id=target_id)
            else:
                card_indices = random.sample(range(3), 2)  # Randomly select two center cards
                action = create_night_action(self.player_id, role, action_name, card_indices=card_indices)
                
        elif role == 'robber':
            target_id = self.get_random_player_except_self()
            action = create_night_action(self.player_id, role, 'swap_role', target_id=target_id)
            
        elif role == 'troublemaker':
            # Randomly select two different players to swap roles
            players = [i for i in range(len(self.game_state.players)) if i != self.player_id]
            if len(players) >= 2:
                target_id1, target_id2 = random.sample(players, 2)
                action = create_night_action(self.player_id, role, 'swap_roles', 
                                          target_id1=target_id1, target_id2=target_id2)
            
        elif role == 'minion':
            action = create_night_action(self.player_id, role, 'check_werewolves')
            
        elif role == 'insomniac':
            action = create_night_action(self.player_id, role, 'check_final_role')
            
        # For roles without night actions or other cases
        else:
            action = create_no_action(self.player_id)
        
        self.log_action(action)
        return action
    
    def _day_action(self, observation: Dict[str, Any]) -> Action:
        """Random day speech"""
        # Randomly select speech type
        speech_types = [t.name for t in SpeechType]
        speech_type = random.choice(speech_types)
        
        if speech_type == SpeechType.CLAIM_ROLE.name:
            # Claim a role (could be true or false)
            possible_roles = ['villager', 'seer', 'robber', 'troublemaker', 'insomniac']
            role = random.choice(possible_roles)
            action = create_speech(self.player_id, speech_type, role=role)
            
        elif speech_type == SpeechType.CLAIM_ACTION_RESULT.name:
            # Claim action result
            claimed_role = self.current_role  # Usually claim true role
            action = "check" if claimed_role == 'seer' else "swap" if claimed_role in ['robber', 'troublemaker'] else "inspect"
            target = f"player{random.randint(0, len(self.game_state.players)-1)}"
            result = random.choice(['villager', 'werewolf', 'seer'])
            
            action = create_speech(self.player_id, speech_type, 
                                role=claimed_role, action=action, target=target, result=result)
                                
        elif speech_type == SpeechType.ACCUSE.name:
            # Accuse someone of being a werewolf
            target_id = self.get_random_player_except_self()
            action = create_speech(self.player_id, speech_type, 
                                target_id=target_id, accused_role='werewolf')
                                
        elif speech_type == SpeechType.DEFEND.name:
            # Defend against being a werewolf
            action = create_speech(self.player_id, speech_type, 
                                not_role='werewolf', reason="I am a good person")
                                
        elif speech_type == SpeechType.VOTE_INTENTION.name:
            # Declare voting intention
            target_id = self.get_random_player_except_self()
            action = create_speech(self.player_id, speech_type, target_id=target_id)
        
        # Default claim to be a villager
        else:
            action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role='villager')
        
        self.log_action(action)
        return action
    
    def _vote_action(self, observation: Dict[str, Any]) -> Action:
        """Random voting"""
        target_id = self.get_random_player_except_self()
        action = create_vote(self.player_id, target_id)
        self.log_action(action)
        return action

    def decide_action(self, game_state) -> Tuple[Dict[str, Any], str]:
        """
        决定要执行的动作并提供理由
        
        Args:
            game_state: 游戏状态
            
        Returns:
            Tuple[Dict, str]: (动作，理由)
        """
        # 确保初始化
        if self.original_role is None:
            self.initialize(game_state)
        
        # 获取随机动作
        action = self.get_action(game_state)
        
        # 随机理由
        reasons = [
            "随机选择的动作",
            "这看起来是个好主意",
            "我觉得这样做很有趣",
            "随机策略",
            "这是我的直觉"
        ]
        reasoning = random.choice(reasons)
        
        # 转换Action对象为字典（如果需要）
        if not isinstance(action, dict):
            if hasattr(action, 'to_dict'):
                action_dict = action.to_dict()
            else:
                # 手动转换
                action_dict = {
                    'action_type': getattr(action, 'action_type', 'UNKNOWN'),
                }
                
                # 根据不同类型添加特定属性
                if hasattr(action, 'action_name'):
                    action_dict['action_name'] = action.action_name
                    action_dict['action_params'] = getattr(action, 'action_params', {})
                elif hasattr(action, 'content'):
                    action_dict['speech_type'] = action.content.get('type', 'GENERAL')
                    action_dict['content'] = action.content
                elif hasattr(action, 'target_id'):
                    action_dict['target_id'] = action.target_id
        else:
            action_dict = action
        
        return action_dict, reasoning 