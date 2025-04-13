"""
Base class for Werewolf game agents
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import random
from abc import ABC, abstractmethod
from collections import defaultdict

from werewolf_env.state import GameState, PlayerObservation
from werewolf_env.actions import (
    ActionType, Action, NightAction, DaySpeech, VoteAction, NoAction,
    create_night_action, create_speech, create_vote, create_no_action,
    SpeechType
)
from utils.belief_updater import (
    BeliefState, RoleSpecificBeliefUpdater, create_belief_updater
)


class BaseAgent(ABC):
    """Base class for agents"""
    
    def __init__(self, player_id: int):
        """
        Initialize the agent
        
        Args:
            player_id: Player ID
        """
        self.player_id = player_id
        self.belief_updater = None
        self.game_state = None
        self.original_role = None
        self.current_role = None
        self.current_phase = None
        self.action_history = []
    
    def initialize(self, game_state: GameState) -> None:
        """
        Initialize agent state
        
        Args:
            game_state: Game state
        """
        self.game_state = game_state
        self.original_role = game_state.players[self.player_id]['original_role']
        self.current_role = game_state.players[self.player_id]['current_role']
        self.current_phase = game_state.phase
        
        # Create role-specific belief updater
        self.belief_updater = create_belief_updater(self.player_id, game_state)
    
    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Receive observation information
        
        Args:
            observation: Observation information
        """
        # Update current role and phase
        self.current_role = observation.get('original_role', self.current_role)
        self.current_phase = observation.get('phase', self.current_phase)
        
        # Process night action results
        if 'action_result' in observation:
            self.belief_updater.update_with_night_action(observation['action_result'])
        
        # Process speech history
        if 'speech_history' in observation:
            for speech in observation['speech_history']:
                if speech not in self.action_history:  # Avoid duplicate processing
                    self.action_history.append(speech)
                    self.belief_updater.update_with_speech(speech['player_id'], speech['content'])
        
        # Process voting information
        if 'votes' in observation:
            self.belief_updater.update_with_votes(observation['votes'])
    
    def act(self, observation: Dict[str, Any]) -> Action:
        """
        Choose action based on observation
        
        Args:
            observation: Observation information
            
        Returns:
            Action
        """
        # Update observation
        self.observe(observation)
        
        # Get current phase
        phase = observation.get('phase', self.current_phase)
        
        # Choose action based on phase
        if phase == 'night':
            return self._night_action(observation)
        elif phase == 'day':
            return self._day_action(observation)
        elif phase == 'vote':
            return self._vote_action(observation)
        else:
            return create_no_action(self.player_id)
    
    @abstractmethod
    def _night_action(self, observation: Dict[str, Any]) -> Action:
        """
        Choose night action
        
        Args:
            observation: Observation information
            
        Returns:
            Night action
        """
        pass
    
    @abstractmethod
    def _day_action(self, observation: Dict[str, Any]) -> Action:
        """
        Choose day speech
        
        Args:
            observation: Observation information
            
        Returns:
            Day speech
        """
        pass
    
    @abstractmethod
    def _vote_action(self, observation: Dict[str, Any]) -> Action:
        """
        Choose voting target
        
        Args:
            observation: Observation information
            
        Returns:
            Voting action
        """
        pass
    
    def get_role_probabilities(self, player_id: int) -> Dict[str, float]:
        """
        Get role probability distribution for a player
        
        Args:
            player_id: Player ID
            
        Returns:
            Role probability distribution
        """
        if self.belief_updater and player_id in self.belief_updater.belief_state.beliefs:
            return dict(self.belief_updater.belief_state.beliefs[player_id])
        return {}
    
    def get_most_suspected_werewolf(self) -> Tuple[int, float]:
        """
        Get the player most likely to be a werewolf
        
        Returns:
            (Player ID, Probability)
        """
        if not self.belief_updater:
            return -1, 0.0
        
        max_prob = -1
        max_player = -1
        
        for player_id in self.belief_updater.belief_state.beliefs:
            if player_id == self.player_id:
                continue  # Skip self
                
            werewolf_prob = self.belief_updater.belief_state.beliefs[player_id].get('werewolf', 0.0)
            if werewolf_prob > max_prob:
                max_prob = werewolf_prob
                max_player = player_id
        
        return max_player, max_prob
    
    def get_random_player_except_self(self) -> int:
        """
        Randomly select a player other than self
        
        Returns:
            Player ID
        """
        if not self.game_state:
            return random.randint(0, 5)  # Default random selection
            
        num_players = len(self.game_state.players)
        other_players = [i for i in range(num_players) if i != self.player_id]
        
        if other_players:
            return random.choice(other_players)
        return -1  # Return -1 if only self remains
    
    def get_action_probabilities(self, action_space: List[Any]) -> np.ndarray:
        """
        Calculate action probability distribution
        
        Args:
            action_space: Available action space
            
        Returns:
            Action probability distribution
        """
        if self.belief_updater:
            action_probs = self.belief_updater.get_action_probabilities(action_space)
            return np.array([action_probs.get(action, 0.0) for action in action_space])
        
        # Default uniform distribution
        return np.ones(len(action_space)) / len(action_space)

    def log_action(self, action: Action):
        # Common action logging logic
        print(f"Agent {self.player_id} executes action {action}")

    def get_action(self, game_state: GameState) -> Action:
        """
        获取动作 - 兼容训练器接口
        
        Args:
            game_state: 游戏状态
            
        Returns:
            Action
        """
        # 将GameState转换为观察结果格式
        observation = {
            'phase': game_state.phase,
            'round': game_state.round,
            'current_player': game_state.get_current_player(),
            'speech_round': getattr(game_state, 'speech_round', 0),
            'votes': getattr(game_state, 'votes', {}),
        }
        
        # 加入角色信息
        if self.player_id < len(game_state.players):
            player_data = game_state.players[self.player_id]
            observation['original_role'] = player_data['original_role']
            observation['current_role'] = player_data['current_role']
        
        # 调用act方法获取动作
        return self.act(observation)


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


class HeuristicAgent(BaseAgent):
    """Rule-based agent"""
    
    def _night_action(self, observation: Dict[str, Any]) -> Action:
        """Rule-based night action"""
        role = self.current_role
        
        if role == 'werewolf':
            # If other werewolves are known, choose to check other werewolves
            # If the only werewolf, check center card pile
            other_werewolves_exist = False
            for player in self.game_state.players:
                if player['id'] != self.player_id and player['original_role'] == 'werewolf':
                    other_werewolves_exist = True
                    break
            
            if other_werewolves_exist:
                action = create_night_action(self.player_id, role, 'check_other_werewolves')
            else:
                # If the only werewolf, randomly check a center card
                card_index = random.randint(0, 2)
                action = create_night_action(self.player_id, role, 'check_center_card', card_index=card_index)
        
        elif role == 'seer':
            # Prioritize checking suspicious players, if none then randomly check or check center cards
            if random.random() < 0.7:  # 70% chance to check players
                # Try to find a suspicious player
                suspected_player, prob = self.get_most_suspected_werewolf()
                if suspected_player >= 0 and prob > 0.3:
                    target_id = suspected_player
                else:
                    # Randomly select a non-self player
                    target_id = self.get_random_player_except_self()
                
                action = create_night_action(self.player_id, role, 'check_player', target_id=target_id)
            else:
                # Check center cards
                card_indices = random.sample(range(3), 2)
                action = create_night_action(self.player_id, role, 'check_center_cards', card_indices=card_indices)
        
        elif role == 'robber':
            # Try to find a player who doesn't appear to be a werewolf to steal from
            best_target = -1
            lowest_werewolf_prob = 1.0
            
            for player_id in range(len(self.game_state.players)):
                if player_id == self.player_id:
                    continue
                
                werewolf_prob = self.get_role_probabilities(player_id).get('werewolf', 0.5)
                if werewolf_prob < lowest_werewolf_prob:
                    lowest_werewolf_prob = werewolf_prob
                    best_target = player_id
            
            if best_target >= 0:
                action = create_night_action(self.player_id, role, 'swap_role', target_id=best_target)
            else:
                # If no good target found, randomly select
                target_id = self.get_random_player_except_self()
                action = create_night_action(self.player_id, role, 'swap_role', target_id=target_id)
        
        elif role == 'troublemaker':
            # Try to swap two players, prioritizing those likely to be werewolves
            players = []
            for player_id in range(len(self.game_state.players)):
                if player_id == self.player_id:
                    continue
                
                werewolf_prob = self.get_role_probabilities(player_id).get('werewolf', 0.5)
                players.append((player_id, werewolf_prob))
            
            # Sort by werewolf probability
            players.sort(key=lambda x: x[1], reverse=True)
            
            if len(players) >= 2:
                target_id1 = players[0][0]  # Most likely werewolf
                target_id2 = players[-1][0]  # Least likely werewolf
                
                action = create_night_action(self.player_id, role, 'swap_roles', 
                                          target_id1=target_id1, target_id2=target_id2)
            else:
                action = create_no_action(self.player_id)
        
        elif role == 'minion':
            # Minion can only check werewolves
            action = create_night_action(self.player_id, role, 'check_werewolves')
        
        elif role == 'insomniac':
            # Insomniac can only check their final role
            action = create_night_action(self.player_id, role, 'check_final_role')
        
        # For roles without night actions or other cases
        else:
            action = create_no_action(self.player_id)
        
        self.log_action(action)
        return action
    
    def _day_action(self, observation: Dict[str, Any]) -> Action:
        """Rule-based day speech"""
        role = self.current_role
        
        # Different roles have different speech strategies
        if role == 'werewolf':
            # Werewolves might pretend to be villagers or special roles
            if random.random() < 0.6:  # 60% chance to pretend to be villager
                action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role='villager')
            else:
                # 40% chance to pretend to be special role (usually won't claim to be seer as it's easily refuted)
                special_roles = ['robber', 'troublemaker', 'insomniac']
                fake_role = random.choice(special_roles)
                
                if fake_role == 'robber':
                    # Make up a stealing story
                    target_id = self.get_random_player_except_self()
                    fake_stolen_role = random.choice(['villager', 'troublemaker', 'insomniac'])
                    
                    action = create_speech(self.player_id, SpeechType.CLAIM_ACTION_RESULT.name,
                                        role='robber', action='steal', target=f"player{target_id}",
                                        result=fake_stolen_role)
                else:
                    # Simply claim role
                    action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role=fake_role)
        
        elif role == 'seer':
            # Seer usually claims to be seer and shares check results
            # Check for night action results
            action_results = [action for action in self.action_history 
                             if action.get('player_id') == self.player_id 
                             and action.get('action') == 'check_player']
            
            if action_results:
                # Has check results, share them
                action_result = action_results[-1]  # Latest check result
                target_id = action_result.get('target')
                result = action_result.get('result')
                
                if target_id is not None and result:
                    action = create_speech(self.player_id, SpeechType.CLAIM_ACTION_RESULT.name,
                                        role='seer', action='check', target=f"player{target_id}",
                                        result=result)
            
            # If no check results or checked center cards, simply claim to be seer
            else:
                action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role='seer')
        
        elif role in ['robber', 'troublemaker', 'insomniac']:
            # Special roles usually claim their role and action results
            action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role=role)
        
        elif role == 'minion':
            # Minion needs to protect werewolves, usually pretends to be villager
            action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role='villager')
        
        else:  # Villager
            # Villagers usually claim to be villagers or accuse suspicious players
            if random.random() < 0.7:  # 70% chance to claim to be villager
                action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role='villager')
            else:
                # 30% chance to accuse suspicious players
                suspected_player, prob = self.get_most_suspected_werewolf()
                if suspected_player >= 0 and prob > 0.3:
                    action = create_speech(self.player_id, SpeechType.ACCUSE.name,
                                        target_id=suspected_player, accused_role='werewolf')
                else:
                    # If no sufficiently suspicious players, claim to be villager
                    action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role='villager')
        
        self.log_action(action)
        return action
    
    def _vote_action(self, observation: Dict[str, Any]) -> Action:
        """
        Choose voting target
        
        Args:
            observation: Observation information
            
        Returns:
            Voting action
        """
        # Check if it's voting phase
        if observation.get('phase') != 'vote':
            return create_no_action(self.player_id)
            
        # Check if player has already voted
        if self.player_id in observation.get('votes', {}):
            return create_no_action(self.player_id)
            
        # Choose target based on role and belief state
        target_id = self._choose_vote_target(observation)
        
        # Create vote action
        return create_vote(self.player_id, target_id)
    
    def _choose_vote_target(self, observation: Dict[str, Any]) -> int:
        """
        Intelligently choose a voting target
        
        Based on role and known information, use different voting strategies:
        - Werewolf/Minion: Try to mislead voting, vote for villagers
        - Villager/Special roles: Try to identify and vote for werewolves
        
        Args:
            observation: Observation information
            
        Returns:
            Target player ID
        """
        # Current role
        role = self.current_role
        
        # Get available players to vote for (excluding self and already-voted players)
        available_players = [i for i in range(len(self.game_state.players)) 
                           if i != self.player_id and i not in observation.get('votes', {}).values()]
        if not available_players:
            return -1
            
        # Execute different voting strategies based on role
        if role == 'werewolf' or role == 'minion':
            # Werewolf team strategy: Avoid voting for werewolf teammates, prioritize villagers who are already suspicious
            
            # Find all werewolves
            werewolves = []
            for i, player in enumerate(self.game_state.players):
                if i != self.player_id and player['original_role'] in ['werewolf', 'minion']:
                    werewolves.append(i)
            
            # Exclude werewolf teammates
            safe_targets = [p for p in available_players if p not in werewolves]
            if not safe_targets:
                return random.choice(available_players)
            
            # Find which villager has been accused the most (prioritize already suspicious players)
            accusations = defaultdict(int)
            for speech in observation.get('speech_history', []):
                content = speech.get('content', {})
                if content.get('type') == 'ACCUSE':
                    target = content.get('target_id')
                    if target is not None and target in safe_targets:
                        accusations[target] += 1
            
            # Modify: Reduce effectiveness by adding random chance
            if random.random() < 0.4:  # 40% chance to make a suboptimal choice
                return random.choice(safe_targets)
                
            # If there are accused villagers, choose the most accused one
            if accusations:
                max_accusations = max(accusations.values())
                most_accused = [p for p, count in accusations.items() if count == max_accusations]
                return random.choice(most_accused)
            
            # Otherwise randomly choose a villager
            return random.choice(safe_targets)
        
        else:
            # Villager team strategy: Try to identify and vote for werewolves
            
            # Use belief system to track werewolf suspicion levels
            werewolf_suspects = {}
            
            # Initialize suspicion value for each player
            for player_id in range(len(self.game_state.players)):
                if player_id != self.player_id and player_id in available_players:
                    # Start with belief updater's probability if available
                    if self.belief_updater:
                        werewolf_prob = self.belief_updater.belief_state.beliefs[player_id].get('werewolf', 0.0)
                        werewolf_suspects[player_id] = werewolf_prob * 15  # Increase weight for villager team (was 10)
                    else:
                        werewolf_suspects[player_id] = 0.0
            
            # 1. Seer-specific logic - use belief_updater information
            if role == 'seer':
                # Seer may have already checked someone as werewolf
                most_suspected, prob = self.get_most_suspected_werewolf()
                if most_suspected >= 0 and prob > 0.4:  # Reduced threshold to make it more sensitive (was 0.5)
                    return most_suspected
            
            # 2. Analyze voting patterns
            vote_history = observation.get('votes', {})
            for voter_id, target_id in vote_history.items():
                # If someone voted for me, they are more suspicious
                if target_id == self.player_id and voter_id in available_players:
                    werewolf_suspects[voter_id] += 4.0  # Increased suspicion value (was 3.0)
                
                # Analyze coordinated voting patterns
                vote_counts = defaultdict(int)
                for vote_target in vote_history.values():
                    vote_counts[vote_target] += 1
                
                # If multiple players are voting for the same target (and it's not a werewolf),
                # they might be coordinating as werewolves
                if vote_counts[target_id] >= 2 and self.belief_updater:
                    target_werewolf_prob = self.belief_updater.belief_state.beliefs[target_id].get('werewolf', 0.5)
                    if target_werewolf_prob < 0.3 and voter_id in available_players:
                        werewolf_suspects[voter_id] += 2.0  # Increased suspicion value (was 1.0)
            
            # 3. Analyze speech behavior
            claimed_roles = {}
            contradictions = defaultdict(int)
            
            for speech in observation.get('speech_history', []):
                speaker_id = speech.get('player_id')
                content = speech.get('content', {})
                
                # Track claimed roles
                if content.get('type') == 'CLAIM_ROLE' and 'role' in content:
                    claimed_role = content.get('role')
                    if speaker_id not in claimed_roles:
                        claimed_roles[speaker_id] = claimed_role
                    elif claimed_roles[speaker_id] != claimed_role:
                        # Contradicting claims increase suspicion
                        contradictions[speaker_id] += 1
                        if speaker_id in available_players:
                            werewolf_suspects[speaker_id] += 5.0  # Increased suspicion (was 4.0)
                    
                    # Suspicious behavior: Claiming to be a special role when I am that role
                    if speaker_id in available_players:
                        if claimed_role == role and role in ['seer', 'robber', 'troublemaker', 'insomniac']:
                            werewolf_suspects[speaker_id] += 7.0  # Increased suspicion (was 5.0)
                        
                        # Werewolves rarely claim to be werewolves
                        if claimed_role == 'werewolf':
                            werewolf_suspects[speaker_id] -= 3.0
                
                # Analyze accusations
                if content.get('type') == 'ACCUSE':
                    accuser = speaker_id
                    accused = content.get('target_id')
                    
                    # If someone accuses me, they are more suspicious
                    if accused == self.player_id and accuser in available_players:
                        werewolf_suspects[accuser] += 3.0  # Increased suspicion (was 2.0)
                    
                    # Check for targeted accusations against special roles
                    if accused in claimed_roles and claimed_roles[accused] in ['seer', 'robber', 'troublemaker']:
                        # Werewolves often target special roles
                        if accuser in available_players:
                            werewolf_suspects[accuser] += 1.0  # Increased suspicion (was 0.5)
            
            # 4. Evaluate consistencies in action claims
            for speaker_id, claimed_role in claimed_roles.items():
                if speaker_id in available_players and self.belief_updater:
                    # Compare claimed role with belief state
                    role_prob = self.belief_updater.belief_state.beliefs[speaker_id].get(claimed_role, 0.0)
                    if role_prob < 0.2:  # Low probability of having the claimed role
                        werewolf_suspects[speaker_id] += 3.0  # Increased suspicion (was 2.0)
            
            # If suspicious targets exist, choose the most suspicious one
            if werewolf_suspects:
                max_suspect_value = max(werewolf_suspects.values())
                # Only when suspicion is high enough
                if max_suspect_value > 0:
                    strongest_suspects = [p for p, v in werewolf_suspects.items() 
                                          if v == max_suspect_value]
                    return random.choice(strongest_suspects)
            
            # If no clear suspicious target, try to vote with the majority (safety in numbers)
            vote_counts = defaultdict(int)
            for target_id in observation.get('votes', {}).values():
                if target_id in available_players:
                    vote_counts[target_id] += 1
            
            if vote_counts:
                max_votes = max(vote_counts.values())
                if max_votes > 0:  # Vote with majority even if only one vote (was 1)
                    most_voted = [p for p, count in vote_counts.items() if count == max_votes]
                    return random.choice(most_voted)
            
            # If still no clear target, make a random choice
            return random.choice(available_players)


# Factory function to create specified type of agent
def create_agent(agent_type: str, player_id: int) -> BaseAgent:
    """
    Create specified type of agent
    
    Args:
        agent_type: Agent type
        player_id: Player ID
        
    Returns:
        Agent instance
    """
    if agent_type == 'random':
        return RandomAgent(player_id)
    elif agent_type == 'heuristic':
        return HeuristicAgent(player_id)
    else:
        # Default return random agent
        return RandomAgent(player_id) 