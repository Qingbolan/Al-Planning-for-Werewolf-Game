"""
Werewolf Game Belief State Updater
Used to track and update players' beliefs about other players' roles
"""
from typing import Dict, List, Any, Set, Tuple, Optional
import numpy as np
from collections import defaultdict
import copy

from werewolf_env.state import GameState
from werewolf_env.actions import SpeechType


class BeliefState:
    """Belief state class, representing beliefs about role distribution in the game"""
    
    def __init__(self, game_state, player_id: int):
        """
        Initialize belief state
        
        Args:
            game_state: Game state (dict or GameState object)
            player_id: Player ID
        """
        self.player_id = player_id
        
        # 处理字典格式的game_state
        if isinstance(game_state, dict):
            self.num_players = len(game_state.get('players', []))
            # 从action_order或center_cards中获取可能的角色列表
            self.possible_roles = list(set(game_state.get('action_order', [])))
            if not self.possible_roles:
                # 如果action_order为空，使用默认角色列表
                self.possible_roles = ['werewolf', 'villager', 'seer', 'robber', 'troublemaker', 'insomniac', 'minion', 'mason', 'drunk', 'hunter', 'tanner']
            
            # 获取当前玩家原始角色
            self.original_role = "unknown"
            players = game_state.get('players', [])
            for p in players:
                if p.get('player_id') == player_id:
                    self.original_role = p.get('original_role', 'unknown')
                    break
                    
            # 获取中央牌数量
            self.num_center_cards = len(game_state.get('center_cards', []))
        else:
            # 处理GameState对象
            self.num_players = game_state.num_players
            self.possible_roles = list(set(game_state.roles))
            self.original_role = game_state.players[player_id]['original_role']
            self.num_center_cards = game_state.num_center_cards
        
        # Initialize role probability distribution for each player (uniform distribution)
        self.beliefs = {}
        for p_id in range(self.num_players):
            if p_id == player_id:
                # Certain belief about own role
                self.beliefs[p_id] = {role: 1.0 if role == self.original_role else 0.0 
                                     for role in self.possible_roles}
            else:
                # Uniform belief distribution for other players' roles
                self.beliefs[p_id] = {role: 1.0 / len(self.possible_roles) 
                                     for role in self.possible_roles}
        
        # Certain roles (known through night actions or other certain means)
        self.certain_roles = {player_id: self.original_role}
        
        # Possible center card roles
        self.center_card_beliefs = {}
        for i in range(self.num_center_cards):
            self.center_card_beliefs[i] = {role: 1.0 / len(self.possible_roles) 
                                         for role in self.possible_roles}
        
        # Known information
        self.known_info = []
        
        # Record claimed roles
        self.claimed_roles = {}
        
        # Record special role action claims
        self.claimed_actions = defaultdict(list)
        
        # Record vote history
        self.vote_history = {}
    
    def get_most_likely_role(self, player_id: int) -> Tuple[str, float]:
        """
        Get the most likely role and its probability for a player
        
        Args:
            player_id: Player ID
            
        Returns:
            Tuple[str, float]: (Most likely role, probability)
        """
        if player_id not in self.beliefs:
            return None, 0.0
            
        # If player has a certain role
        if player_id in self.certain_roles:
            return self.certain_roles[player_id], 1.0
            
        # Otherwise return the role with highest probability
        beliefs = self.beliefs[player_id]
        max_role = max(beliefs.items(), key=lambda x: x[1])
        return max_role[0], max_role[1]
    
    def normalize_beliefs(self) -> None:
        """Normalize all belief probabilities to sum to 1"""
        for player_id in self.beliefs:
            total = sum(self.beliefs[player_id].values())
            if total > 0:
                for role in self.beliefs[player_id]:
                    self.beliefs[player_id][role] /= total
        
        # Normalize center card beliefs
        for card_idx in self.center_card_beliefs:
            total = sum(self.center_card_beliefs[card_idx].values())
            if total > 0:
                for role in self.center_card_beliefs[card_idx]:
                    self.center_card_beliefs[card_idx][role] /= total
    
    def update_with_certain_role(self, player_id: int, role: str) -> None:
        """
        Update beliefs with certain role information
        
        Args:
            player_id: Player ID
            role: Certain role
        """
        if player_id < 0 or player_id >= self.num_players:
            return
            
        # Update certain roles dictionary
        self.certain_roles[player_id] = role
        
        # Update belief distribution for this player
        for r in self.beliefs[player_id]:
            self.beliefs[player_id][r] = 1.0 if r == role else 0.0
        
        # Update known information
        self.known_info.append({
            'type': 'certain_role',
            'player_id': player_id,
            'role': role
        })
        
        # Update other players' beliefs (same role cannot be held by multiple players)
        for p_id in self.beliefs:
            if p_id != player_id:
                if role in self.beliefs[p_id]:
                    # Reduce probability of this player having this role (not completely exclude to handle uncertainty)
                    self.beliefs[p_id][role] *= 0.1
        
        # Normalize beliefs
        self.normalize_beliefs()
    
    def update_with_center_card(self, card_idx: int, role: str) -> None:
        """
        Update center card beliefs
        
        Args:
            card_idx: Card index
            role: Certain role
        """
        if card_idx not in self.center_card_beliefs:
            return
            
        # Update card beliefs
        for r in self.center_card_beliefs[card_idx]:
            self.center_card_beliefs[card_idx][r] = 1.0 if r == role else 0.0
        
        # Update known information
        self.known_info.append({
            'type': 'center_card',
            'card_idx': card_idx,
            'role': role
        })
        
        # Update player beliefs (role in center cards, unlikely to be held by players)
        for p_id in self.beliefs:
            if role in self.beliefs[p_id]:
                # Reduce probability of player having this role
                self.beliefs[p_id][role] *= 0.5
        
        # Normalize beliefs
        self.normalize_beliefs()
    
    def update_with_role_swap(self, player_id1: int, player_id2: int) -> None:
        """
        Update beliefs after role swap
        
        Args:
            player_id1: First player ID
            player_id2: Second player ID
        """
        if player_id1 < 0 or player_id1 >= self.num_players or player_id2 < 0 or player_id2 >= self.num_players:
            return
            
        # Swap beliefs
        self.beliefs[player_id1], self.beliefs[player_id2] = self.beliefs[player_id2], self.beliefs[player_id1]
        
        # Update certain roles (if any)
        if player_id1 in self.certain_roles and player_id2 in self.certain_roles:
            self.certain_roles[player_id1], self.certain_roles[player_id2] = self.certain_roles[player_id2], self.certain_roles[player_id1]
        elif player_id1 in self.certain_roles:
            self.certain_roles[player_id2] = self.certain_roles.pop(player_id1)
        elif player_id2 in self.certain_roles:
            self.certain_roles[player_id1] = self.certain_roles.pop(player_id2)
        
        # Update known information
        self.known_info.append({
            'type': 'role_swap',
            'player_id1': player_id1,
            'player_id2': player_id2
        })


class RoleSpecificBeliefUpdater:
    """Base class for role-specific belief updaters"""
    
    def __init__(self, player_id: int, game_state):
        """
        Initialize belief updater
        
        Args:
            player_id: Player ID
            game_state: Game state (dict or GameState object)
        """
        self.player_id = player_id
        self.game_state = game_state
        self.belief_state = BeliefState(game_state, player_id)
        
        # 处理字典格式的game_state
        if isinstance(game_state, dict):
            players = game_state.get('players', [])
            self.role = "unknown"
            # 查找当前玩家信息
            for p in players:
                if p.get('player_id') == player_id:
                    self.role = p.get('original_role', 'unknown')
                    break
        else:
            # 处理GameState对象
            self.role = game_state.players[player_id]['original_role']
    
    def update_with_night_action(self, action_result: Dict[str, Any]) -> None:
        """
        Update beliefs based on night action results
        
        Args:
            action_result: Action result
        """
        # Base class method, can be overridden by subclasses
        pass
    
    def update_with_speech(self, speaker_id: int, speech_content: Dict[str, Any]) -> None:
        """
        Update beliefs based on speech
        
        Args:
            speaker_id: Speaker ID
            speech_content: Speech content
        """
        # Record player's claimed role
        if speech_content.get('type') == SpeechType.CLAIM_ROLE.name and 'role' in speech_content:
            self.belief_state.claimed_roles[speaker_id] = speech_content['role']
            
            # Update beliefs (slightly increase probability of this role)
            role = speech_content['role']
            if role in self.belief_state.beliefs[speaker_id]:
                # Increase probability of this role
                self.belief_state.beliefs[speaker_id][role] *= 1.2
                
                # If player claims to be werewolf (unlikely), reduce belief
                if role == 'werewolf':
                    self.belief_state.beliefs[speaker_id][role] *= 0.5
        
        # Handle action result claims
        elif speech_content.get('type') == SpeechType.CLAIM_ACTION_RESULT.name:
            if 'role' in speech_content and 'action' in speech_content and 'target' in speech_content and 'result' in speech_content:
                self.belief_state.claimed_actions[speaker_id].append({
                    'role': speech_content['role'],
                    'action': speech_content['action'],
                    'target': speech_content['target'],
                    'result': speech_content['result']
                })
                
                # If claiming check result, update beliefs
                if speech_content['role'] == 'seer' and speech_content['action'] in ['check', 'view']:
                    # Try to parse target player ID
                    target_id = -1
                    target = speech_content['target']
                    if isinstance(target, int):
                        target_id = target
                    elif isinstance(target, str) and target.startswith('Player') and target[6:].isdigit():
                        target_id = int(target[6:])
                    
                    if 0 <= target_id < self.game_state.num_players:
                        claimed_role = speech_content['result']
                        
                        # Judge credibility of this claim based on own role and information
                        if self.role == 'seer':
                            # If self is seer, know if this claim is true
                            # Simplified handling, actually need to judge based on own check results
                            credibility = 0.1  # Assume low credibility
                        else:
                            # If not seer, judge credibility based on other information
                            credibility = 0.5  # Medium credibility
                        
                        # Update target player's role beliefs
                        if claimed_role in self.belief_state.beliefs[target_id]:
                            # Adjust probability based on credibility
                            self.belief_state.beliefs[target_id][claimed_role] *= (1.0 + credibility)
        
        # Handle accusations
        elif speech_content.get('type') == SpeechType.ACCUSE.name:
            if 'target_id' in speech_content and 'accused_role' in speech_content:
                target_id = speech_content['target_id']
                accused_role = speech_content['accused_role']
                
                if 0 <= target_id < self.game_state.num_players and accused_role in self.belief_state.beliefs[target_id]:
                    # Increase probability of player being accused role
                    self.belief_state.beliefs[target_id][accused_role] *= 1.1
        
        # Normalize beliefs
        self.belief_state.normalize_beliefs()
    
    def update_with_votes(self, votes: Dict[int, int]) -> None:
        """
        Update beliefs based on votes
        
        Args:
            votes: Vote results, key is voter ID, value is target ID
        """
        # Record vote history
        self.belief_state.vote_history.update(votes)
        
        # Analyze voting patterns
        vote_counts = defaultdict(int)
        for target_id in votes.values():
            vote_counts[target_id] += 1
        
        # Who players vote for may reflect their alignment
        for voter_id, target_id in votes.items():
            if voter_id == self.player_id:
                continue  # Skip own votes
                
            # Adjust beliefs based on vote target
            # If voting for possible werewolf, increase probability of voter being good
            werewolf_prob = self.belief_state.beliefs[target_id].get('werewolf', 0.0)
            if werewolf_prob > 0.5:
                # Increase probability of voter being villager alignment
                for role in ['villager', 'seer', 'robber', 'troublemaker', 'insomniac']:
                    if role in self.belief_state.beliefs[voter_id]:
                        self.belief_state.beliefs[voter_id][role] *= 1.1
                        
                # Decrease probability of voter being werewolf alignment
                for role in ['werewolf', 'minion']:
                    if role in self.belief_state.beliefs[voter_id]:
                        self.belief_state.beliefs[voter_id][role] *= 0.9
            
            # If voting for possible seer, increase probability of voter being werewolf
            seer_prob = self.belief_state.beliefs[target_id].get('seer', 0.0)
            if seer_prob > 0.5:
                # Increase probability of voter being werewolf alignment
                for role in ['werewolf', 'minion']:
                    if role in self.belief_state.beliefs[voter_id]:
                        self.belief_state.beliefs[voter_id][role] *= 1.1
                        
                # Decrease probability of voter being villager alignment
                for role in ['villager', 'seer', 'robber', 'troublemaker', 'insomniac']:
                    if role in self.belief_state.beliefs[voter_id]:
                        self.belief_state.beliefs[voter_id][role] *= 0.9
        
        # Normalize beliefs
        self.belief_state.normalize_beliefs()
    
    def get_action_probabilities(self, action_space: List[Any]) -> Dict[Any, float]:
        """
        Get action probability distribution
        
        Args:
            action_space: Available action space
            
        Returns:
            Action probability distribution
        """
        # Base class method, can be overridden by subclasses
        return {action: 1.0 / len(action_space) for action in action_space}


class VillagerBeliefUpdater(RoleSpecificBeliefUpdater):
    """Villager belief updater"""
    
    def update_with_night_action(self, action_result: Dict[str, Any]) -> None:
        """Villager has no night actions"""
        pass


class WerewolfBeliefUpdater(RoleSpecificBeliefUpdater):
    """Werewolf belief updater"""
    
    def update_with_night_action(self, action_result: Dict[str, Any]) -> None:
        """Update beliefs based on werewolf night action results"""
        action = action_result.get('action', '')
        
        if action == 'check_other_werewolves':
            # Update other werewolves' information
            werewolves = action_result.get('result', [])
            for werewolf_id in werewolves:
                if 0 <= werewolf_id < self.game_state.num_players:
                    self.belief_state.update_with_certain_role(werewolf_id, 'werewolf')
        
        elif action == 'check_center_card':
            # Update center card information
            card_index = action_result.get('card_index', -1)
            role = action_result.get('result', '')
            if card_index >= 0 and role:
                self.belief_state.update_with_center_card(card_index, role)


class SeerBeliefUpdater(RoleSpecificBeliefUpdater):
    """Seer belief updater"""
    
    def update_with_night_action(self, action_result: Dict[str, Any]) -> None:
        """Update beliefs based on seer night action results"""
        action = action_result.get('action', '')
        
        if action == 'check_player':
            # Update player role information
            target_id = action_result.get('target', -1)
            result = action_result.get('result', '')
            if target_id >= 0 and result:
                self.belief_state.update_with_certain_role(target_id, result)
        
        elif action == 'check_center_cards':
            # Update center card information
            targets = action_result.get('targets', [])
            results = action_result.get('result', [])
            for i, card_idx in enumerate(targets):
                if i < len(results):
                    self.belief_state.update_with_center_card(card_idx, results[i])


class RobberBeliefUpdater(RoleSpecificBeliefUpdater):
    """Robber belief updater"""
    
    def update_with_night_action(self, action_result: Dict[str, Any]) -> None:
        """Update beliefs based on robber night action results"""
        action = action_result.get('action', '')
        
        if action == 'swap_role':
            # Update role swap information
            target_id = action_result.get('target', -1)
            result = action_result.get('result', '')
            if target_id >= 0 and result:
                # Know target player's original role
                self.belief_state.update_with_certain_role(target_id, result)
                
                # Own role becomes target player's role
                self.belief_state.update_with_certain_role(self.player_id, result)
                
                # Target player's role becomes robber
                self.belief_state.update_with_certain_role(target_id, 'robber')


class TroublemakerBeliefUpdater(RoleSpecificBeliefUpdater):
    """Troublemaker belief updater"""
    
    def update_with_night_action(self, action_result: Dict[str, Any]) -> None:
        """Update beliefs based on troublemaker night action results"""
        action = action_result.get('action', '')
        
        if action == 'swap_roles':
            # Update role swap information
            targets = action_result.get('targets', [])
            result = action_result.get('result', False)
            
            if len(targets) == 2 and result:
                target_id1, target_id2 = targets
                
                # Know two players swapped roles, but don't know their specific roles
                # Can only update belief state to reflect this
                self.belief_state.update_with_role_swap(target_id1, target_id2)


class MinionBeliefUpdater(RoleSpecificBeliefUpdater):
    """Minion belief updater"""
    
    def update_with_night_action(self, action_result: Dict[str, Any]) -> None:
        """Update beliefs based on minion night action results"""
        action = action_result.get('action', '')
        
        if action == 'check_werewolves':
            # Update werewolf information
            werewolves = action_result.get('result', [])
            for werewolf_id in werewolves:
                if 0 <= werewolf_id < self.game_state.num_players:
                    self.belief_state.update_with_certain_role(werewolf_id, 'werewolf')


class InsomniacBeliefUpdater(RoleSpecificBeliefUpdater):
    """Insomniac belief updater"""
    
    def update_with_night_action(self, action_result: Dict[str, Any]) -> None:
        """Update beliefs based on insomniac night action results"""
        action = action_result.get('action', '')
        
        if action == 'check_final_role':
            # Update own final role
            result = action_result.get('result', '')
            if result:
                self.belief_state.update_with_certain_role(self.player_id, result)


class EnhancedBeliefUpdater(RoleSpecificBeliefUpdater):
    """Enhanced belief updater with improved inference capabilities"""
    
    def __init__(self, player_id: int, game_state):
        """Initialize the enhanced belief updater"""
        super().__init__(player_id, game_state)
        
        # Track speech consistency and behavior patterns
        self.player_speech_history = defaultdict(list)
        self.player_vote_history = defaultdict(list)
        self.player_consistency_score = defaultdict(float)
        self.claimed_roles = {}
        
        # Track voting patterns
        self.vote_patterns = defaultdict(list)
        
        # Track known information based on player's own observations
        self.confirmed_information = {}
        
        # 处理字典格式的game_state
        if isinstance(game_state, dict):
            num_players = len(game_state.get('players', []))
        else:
            num_players = game_state.num_players
            
        # Initialize consistency scores (higher = more consistent/trustworthy)
        for p_id in range(num_players):
            if p_id != player_id:
                self.player_consistency_score[p_id] = 0.5  # Neutral starting point
    
    def update_with_speech(self, speaker_id: int, speech_content: Dict[str, Any]) -> None:
        """
        Enhanced update based on speech with pattern analysis
        
        Args:
            speaker_id: Speaker ID
            speech_content: Speech content
        """
        # Call the base update method
        super().update_with_speech(speaker_id, speech_content)
        
        # Track this speech in history
        self.player_speech_history[speaker_id].append(speech_content)
        
        # Handle role claims
        if speech_content.get('type') == SpeechType.CLAIM_ROLE.name and 'role' in speech_content:
            claimed_role = speech_content['role']
            
            # Check for contradicting claims
            if speaker_id in self.claimed_roles and self.claimed_roles[speaker_id] != claimed_role:
                # Contradiction detected - decrease consistency score
                self.player_consistency_score[speaker_id] -= 0.2
                
                # Increase probability of being werewolf
                if 'werewolf' in self.belief_state.beliefs[speaker_id]:
                    self.belief_state.beliefs[speaker_id]['werewolf'] *= 1.5
            
            # Record new claim
            self.claimed_roles[speaker_id] = claimed_role
            
            # Special cases based on role claim
            if claimed_role == 'werewolf':
                # Claiming to be werewolf is unusual - likely village team trying to be funny or confuse
                self.belief_state.beliefs[speaker_id]['werewolf'] *= 0.3
                # Increase probability of being villager
                for role in ['villager', 'seer', 'robber', 'troublemaker', 'insomniac']:
                    if role in self.belief_state.beliefs[speaker_id]:
                        self.belief_state.beliefs[speaker_id][role] *= 1.2
            
            # If claimed role is the same as the player's actual role
            if claimed_role == self.role and claimed_role in ['seer', 'robber', 'troublemaker', 'insomniac']:
                # Increase probability of being werewolf (they're likely lying)
                if 'werewolf' in self.belief_state.beliefs[speaker_id]:
                    self.belief_state.beliefs[speaker_id]['werewolf'] *= 1.8
        
        # Handle action result claims
        elif speech_content.get('type') == SpeechType.CLAIM_ACTION_RESULT.name:
            # Check if the claim is consistent with our knowledge
            if 'role' in speech_content and 'action' in speech_content:
                claimed_role = speech_content.get('role')
                
                # Check if this role claim is consistent with previous claims
                if speaker_id in self.claimed_roles:
                    if self.claimed_roles[speaker_id] != claimed_role:
                        # Inconsistent role claims
                        self.player_consistency_score[speaker_id] -= 0.2
                        # Increase probability of being werewolf
                        if 'werewolf' in self.belief_state.beliefs[speaker_id]:
                            self.belief_state.beliefs[speaker_id]['werewolf'] *= 1.5
                
                # Update claimed roles
                self.claimed_roles[speaker_id] = claimed_role
                
                # Check if action result is consistent with our knowledge
                if self.player_id == speaker_id:
                    # We know our own actions - no need to assess
                    pass
                elif claimed_role in ['seer', 'robber', 'troublemaker'] and 'target' in speech_content:
                    # These roles could provide information we can verify
                    if claimed_role == 'seer' and self.role == 'seer':
                        # They're claiming to be seer, but we are the seer - they're lying
                        self.player_consistency_score[speaker_id] -= 0.3
                        # Increase probability of being werewolf significantly
                        if 'werewolf' in self.belief_state.beliefs[speaker_id]:
                            self.belief_state.beliefs[speaker_id]['werewolf'] *= 2.0
        
        # Handle accusations
        elif speech_content.get('type') == SpeechType.ACCUSE.name:
            if 'target_id' in speech_content:
                target_id = speech_content['target_id']
                
                # If player accuses someone we know is on their team
                if target_id in self.confirmed_information and speaker_id in self.confirmed_information:
                    if self.confirmed_information[target_id] == self.confirmed_information[speaker_id]:
                        # They're accusing someone on their own team - suspicious
                        self.player_consistency_score[speaker_id] -= 0.1
                
                # If player accuses us, consider them more likely to be werewolf
                if target_id == self.player_id:
                    # Increase probability of being werewolf
                    if 'werewolf' in self.belief_state.beliefs[speaker_id]:
                        self.belief_state.beliefs[speaker_id]['werewolf'] *= 1.3
        
        # Normalize beliefs
        self.belief_state.normalize_beliefs()
    
    def update_with_votes(self, votes: Dict[int, int]) -> None:
        """
        Enhanced update with votes with pattern detection
        
        Args:
            votes: Vote results, key is voter ID, value is target ID
        """
        # Call the base update method
        super().update_with_votes(votes)
        
        # Track all votes
        for voter_id, target_id in votes.items():
            self.player_vote_history[voter_id].append(target_id)
            
            # Add to vote patterns
            self.vote_patterns[target_id].append(voter_id)
        
        # Analyze voting patterns for groups
        voting_groups = self._identify_voting_groups(votes)
        
        # If we find suspicious voting patterns (e.g., all werewolves voting together)
        for group in voting_groups:
            # Skip single voters
            if len(group) <= 1:
                continue
                
            # Check if any player in this group is confirmed as werewolf
            known_werewolf_in_group = False
            for voter_id in group:
                if voter_id in self.belief_state.certain_roles and self.belief_state.certain_roles[voter_id] == 'werewolf':
                    known_werewolf_in_group = True
                    break
            
            if known_werewolf_in_group:
                # Increase werewolf probability for everyone in the group
                for voter_id in group:
                    if 'werewolf' in self.belief_state.beliefs[voter_id]:
                        self.belief_state.beliefs[voter_id]['werewolf'] *= 1.4
        
        # Check for players who vote against the majority
        vote_counts = defaultdict(int)
        for target_id in votes.values():
            vote_counts[target_id] += 1
        
        if vote_counts:
            # Find the majority vote target
            max_votes = max(vote_counts.values())
            majority_targets = [target for target, count in vote_counts.items() if count == max_votes]
            
            # Check players voting against a clear majority
            if len(majority_targets) == 1 and max_votes > 2:
                majority_target = majority_targets[0]
                
                # Players who vote differently might be protecting werewolves
                for voter_id, target_id in votes.items():
                    if target_id != majority_target:
                        # Make minor adjustment to consistency score
                        self.player_consistency_score[voter_id] -= 0.05
        
        # Normalize beliefs
        self.belief_state.normalize_beliefs()
    
    def _identify_voting_groups(self, votes: Dict[int, int]) -> List[List[int]]:
        """
        Identify groups of players voting for the same target
        
        Args:
            votes: Vote dict
            
        Returns:
            List of voter groups
        """
        target_to_voters = defaultdict(list)
        
        # Group voters by their targets
        for voter_id, target_id in votes.items():
            target_to_voters[target_id].append(voter_id)
        
        # Return groups with more than one voter
        return [voters for target, voters in target_to_voters.items() if len(voters) > 1]
    
    def get_action_probabilities(self, action_space: List[Any]) -> Dict[Any, float]:
        """
        Get enhanced action probability distribution
        
        Args:
            action_space: Available action space
            
        Returns:
            Action probability distribution
        """
        # Implement more sophisticated logic based on game state
        # For now, return uniform distribution
        return {action: 1.0 / len(action_space) for action in action_space}
    
    def update_with_night_action(self, action_result: Dict[str, Any]) -> None:
        """
        Update beliefs based on night action results
        
        Args:
            action_result: Action result
        """
        super().update_with_night_action(action_result)
        
        # Process confirmed information from night actions
        action_type = action_result.get('action', '')
        
        if self.role == 'seer' and action_type == 'check_player':
            # Seer checked a player and knows their role
            target_id = action_result.get('target_id')
            result = action_result.get('result')
            
            if target_id is not None and result:
                # Add to confirmed information
                self.confirmed_information[target_id] = result
                
                # Update belief state with certain role
                self.belief_state.update_with_certain_role(target_id, result)
        
        elif self.role == 'werewolf' and action_type == 'check_other_werewolves':
            # Werewolf knows other werewolves
            werewolves = action_result.get('result', [])
            for werewolf_id in werewolves:
                # Add to confirmed information
                self.confirmed_information[werewolf_id] = 'werewolf'
                
                # Update belief state with certain role
                self.belief_state.update_with_certain_role(werewolf_id, 'werewolf')


# Role belief updater mapping table
BELIEF_UPDATER_MAP = {
    'villager': VillagerBeliefUpdater,
    'werewolf': WerewolfBeliefUpdater,
    'minion': MinionBeliefUpdater,
    'seer': SeerBeliefUpdater,
    'robber': RobberBeliefUpdater,
    'troublemaker': TroublemakerBeliefUpdater,
    'insomniac': InsomniacBeliefUpdater
}


def create_belief_updater(player_id: int, game_state) -> RoleSpecificBeliefUpdater:
    """
    Create a belief updater for a player based on their role
    
    Args:
        player_id: Player ID
        game_state: Game state (dict or GameState object)
        
    Returns:
        RoleSpecificBeliefUpdater instance
    """
    # 处理字典格式的game_state
    if isinstance(game_state, dict):
        # 从字典中获取role
        players = game_state.get('players', [])
        role = 'unknown'
        
        # 查找当前玩家的角色
        for p in players:
            if p.get('player_id') == player_id:
                role = p.get('original_role', 'unknown')
                break
    else:
        # 原来的GameState对象处理
        role = game_state.players[player_id]['original_role'] if player_id < len(game_state.players) else 'unknown'
    
    # Create role-specific belief updater
    if role in BELIEF_UPDATER_MAP:
        # First create a role-specific belief updater to handle unique logic
        role_updater = BELIEF_UPDATER_MAP[role](player_id, game_state)
        
        # Then enhance it with the improved EnhancedBeliefUpdater
        enhanced_updater = EnhancedBeliefUpdater(player_id, game_state)
        
        # Copy the role-specific knowledge to the enhanced updater
        enhanced_updater.role = role
        if hasattr(role_updater, 'belief_state') and hasattr(role_updater.belief_state, 'certain_roles'):
            for player, certain_role in role_updater.belief_state.certain_roles.items():
                enhanced_updater.belief_state.update_with_certain_role(player, certain_role)
        
        # Add special role-based enhancements
        if role == 'villager' or role == 'seer' or role == 'robber' or role == 'troublemaker' or role == 'insomniac':
            # 处理字典格式的game_state
            if isinstance(game_state, dict):
                num_players = len(game_state.get('players', []))
            else:
                num_players = game_state.num_players
                
            # Villager team gets better at detecting werewolves (improved accuracy)
            for player_idx in range(num_players):
                if player_idx != enhanced_updater.player_id:
                    # Substantially reduce initial werewolf probability for village team members
                    if 'werewolf' in enhanced_updater.belief_state.beliefs[player_idx]:
                        enhanced_updater.belief_state.beliefs[player_idx]['werewolf'] *= 0.7
                    
                    # More aggressively detect inconsistent claims from potential werewolves
                    enhanced_updater.player_consistency_score[player_idx] = 0.7  # Start with higher trust baseline
        elif role == 'werewolf' or role == 'minion':
            # 处理字典格式的game_state
            if isinstance(game_state, dict):
                num_players = len(game_state.get('players', []))
            else:
                num_players = game_state.num_players
                
            # Slightly reduce werewolf team's ability to detect other werewolves (to balance gameplay)
            for player_idx in range(num_players):
                if player_idx != enhanced_updater.player_id:
                    # Adjust beliefs to make werewolf team slightly less effective
                    if 'villager' in enhanced_updater.belief_state.beliefs[player_idx]:
                        enhanced_updater.belief_state.beliefs[player_idx]['villager'] *= 0.9
        
        # Normalize beliefs
        enhanced_updater.belief_state.normalize_beliefs()
        return enhanced_updater
    else:
        # Fallback to basic belief updater
        return RoleSpecificBeliefUpdater(player_id, game_state)