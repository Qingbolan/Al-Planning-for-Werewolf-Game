"""
Werewolf Game State Representation
"""
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import copy
import random
from collections import defaultdict

from werewolf_env.roles import create_role
from utils.common import validate_state

# Define night action roles
NIGHT_ACTIONS = {
    'werewolf': ['check_other_werewolves', 'check_center_card'],
    'seer': ['check_player', 'check_center_cards'],
    'robber': ['swap_role'],
    'troublemaker': ['swap_roles'],
    'minion': ['check_werewolves'],
    'insomniac': ['check_final_role']
}

class GameState:
    """Game state class, maintains complete game state"""
    
    GAME_PHASES = ['init', 'night', 'day', 'vote', 'end']
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize game state
        
        Args:
            config: Game configuration dictionary
        """
        self.config = config
        self.num_players = config['num_players']
        self.num_center_cards = config['num_center_cards']
        
        # Game phase
        self.phase = 'init'
        self.round = 0
        
        # Role assignment
        self.roles = copy.deepcopy(config['roles'])
        
        # 分离关键角色（2个狼人和1个爪牙）和其他角色
        critical_roles = []
        other_roles = []
        
        # 先找出所有狼人和爪牙角色
        werewolf_count = 0
        minion_found = False
        
        for role in self.roles:
            if role == 'werewolf' and werewolf_count < 2:
                critical_roles.append(role)
                werewolf_count += 1
            elif role == 'minion' and not minion_found:
                critical_roles.append(role)
                minion_found = True
            else:
                other_roles.append(role)
        
        # 打乱非关键角色的顺序
        random.shuffle(other_roles)
        
        # 确保关键角色在玩家手中（而不是中央牌堆）
        player_roles = critical_roles + other_roles[:self.num_players - len(critical_roles)]
        center_roles = other_roles[self.num_players - len(critical_roles):]
        
        # 打乱玩家角色的顺序
        random.shuffle(player_roles)
        
        # 重新组合角色列表
        self.roles = player_roles + center_roles
        
        # Player states
        self.players = []
        for i in range(self.num_players):
            self.players.append({
                'id': i,
                'original_role': self.roles[i],
                'current_role': self.roles[i],
                'known_info': [],
                'belief_states': defaultdict(lambda: {role: 1/len(self.roles) for role in set(self.roles)})
            })
        
        # Center cards
        self.center_cards = self.roles[self.num_players:]
        
        # Role instances
        self.role_instances = {}
        for i in range(self.num_players):
            self.role_instances[i] = create_role(self.roles[i], i)
        
        # History records
        self.action_history = []
        self.speech_history = []
        
        # Voting results
        self.votes = {}
        self.has_voted = set()  # Track which players have voted
        
        # Game result
        self.game_result = None
        
        # Current round player
        self.current_player = 0
        
        # Current night action role index
        self.night_action_index = 0
        
        # Speech round counter
        self.speech_round = 0
        self.max_speech_rounds = config.get('max_speech_rounds', 3)  # 默认为3轮，确保一致性
        
        # Night action roles
        self.night_action_roles = self._get_night_action_roles()
        
        # Set initial phase to night
        self.phase = 'night'
        
    def _get_night_action_roles(self) -> List[int]:
        """Get list of roles with night actions"""
        night_roles = []
        
        # Determine night action order based on role order in config
        role_order = self.config.get('role_action_order', [
            'werewolf', 'minion', 'seer', 'robber', 'troublemaker', 'insomniac'
        ])
        
        for role in role_order:
            for i, player in enumerate(self.players):
                if player['original_role'] == role and role in NIGHT_ACTIONS:
                    night_roles.append(i)
        
        return night_roles
    
    def get_current_player(self) -> int:
        """Get current active player ID"""
        if self.phase == 'night':
            if self.night_action_index < len(self.night_action_roles):
                return self.night_action_roles[self.night_action_index]
            return -1
        elif self.phase in ['day', 'vote']:
            return self.current_player
        return -1
    
    def next_player(self) -> int:
        """Move to next player, return new current player ID"""
        if self.phase == 'night':
            self.night_action_index += 1
            if self.night_action_index >= len(self.night_action_roles):
                # Night phase ends
                self.phase = 'day'
                self.speech_round = 0  # Initialize speech round
                self.current_player = 0
                return self.current_player
            return self.night_action_roles[self.night_action_index]
        
        elif self.phase == 'day':
            self.current_player = (self.current_player + 1) % self.num_players
            # If all players have spoken in a round, start new round
            if self.current_player == 0:
                self.speech_round += 1
                # If all three rounds of speech are completed, enter voting phase
                if self.speech_round >= self.max_speech_rounds:
                    self.phase = 'vote'
                    self.current_player = 0  # Start voting from player 0
            return self.current_player
        
        elif self.phase == 'vote':
            self.current_player = (self.current_player + 1) % self.num_players
            # If all players have voted, end game
            if self.current_player == 0 and len(self.votes) == self.num_players:
                self.phase = 'end'
                self._determine_winner()
            return self.current_player
            
        return -1
    
    def perform_night_action(self, action_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute night action
        
        Args:
            action_params: Action parameters
            
        Returns:
            Action result
        """
        player_id = self.get_current_player()
        if player_id < 0 or self.phase != 'night':
            return {'success': False, 'message': 'Invalid action phase or player'}
        
        player_role = self.role_instances[player_id]
        result = player_role.night_action(self.to_dict(), action_params)
        
        # Record action history
        action_record = {
            'player_id': player_id,
            'role': player_role.original_role_name,
            'action': result['action'],
            'params': action_params,
            'result': result['result']
        }
        self.action_history.append(action_record)
        
        # Update game state (e.g., role swaps)
        self._update_state_after_action(player_id, result)
        
        # Move to next player
        self.next_player()
        
        return result
    
    def _update_state_after_action(self, player_id: int, action_result: Dict[str, Any]) -> None:
        """
        Update game state based on action result
        
        Args:
            player_id: Acting player ID
            action_result: Action result
        """
        action = action_result['action']
        
        # Update state based on different action types
        if action == 'swap_role':
            target_id = action_result.get('target')
            if target_id is not None and 0 <= target_id < self.num_players:
                # Swap roles
                self.players[player_id]['current_role'], self.players[target_id]['current_role'] = \
                    self.players[target_id]['current_role'], self.players[player_id]['current_role']
                
                # Update role instances
                self.role_instances[player_id] = create_role(self.players[player_id]['current_role'], player_id)
                self.role_instances[target_id] = create_role(self.players[target_id]['current_role'], target_id)
                
        elif action == 'swap_roles':
            targets = action_result.get('targets', [])
            if len(targets) == 2 and action_result.get('result') == True:
                target_id1, target_id2 = targets
                # Swap roles between two players
                self.players[target_id1]['current_role'], self.players[target_id2]['current_role'] = \
                    self.players[target_id2]['current_role'], self.players[target_id1]['current_role']
                
                # Update role instances
                self.role_instances[target_id1] = create_role(self.players[target_id1]['current_role'], target_id1)
                self.role_instances[target_id2] = create_role(self.players[target_id2]['current_role'], target_id2)
    
    def record_speech(self, player_id: int, speech_content: Dict[str, Any]) -> None:
        """
        Record player speech
        
        Args:
            player_id: Player ID
            speech_content: Speech content
        """
        speech_record = {
            'player_id': player_id,
            'round': self.speech_round,  # Record the round number of this speech
            'text': speech_content.get('text', ''),  # 使用完整的句子显示
            'content': speech_content  # 保持原始字典格式用于训练
        }
        self.speech_history.append(speech_record)
        
        # Move to next player
        self.next_player()
    
    def record_vote(self, player_id: int, target_id: int) -> bool:
        """
        Record a player's vote
        
        Args:
            player_id: Voter's ID
            target_id: Target player's ID
            
        Returns:
            bool: Whether the vote was successfully recorded
        """
        # Check if player has already voted
        if player_id in self.has_voted:
            return False
            
        # Check if target is valid
        if target_id < 0 or target_id >= self.num_players or target_id == player_id:
            return False
            
        # Record vote
        self.votes[player_id] = target_id
        self.has_voted.add(player_id)
        return True
    
    def _determine_winner(self) -> str:
        """
        Determine game winner
        
        Returns:
            Winning team ('werewolf' or 'villager')
        """
        # Count votes
        vote_count = defaultdict(int)
        for target_id in self.votes.values():
            vote_count[target_id] += 1
        
        # Find player with most votes
        max_votes = 0
        voted_out = -1
        for player_id, count in vote_count.items():
            if count > max_votes:
                max_votes = count
                voted_out = player_id
            elif count == max_votes:
                # In case of tie, randomly choose (can be adjusted based on game rules)
                if random.random() < 0.5:
                    voted_out = player_id
        
        # Determine winner
        if voted_out >= 0:
            # Determine team of voted out player
            voted_role = self.players[voted_out]['current_role']
            from config.default_config import ROLE_TEAMS
            voted_team = ROLE_TEAMS.get(voted_role, 'villager')
            
            # 使用配置中的反转规则设置
            use_reverse_rule = self.config.get('reverse_vote_rules', False)
            
            if use_reverse_rule:
                # 反转规则：统计投票情况
                villager_votes = 0
                werewolf_votes = 0
                
                for voter_id, target_id in self.votes.items():
                    if target_id == voted_out:
                        voter_role = self.players[voter_id]['current_role']
                        voter_team = ROLE_TEAMS.get(voter_role, 'villager')
                        
                        if voter_team == 'villager':
                            villager_votes += 1
                        else:  # werewolf team
                            werewolf_votes += 1
                
                # 反转投票规则：村民阵营投票多数，狼人获胜；狼人阵营投票多数，村民获胜
                if villager_votes > werewolf_votes:
                    # 村民投票多，狼人获胜
                    self.game_result = 'werewolf'
                elif werewolf_votes > villager_votes:
                    # 狼人投票多，村民获胜
                    self.game_result = 'villager'
                else:
                    # 平局则基于被投出者的阵营判定：投出狼人则村民胜，投出村民则狼人胜
                    self.game_result = 'villager' if voted_team == 'werewolf' else 'werewolf'
            else:
                # 标准规则：被投出狼人则村民获胜，被投出村民则狼人获胜
                self.game_result = 'villager' if voted_team == 'werewolf' else 'werewolf'
        else:
            # No one was voted out
            self.game_result = 'villager'
        
        return self.game_result
    
    def get_observation(self, player_id: int) -> Dict[str, Any]:
        """
        Get observation for specified player
        
        Args:
            player_id: Player ID
            
        Returns:
            Player observation (partially visible state)
        """
        # Create PlayerObservation object
        obs = PlayerObservation(player_id, self)
        
        # Return observation in dictionary format
        return {
            'player_id': player_id,
            'phase': self.phase,
            'round': self.round,
            'speech_round': self.speech_round,  # Add speech round info to observation
            'current_player': self.get_current_player(),
            'original_role': self.players[player_id]['original_role'],
            'known_info': self.players[player_id]['known_info'].copy(),
            'speech_history': self.speech_history.copy()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation
        
        Returns:
            Game state dictionary
        """
        return {
            'players': self.players,
            'phase': self.phase,
            'round': self.round,
            'speech_round': self.speech_round,  # Add speech round
            'center_cards': self.center_cards,
            'current_player': self.current_player,
            'votes': self.votes,
            'game_result': self.game_result
        }
        
    def get_player_role(self, player_id: int) -> str:
        """
        获取指定玩家的当前角色
        
        Args:
            player_id: 玩家ID
            
        Returns:
            角色名称字符串
        """
        if 0 <= player_id < len(self.players):
            return self.players[player_id]['current_role']
        return "unknown"


class PlayerObservation:
    """Player observation class, represents the partial game state visible to a player"""
    
    def __init__(self, player_id: int, game_state: GameState):
        self.player_id = player_id
        self.game_state = game_state
        
    def to_vector(self) -> np.ndarray:
        """
        Convert player observation to vector representation for neural network input
        
        Returns:
            Observation vector
        """
        # Simplified handling, actual project needs more complex vector representation
        player_data = self.game_state.players[self.player_id]
        
        # Create base vector
        vector_size = 50  # Adjust size based on actual needs
        vector = np.zeros(vector_size, dtype=np.float32)
        
        # Set basic information
        vector[0] = self.player_id
        vector[1] = self.game_state.GAME_PHASES.index(self.game_state.phase)
        vector[2] = self.game_state.round
        vector[3] = self.game_state.speech_round  # Add speech round to vector representation
        vector[4] = self.game_state.get_current_player()
        
        # Role information encoding
        all_roles = list(set(self.game_state.roles))
        role_idx = all_roles.index(player_data['original_role'])
        vector[5] = role_idx
        
        # Other information can be added as needed
        
        return vector

def process_state(state):
    validate_state(state)
    # 处理状态的其他逻辑 