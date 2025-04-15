"""
Base class for Werewolf game agents
"""
from typing import Dict, List, Any, Tuple
import numpy as np
import random
from abc import ABC, abstractmethod

from werewolf_env.state import GameState, PlayerObservation
from werewolf_env.actions import (
    Action, 
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
    
    def initialize(self, game_state) -> None:
        """
        Initialize agent state
        
        Args:
            game_state: Game state (can be dict or GameState object)
        """
        self.game_state = game_state
        
        # 处理字典格式的game_state
        if isinstance(game_state, dict):
            players = game_state.get('players', [])
            # 查找当前玩家信息
            player = None
            for p in players:
                if p.get('player_id') == self.player_id:
                    player = p
                    break
                    
            if player:
                self.original_role = player.get('original_role')
                self.current_role = player.get('current_role')
            else:
                # 找不到玩家信息，使用默认值
                self.original_role = "unknown"
                self.current_role = "unknown"
                
            self.current_phase = game_state.get('phase')
        else:
            # 处理GameState对象格式
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
        
        # 处理字典格式的game_state
        if isinstance(self.game_state, dict):
            players = self.game_state.get('players', [])
            num_players = len(players)
            other_players = [p.get('player_id') for p in players 
                            if p.get('player_id') != self.player_id]
        else:
            # 处理GameState对象格式
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

    def get_action(self, game_state) -> Action:
        """
        获取动作 - 兼容训练器接口
        
        Args:
            game_state: 游戏状态 (可以是字典或GameState对象)
            
        Returns:
            Action
        """
        # 将GameState转换为观察结果格式
        if isinstance(game_state, dict):
            # 如果已经是字典格式，直接使用
            observation = {
                'phase': game_state.get('phase'),
                'round': game_state.get('round', 0),
                'current_player': game_state.get('current_player'),
                'speech_round': game_state.get('speech_round', 0),
                'votes': game_state.get('votes', {})
            }
            
            # 查找角色信息
            player = None
            for p in game_state.get('players', []):
                if p.get('player_id') == self.player_id:
                    player = p
                    break
                    
            if player:
                observation['original_role'] = player.get('original_role')
                observation['current_role'] = player.get('current_role')
        else:
            # 原始GameState对象格式转换
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
        
        # 如果没有初始化过，先初始化
        if self.original_role is None:
            self.initialize(game_state)
        
        # 调用act方法获取动作
        return self.act(observation) 