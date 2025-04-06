"""
Werewolf Game Environment - Based on Gymnasium
"""
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict

from werewolf_env.state import GameState, PlayerObservation
from werewolf_env.actions import ActionType, Action, NightAction, DaySpeech, VoteAction, NoAction
from werewolf_env.actions import create_night_action, create_speech, create_vote, create_no_action
from werewolf_env.roles import create_role
from config.default_config import DEFAULT_GAME_CONFIG, ROLE_TEAMS


class WerewolfEnv(gym.Env):
    """
    Werewolf Game Environment
    
    Features:
    1. Multi-agent environment
    2. Partially observable state
    3. Discrete action space
    4. Modified rules:
       - Three rounds of sequential speech during day
       - Single round of voting at night, reversed victory conditions
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None):
        """
        Initialize environment
        
        Args:
            config: Game configuration, use default if None
            render_mode: Render mode
        """
        # Merge configuration
        self.config = DEFAULT_GAME_CONFIG.copy()
        if config:
            self.config.update(config)
        
        self.render_mode = render_mode
        
        # Game state
        self.game_state = None
        
        # Current player
        self.current_player_id = -1
        
        # Game over flag
        self.done = False
        
        # Cumulative rewards for each player
        self.rewards = defaultdict(float)
        
        # Set up observation and action spaces
        self._setup_spaces()
    
    def _setup_spaces(self) -> None:
        """Set up observation and action spaces"""
        # Define observation space (this is a simplified version, actual project may need more complex representation)
        num_roles = len(set(self.config['roles']))
        num_players = self.config['num_players']
        
        # Observation space will be a mix of multi-discrete and Box spaces
        # Since Gymnasium doesn't directly support mixed spaces, we use Dict space
        self.observation_space = spaces.Dict({
            # Player ID
            'player_id': spaces.Discrete(num_players),
            # Game phase
            'phase': spaces.Discrete(len(GameState.GAME_PHASES)),
            # Current round
            'round': spaces.Discrete(self.config['max_rounds'] + 1),
            # Current speech round
            'speech_round': spaces.Discrete(4),  # 0-3 rounds
            # Current player
            'current_player': spaces.Discrete(num_players),
            # Original role (one-hot)
            'original_role': spaces.Discrete(num_roles),
            # Known information (vector representation)
            'known_info': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(20,),  # Adjust size based on project needs
                dtype=np.float32
            ),
            # Speech history (vector representation)
            'speech_history': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(3 * num_players, 10),  # 3 rounds * num_players * feature_dim
                dtype=np.float32
            )
        })
        
        # Define action space
        # Since different phases have different action spaces, we use a discrete space to represent all possible actions
        # Actions will be parsed based on current phase during execution
        
        # Number of night actions (sum of actions for each role)
        num_night_actions = sum(len(actions) for actions in self.config.get('night_actions', {}).values())
        
        # Number of speech templates
        num_speech_templates = len(self.config.get('speech_templates', [])) * num_players
        
        # Number of vote targets (can vote for any player)
        num_vote_targets = num_players
        
        # Total number of actions
        total_actions = num_night_actions + num_speech_templates + num_vote_targets + 1  # +1 for NO_ACTION
        
        self.action_space = spaces.Discrete(total_actions)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset environment
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            Observation and info dictionary
        """
        super().reset(seed=seed)
        
        # Update configuration (if provided)
        if options and 'config' in options:
            self.config.update(options['config'])
            self._setup_spaces()
        
        # Create new game state
        self.game_state = GameState(self.config)
        
        # Reset current player
        self.current_player_id = self.game_state.get_current_player()
        
        # Reset game over flag
        self.done = False
        
        # Reset rewards
        self.rewards = defaultdict(float)
        
        # Set initial phase to night
        self.game_state.phase = 'night'
        
        # Render (if needed)
        if self.render_mode == 'human':
            self.render()
            
        # Return initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute action
        
        Args:
            action: Action ID
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self.done:
            raise RuntimeError("Environment is done, please call reset() first")
        
        # Get current player
        player_id = self.current_player_id
        if player_id < 0:
            raise RuntimeError("Invalid current player ID")
        
        # Parse and execute action
        action_obj = self._parse_action(action, player_id)
        result = self._execute_action(action_obj)
        
        # Update current player
        self.current_player_id = self.game_state.get_current_player()
        
        # Calculate reward
        reward = self._compute_reward(player_id, action_obj, result)
        self.rewards[player_id] += reward
        
        # Check if game is over
        terminated = self.game_state.phase == 'end'
        self.done = terminated
        
        # Check if maximum steps exceeded
        truncated = False
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        
        # Render (if needed)
        if self.render_mode == 'human':
            self.render()
            
        return obs, reward, terminated, truncated, info
    
    def _parse_action(self, action: Union[int, Action], player_id: int) -> Action:
        """
        Parse action ID into specific action object
        
        Args:
            action: Action ID or action object
            player_id: Player ID
            
        Returns:
            Action object
        """
        # If already an action object, return directly
        if isinstance(action, Action):
            return action
            
        # Get current phase
        phase = self.game_state.phase
        
        # Parse action based on different phases
        if phase == 'night':
            # Night action
            role = self.game_state.players[player_id]['original_role']
            
            # Get available night actions for this role
            available_actions = self.config.get('night_actions', {}).get(role, [])
            
            if not available_actions:
                # Role has no night actions
                return create_no_action(player_id)
            
            # Choose action
            action_index = action % len(available_actions)
            action_name = available_actions[action_index]
            
            # Create night action
            # Note: This is simplified, might need more parameters in practice
            return create_night_action(player_id, role, action_name)
            
        elif phase == 'day':
            # Day speech - Modified to use templates
            templates = self.config.get('speech_templates', [])
            if not templates:
                return create_no_action(player_id)
            
            # Choose speech template
            template_index = action % len(templates)
            template = templates[template_index]
            
            # Determine target player (if template needs it)
            target_id = (action // len(templates)) % self.config['num_players']
            
            # Create speech based on template type
            if template == 'CLAIM_ROLE':
                return create_speech(player_id, template, role="villager")
            elif template == 'CLAIM_ACTION_RESULT':
                return create_speech(player_id, template, role="seer", action="check_player", target=target_id, result="werewolf")
            elif template == 'ACCUSE':
                return create_speech(player_id, template, target_id=target_id, accused_role="werewolf")
            elif template == 'DEFEND':
                return create_speech(player_id, template, not_role="werewolf", reason="I'm innocent")
            elif template == 'VOTE_INTENTION':
                return create_speech(player_id, template, target_id=target_id)
            else:
                return create_no_action(player_id)
            
        elif phase == 'vote':
            # Vote - Modified for reversed victory conditions
            target_id = action % self.config['num_players']
            return create_vote(player_id, target_id)
            
        # Default to no action
        return create_no_action(player_id)
    
    def _execute_action(self, action: Action) -> Dict[str, Any]:
        """
        Execute action and return result
        
        Args:
            action: Action object
            
        Returns:
            Execution result dictionary
        """
        result = {'success': False, 'message': 'Unknown action'}
        
        if action.action_type == ActionType.NIGHT_ACTION:
            if isinstance(action, NightAction):
                result = self.game_state.perform_night_action(action.action_params)
                
        elif action.action_type == ActionType.DAY_SPEECH:
            if isinstance(action, DaySpeech):
                # Record speech content to game state
                self.game_state.record_speech(action.player_id, action.content)
                result = {'success': True, 'message': f'Player {action.player_id} completed speech'}
                
        elif action.action_type == ActionType.VOTE:
            if isinstance(action, VoteAction):
                # Record vote to game state
                self.game_state.record_vote(action.player_id, action.target_id)
                result = {'success': True, 'message': f'Player {action.player_id} voted for player {action.target_id}'}
                
        elif action.action_type == ActionType.NO_ACTION:
            # No action, proceed to next phase
            self.game_state.next_player()
            result = {'success': True, 'message': 'No action'}
            
        return result
    
    def _compute_reward(self, player_id: int, action: Action, result: Dict[str, Any]) -> float:
        """
        Calculate reward function
        
        Args:
            player_id: Player executing the action
            action: Executed action
            result: Action execution result
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Basic reward: action success or failure
        if result.get('success', False):
            reward += 0.1
        else:
            reward -= 0.05
            
        # Game end reward
        if self.game_state.phase == 'end':
            # Get player role and team
            player_role = self.game_state.players[player_id]['current_role']
            player_team = ROLE_TEAMS.get(player_role, 'villager')
            
            # Game result
            game_result = self.game_state.game_result
            
            # Assign rewards based on team
            if game_result == player_team:
                # Victory
                reward += 1.0
            else:
                # Defeat
                reward -= 0.5
                
            # Special reward: extra reward for successful reversed voting
            if player_team == 'werewolf' and game_result == 'werewolf':
                # If werewolves won, check if this werewolf successfully guided villager votes
                for voter_id, target_id in self.game_state.votes.items():
                    voter_role = self.game_state.players[voter_id]['current_role']
                    voter_team = ROLE_TEAMS.get(voter_role, 'villager')
                    if voter_team == 'villager' and target_id in self.game_state.votes.values():
                        # Villager was guided to vote, give werewolf extra reward
                        reward += 0.3
                        break
                
        return reward
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        Get current observation
        
        Returns:
            Observation dictionary for current player
        """
        if self.current_player_id < 0:
            return {}
            
        return self.game_state.get_observation(self.current_player_id)
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information
        
        Returns:
            Information dictionary
        """
        return {
            'phase': self.game_state.phase,
            'round': self.game_state.round,
            'speech_round': self.game_state.speech_round,  # Add speech round information
            'current_player': self.current_player_id,
            'cumulative_rewards': dict(self.rewards)
        }
    
    def render(self) -> Optional[Union[str, np.ndarray]]:
        """
        Render environment
        
        Returns:
            Different types of rendering results based on render mode
        """
        if self.render_mode is None:
            return None
            
        if self.render_mode == "ansi":
            return self._render_text()
            
        return None
    
    def _render_text(self) -> str:
        """
        Text rendering
        
        Returns:
            Text representation of game state
        """
        if self.game_state is None:
            return "Environment not initialized"
            
        lines = []
        lines.append("=" * 50)
        lines.append(f"Game phase: {self.game_state.phase}")
        lines.append(f"Current round: {self.game_state.round}")
        
        if self.game_state.phase == 'day':
            lines.append(f"Current speech round: {self.game_state.speech_round}/3")  # Show current speech round
            
        lines.append(f"Current player: {self.current_player_id}")
        
        # Player information
        lines.append("\nPlayer states:")
        for i, player in enumerate(self.game_state.players):
            role_info = f"[Original role: {player['original_role']}]"
            lines.append(f"  Player {i}: {role_info}")
            
        # Speech history
        if self.game_state.speech_history:
            lines.append("\nSpeech history:")
            for i, speech in enumerate(self.game_state.speech_history[-5:]):  # Only show last 5 entries
                lines.append(f"  Round {speech['round']}, Player {speech['player_id']}: {speech['content']}")
                
        # Voting status
        if self.game_state.votes:
            lines.append("\nVoting status:")
            for voter, target in self.game_state.votes.items():
                lines.append(f"  Player {voter} voted for Player {target}")
                
        # Game result
        if self.game_state.phase == 'end':
            lines.append("\nGame result:")
            lines.append(f"  Winning team: {self.game_state.game_result}")
            
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def close(self) -> None:
        """Close environment"""
        pass


# Example usage
if __name__ == "__main__":
    # Create environment
    env = WerewolfEnv(render_mode="human")
    
    # Reset environment
    obs, info = env.reset()
    
    # Simulate some random actions
    done = False
    while not done:
        # Randomly choose action
        action = env.action_space.sample()
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check if done
        done = terminated or truncated
    
    # Close environment
    env.close() 