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
    
    def step(self, action: Union[int, Dict[str, Any]]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute step in environment
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        try:
            if self.done:
                raise RuntimeError("Environment is done, please call reset() first")
            
            # Get current player
            player_id = self.current_player_id
            if player_id < 0:
                raise RuntimeError("Invalid current player ID")
            
            # 检查game_state类型
            if isinstance(self.game_state, dict):
                # 如果是字典，按字典方式处理
                phase = self.game_state.get('phase')
                terminated = phase == 'end'
                
                # Parse and execute action
                action_obj = self._parse_action(action, player_id)
                result = self._execute_action(action_obj)
                
                # 存储动作执行结果
                self.action_result = result
                
                # 更新当前玩家
                if 'current_player' in self.game_state:
                    self.current_player_id = self.game_state['current_player']
                
                # 计算奖励
                reward = 0.0
                
                # 获取观察和信息
                obs = self._get_observation()
                info = self._get_info()
                
                return obs, reward, terminated, False, info
            else:
                # 原来的对象方式处理
                # Parse and execute action
                action_obj = self._parse_action(action, player_id)
                result = self._execute_action(action_obj)
                
                # 存储动作执行结果，以便在info中返回
                self.action_result = result
                
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
            
        except Exception as e:
            # 处理异常，返回有意义的错误信息
            import traceback
            error_info = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'success': False,
                'message': f"Error executing step: {str(e)}"
            }
            
            # 返回空观察和错误信息
            empty_obs = {}
            return empty_obs, 0.0, False, False, error_info
    
    def _parse_action(self, action: Union[int, Dict, Action], player_id: int) -> Action:
        """
        Parse action ID into specific action object
        
        Args:
            action: Action ID, action dict, or action object
            player_id: Player ID
            
        Returns:
            Action object
        """
        # If already an action object, return directly
        if isinstance(action, Action):
            return action
            
        # If action is a dict, convert to action object
        if isinstance(action, dict):
            action_type = action.get('action_type')
            
            if action_type == 'NIGHT_ACTION':
                action_name = action.get('action_name', '')
                action_params = action.get('action_params', {})
                
                # 获取玩家原始角色
                player_role = None
                if isinstance(self.game_state, dict):
                    for player in self.game_state.get('players', []):
                        if player.get('player_id') == player_id:
                            player_role = player.get('original_role')
                            break
                else:
                    player_role = self.game_state.players[player_id]['original_role']
                    
                return create_night_action(player_id, player_role, action_name, **{})
                
            elif action_type == 'DAY_SPEECH':
                speech_type = action.get('speech_type', 'GENERAL')
                content = action.get('content', '')
                return create_speech(player_id, speech_type, content=content)
                
            elif action_type == 'VOTE':
                target_id = action.get('target_id', 0)
                return create_vote(player_id, target_id)
                
            else:
                return create_no_action(player_id)
        
        # Get current phase
        if isinstance(self.game_state, dict):
            phase = self.game_state.get('phase', 'night')
        else:
            phase = self.game_state.phase
        
        # Parse action based on different phases
        if phase == 'night':
            # Night action
            if isinstance(self.game_state, dict):
                # 从字典获取角色
                player_role = None
                for player in self.game_state.get('players', []):
                    if player.get('player_id') == player_id:
                        player_role = player.get('original_role')
                        break
                
                if not player_role:
                    return create_no_action(player_id)
            else:
                player_role = self.game_state.players[player_id]['original_role']
            
            # Get available night actions for this role
            available_actions = self.config.get('night_actions', {}).get(player_role, [])
            
            if not available_actions:
                # Role has no night actions
                return create_no_action(player_id)
            
            # Choose action
            action_index = action % len(available_actions)
            action_name = available_actions[action_index]
            
            # Create night action
            return create_night_action(player_id, player_role, action_name, **{})
            
        elif phase == 'day':
            # Day speech - Modified to use templates
            templates = self.config.get('speech_templates', [])
            if not templates:
                return create_no_action(player_id)
            
            # Choose speech template
            template_index = action % len(templates)
            template = templates[template_index]
            
            # Determine target player (if template needs it)
            if isinstance(self.game_state, dict):
                num_players = len(self.game_state.get('players', []))
            else:
                num_players = self.config['num_players']
                
            target_id = (action // len(templates)) % num_players
            
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
            # Vote
            if isinstance(self.game_state, dict):
                num_players = len(self.game_state.get('players', []))
            else:
                num_players = self.config['num_players']
                
            target_id = action % num_players
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
        
        try:
            # 检查game_state类型
            if isinstance(self.game_state, dict):
                # 如果是字典，按字典方式处理
                player_id = action.player_id
                
                if action.action_type == ActionType.NIGHT_ACTION:
                    if isinstance(action, NightAction):
                        # 模拟夜间动作结果
                        # 获取玩家角色
                        player_role = None
                        for player in self.game_state.get('players', []):
                            if player.get('player_id') == player_id:
                                player_role = player.get('original_role')
                                break
                                
                        # 处理特定角色的动作
                        if player_role == 'werewolf' and action.action_name == 'check_other_werewolves':
                            # 狼人查看其他狼人
                            werewolf_indices = self.game_state.get('werewolf_indices', [])
                            # 排除当前玩家
                            other_werewolves = [idx for idx in werewolf_indices if idx != player_id]
                            result = {
                                'success': True,
                                'message': 'Checked other werewolves',
                                'result': other_werewolves
                            }
                        # 处理其他角色的动作...
                        else:
                            result = {
                                'success': True,
                                'message': f'Executed night action for {player_role}: {action.action_name}',
                                'action': action.action_name,
                                'params': action.action_params
                            }
                        
                        # 更新当前玩家
                        # 查找角色在action_order中的位置
                        action_order = self.game_state.get('action_order', [])
                        if player_role in action_order:
                            # 确定下一个角色
                            current_index = action_order.index(player_role)
                            next_index = (current_index + 1) % len(action_order)
                            next_role = action_order[next_index]
                            
                            # 查找具有该角色的玩家
                            for player in self.game_state.get('players', []):
                                if player.get('original_role') == next_role:
                                    self.game_state['current_player'] = player.get('player_id')
                                    break
                        
                elif action.action_type == ActionType.DAY_SPEECH:
                    if isinstance(action, DaySpeech):
                        # 记录发言内容
                        speech_history = self.game_state.get('speech_history', [])
                        speech_history.append({
                            'player_id': player_id,
                            'content': action.content,
                            'round': self.game_state.get('speech_round', 0)
                        })
                        self.game_state['speech_history'] = speech_history
                        
                        result = {
                            'success': True,
                            'message': f'Player {player_id} completed speech'
                        }
                        
                        # 更新当前玩家
                        next_player = (player_id + 1) % len(self.game_state.get('players', []))
                        self.game_state['current_player'] = next_player
                        
                        # 如果所有玩家都发言完毕，进入下一轮
                        if next_player == 0:
                            self.game_state['speech_round'] = self.game_state.get('speech_round', 0) + 1
                            
                            # 如果达到最大发言轮数，进入投票阶段
                            if self.game_state.get('speech_round', 0) >= self.game_state.get('max_speech_rounds', 3):
                                self.game_state['phase'] = 'vote'
                        
                elif action.action_type == ActionType.VOTE:
                    if isinstance(action, VoteAction):
                        # 检查是否是投票阶段
                        if self.game_state.get('phase') != 'vote':
                            result = {
                                'success': False,
                                'message': 'Voting is only allowed in the voting phase'
                            }
                            return result
                            
                        # 记录投票
                        votes = self.game_state.get('votes', {})
                        votes[str(player_id)] = action.target_id
                        self.game_state['votes'] = votes
                        
                        result = {
                            'success': True,
                            'message': f'Player {player_id} voted for player {action.target_id}'
                        }
                        
                        # 更新当前玩家
                        next_player = (player_id + 1) % len(self.game_state.get('players', []))
                        self.game_state['current_player'] = next_player
                        
                        # 如果所有玩家都投票完毕，结束游戏
                        if len(votes) == len(self.game_state.get('players', [])):
                            self.game_state['phase'] = 'end'
                            # 计算投票结果
                            vote_counts = {}
                            for target in votes.values():
                                vote_counts[target] = vote_counts.get(target, 0) + 1
                                
                            max_votes = 0
                            most_voted = -1
                            for target, count in vote_counts.items():
                                if count > max_votes:
                                    max_votes = count
                                    most_voted = target
                                    
                            # 设置投票结果
                            self.game_state['voting_results'] = {
                                'most_voted': most_voted,
                                'vote_counts': vote_counts
                            }
                            
                            # 确定获胜团队
                            if most_voted in self.game_state.get('werewolf_indices', []):
                                self.game_state['winner'] = 'villager'  # 狼人被投出，村民胜利
                            else:
                                self.game_state['winner'] = 'werewolf'  # 平民被投出，狼人胜利
                
                elif action.action_type == ActionType.NO_ACTION:
                    # 无动作，进入下一玩家
                    next_player = (player_id + 1) % len(self.game_state.get('players', []))
                    self.game_state['current_player'] = next_player
                    result = {
                        'success': True,
                        'message': 'No action'
                    }
                    
                return result
                
            else:
                # 原来的对象方式处理
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
                        # Check if it's voting phase
                        if self.game_state.phase != 'vote':
                            result = {'success': False, 'message': 'Voting is only allowed in the final round'}
                            return result
                        
                        # Record vote to game state
                        vote_success = self.game_state.record_vote(action.player_id, action.target_id)
                        if vote_success:
                            result = {'success': True, 'message': f'Player {action.player_id} voted for player {action.target_id}'}
                            # Move to next player only if vote was successful
                            self.game_state.next_player()
                        else:
                            result = {'success': False, 'message': f'Player {action.player_id} has already voted or invalid target'}
                        
                elif action.action_type == ActionType.NO_ACTION:
                    # No action, proceed to next phase
                    self.game_state.next_player()
                    result = {'success': True, 'message': 'No action'}
                    
            return result
            
        except Exception as e:
            # 捕获所有异常并返回错误信息
            import traceback
            error_result = {
                'success': False,
                'message': f'Error executing action: {str(e)}',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            return error_result
    
    def _compute_reward(self, player_id: int, action: Action, result: Dict[str, Any]) -> float:
        """
        Calculate reward function
        
        Args:
            player_id: Player executing the action
            action: Executed action
            result: Action execution result
            
        Returns:
            Reward value (always 0 as rewards are not used)
        """
        # 由于不是强化学习环境，奖励始终为0
        return 0.0
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        Get current observation
        
        Returns:
            Observation dictionary for current player
        """
        if self.current_player_id < 0:
            return {}
            
        if isinstance(self.game_state, dict):
            # 如果是字典，构造观察
            observation = {
                'phase': self.game_state.get('phase', 'night'),
                'round': self.game_state.get('round', 0),
                'speech_round': self.game_state.get('speech_round', 0),
                'current_player': self.game_state.get('current_player', self.current_player_id),
                'player_id': self.current_player_id
            }
            
            # 获取当前玩家信息
            current_player = None
            for player in self.game_state.get('players', []):
                if player.get('player_id') == self.current_player_id:
                    current_player = player
                    break
                    
            if current_player:
                observation['original_role'] = current_player.get('original_role')
                observation['current_role'] = current_player.get('current_role')
                observation['team'] = current_player.get('team')
                
            # 添加其他玩家信息（隐藏角色）
            players_info = []
            for player in self.game_state.get('players', []):
                player_id = player.get('player_id')
                if player_id != self.current_player_id:
                    # 隐藏角色信息
                    players_info.append({
                        'player_id': player_id,
                        'name': player.get('name'),
                        'is_human': player.get('is_human')
                    })
                else:
                    # 包含当前玩家完整信息
                    players_info.append(player)
                    
            observation['players'] = players_info
            
            # 投票信息
            if 'votes' in self.game_state:
                observation['votes'] = self.game_state['votes']
                
            # 中心牌信息 - 默认隐藏
            center_cards = ['?' for _ in range(len(self.game_state.get('center_cards', [])))]
            observation['center_cards'] = center_cards
            
            # 根据角色添加特定信息
            current_role = current_player.get('original_role') if current_player else None
            
            # 狼人可以看到其他狼人
            if current_role == 'werewolf':
                werewolf_indices = self.game_state.get('werewolf_indices', [])
                observation['werewolf_indices'] = werewolf_indices
                
            # 其他角色的特殊观察信息...
            
            return observation
        else:
            # 对象方式
            return self.game_state.get_observation(self.current_player_id)
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information
        
        Returns:
            Information dictionary
        """
        info = {
            'phase': self.game_state.phase,
            'round': self.game_state.round,
            'speech_round': self.game_state.speech_round,
            'current_player': self.current_player_id,
            'cumulative_rewards': dict(self.rewards)
        }
        
        # 如果存在动作执行结果，添加到info中
        if hasattr(self, 'action_result'):
            info.update(self.action_result)
            
        return info
    
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