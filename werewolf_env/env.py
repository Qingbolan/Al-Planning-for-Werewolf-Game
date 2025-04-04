"""
狼人杀游戏环境 - 基于Gymnasium
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
    狼人杀游戏环境
    
    特点：
    1. 多智能体环境
    2. 部分可观察状态
    3. 离散动作空间
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None):
        """
        初始化环境
        
        Args:
            config: 游戏配置，如果为None则使用默认配置
            render_mode: 渲染模式
        """
        # 合并配置
        self.config = DEFAULT_GAME_CONFIG.copy()
        if config:
            self.config.update(config)
        
        self.render_mode = render_mode
        
        # 游戏状态
        self.game_state = None
        
        # 当前玩家
        self.current_player_id = -1
        
        # 游戏是否结束
        self.done = False
        
        # 每个玩家的累积奖励
        self.rewards = defaultdict(float)
        
        # 设置观察空间和动作空间
        self._setup_spaces()
    
    def _setup_spaces(self) -> None:
        """设置观察空间和动作空间"""
        # 定义观察空间 (这是一个简化版本，实际项目中可能需要更复杂的表示)
        num_roles = len(set(self.config['roles']))
        num_players = self.config['num_players']
        
        # 观察空间将是一个多离散空间和一个Box空间的混合
        # 由于Gymnasium不直接支持混合空间，这里使用Dict空间
        self.observation_space = spaces.Dict({
            # 玩家ID
            'player_id': spaces.Discrete(num_players),
            # 游戏阶段
            'phase': spaces.Discrete(len(GameState.GAME_PHASES)),
            # 当前轮次
            'round': spaces.Discrete(self.config['max_rounds'] + 1),
            # 当前玩家
            'current_player': spaces.Discrete(num_players),
            # 初始角色 (one-hot)
            'original_role': spaces.Discrete(num_roles),
            # 已知信息 (向量表示)
            'known_info': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(20,),  # 具体大小需要根据项目需求调整
                dtype=np.float32
            ),
            # 发言历史 (向量表示)
            'speech_history': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.config['max_steps_day'], 10),  # 具体大小需要根据项目需求调整
                dtype=np.float32
            )
        })
        
        # 定义动作空间
        # 由于不同阶段有不同的动作空间，这里使用一个离散空间来表示所有可能的动作
        # 实际执行时需要根据当前阶段进行解析
        
        # 夜晚行动数量 (每个角色的行动数量之和)
        num_night_actions = sum(len(actions) for actions in self.config.get('night_actions', {}).values())
        
        # 白天发言模板数量
        num_speech_templates = len(self.config.get('speech_templates', [])) * num_players
        
        # 投票目标数量 (可以投给任何玩家)
        num_vote_targets = num_players
        
        # 总动作数量
        total_actions = num_night_actions + num_speech_templates + num_vote_targets + 1  # +1 for NO_ACTION
        
        self.action_space = spaces.Discrete(total_actions)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 重置选项
            
        Returns:
            观察结果和信息字典
        """
        super().reset(seed=seed)
        
        # 更新配置（如果有的话）
        if options and 'config' in options:
            self.config.update(options['config'])
            self._setup_spaces()
        
        # 创建新的游戏状态
        self.game_state = GameState(self.config)
        
        # 重置当前玩家
        self.current_player_id = self.game_state.get_current_player()
        
        # 重置游戏结束标志
        self.done = False
        
        # 重置奖励
        self.rewards = defaultdict(float)
        
        # 初始阶段设为夜晚
        self.game_state.phase = 'night'
        
        # 渲染（如果需要）
        if self.render_mode == 'human':
            self.render()
            
        # 返回初始观察
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        执行动作
        
        Args:
            action: 动作ID
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self.done:
            raise RuntimeError("环境已经结束，请先调用reset()重置环境")
        
        # 获取当前玩家
        player_id = self.current_player_id
        if player_id < 0:
            raise RuntimeError("无效的当前玩家ID")
        
        # 解析并执行动作
        action_obj = self._parse_action(action, player_id)
        result = self._execute_action(action_obj)
        
        # 更新当前玩家
        self.current_player_id = self.game_state.get_current_player()
        
        # 计算奖励
        reward = self._compute_reward(player_id, action_obj, result)
        self.rewards[player_id] += reward
        
        # 检查游戏是否结束
        terminated = self.game_state.phase == 'end'
        self.done = terminated
        
        # 检查是否超过最大步数
        truncated = False
        
        # 获取观察和信息
        obs = self._get_observation()
        info = self._get_info()
        
        # 渲染（如果需要）
        if self.render_mode == 'human':
            self.render()
            
        return obs, reward, terminated, truncated, info
    
    def _parse_action(self, action: int, player_id: int) -> Action:
        """
        解析动作ID为具体的动作对象
        
        Args:
            action: 动作ID
            player_id: 玩家ID
            
        Returns:
            动作对象
        """
        # 这里需要根据具体项目需求实现动作解析
        # 示例实现：
        
        # 获取当前阶段
        phase = self.game_state.phase
        
        # 根据不同阶段解析动作
        if phase == 'night':
            # 夜晚行动
            role = self.game_state.players[player_id]['original_role']
            
            # 获取该角色可用的夜晚行动
            available_actions = self.config.get('night_actions', {}).get(role, [])
            
            if not available_actions:
                # 该角色没有夜晚行动
                return create_no_action(player_id)
            
            # 选择行动
            action_index = action % len(available_actions)
            action_name = available_actions[action_index]
            
            # 创建夜晚行动
            # 注意：这里简化处理，实际上可能需要更多参数
            return create_night_action(player_id, role, action_name)
            
        elif phase == 'day':
            # 白天发言
            templates = self.config.get('speech_templates', [])
            if not templates:
                return create_no_action(player_id)
            
            # 选择发言模板
            template_index = action % len(templates)
            
            # 创建发言
            # 注意：这里简化处理，实际上需要根据模板填充具体内容
            speech_type = "CLAIM_ROLE"  # 示例
            return create_speech(player_id, speech_type, role="villager")
            
        elif phase == 'vote':
            # 投票
            target_id = action % self.config['num_players']
            return create_vote(player_id, target_id)
            
        # 默认返回无行动
        return create_no_action(player_id)
    
    def _execute_action(self, action: Action) -> Dict[str, Any]:
        """
        执行动作
        
        Args:
            action: 动作对象
            
        Returns:
            执行结果
        """
        result = {'success': False}
        
        if action.action_type == ActionType.NIGHT_ACTION:
            # 夜晚行动
            night_action = action  # type: NightAction
            result = self.game_state.perform_night_action(night_action.action_params)
            
        elif action.action_type == ActionType.DAY_SPEECH:
            # 白天发言
            speech = action  # type: DaySpeech
            self.game_state.record_speech(speech.player_id, {
                'type': speech.speech_type,
                'content': speech.content
            })
            result = {'success': True, 'speech_recorded': True}
            
        elif action.action_type == ActionType.VOTE:
            # 投票
            vote = action  # type: VoteAction
            self.game_state.record_vote(vote.player_id, vote.target_id)
            result = {'success': True, 'vote_recorded': True}
            
        elif action.action_type == ActionType.NO_ACTION:
            # 无行动
            # 直接进入下一个玩家
            self.game_state.next_player()
            result = {'success': True, 'no_action': True}
        
        return result
    
    def _compute_reward(self, player_id: int, action: Action, result: Dict[str, Any]) -> float:
        """
        计算奖励
        
        Args:
            player_id: 玩家ID
            action: 执行的动作
            result: 执行结果
            
        Returns:
            奖励值
        """
        reward = 0.0
        
        # 游戏结束时计算胜利奖励
        if self.game_state.phase == 'end' and self.game_state.game_result is not None:
            # 获取玩家所属阵营
            player_role = self.game_state.players[player_id]['current_role']
            player_team = ROLE_TEAMS.get(player_role, 'villager_team')
            
            # 判断是否胜利
            if player_team == self.game_state.game_result:
                reward += self.config.get('reward_team_win', 1.0)
            else:
                reward += self.config.get('reward_team_loss', -1.0)
        
        # 根据行动类型给予中间奖励
        if action.action_type == ActionType.NIGHT_ACTION:
            # 夜晚行动奖励
            # 例如：正确识别角色的奖励
            if result.get('success') and 'result' in result:
                reward += self.config.get('reward_correct_identify', 0.0) * 0.5
                
        elif action.action_type == ActionType.DAY_SPEECH:
            # 白天发言奖励
            # 例如：成功说服他人的奖励（这需要在后续投票中体现）
            pass
            
        elif action.action_type == ActionType.VOTE:
            # 投票奖励
            # 在游戏结束时已经考虑了胜利奖励
            pass
        
        return reward
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        获取当前玩家的观察
        
        Returns:
            观察字典
        """
        if self.current_player_id < 0:
            # 如果没有当前玩家，返回空观察
            return {}
        
        # 获取当前玩家的观察
        return self.game_state.get_observation(self.current_player_id)
    
    def _get_info(self) -> Dict[str, Any]:
        """
        获取额外信息
        
        Returns:
            信息字典
        """
        return {
            'phase': self.game_state.phase,
            'round': self.game_state.round,
            'current_player': self.current_player_id,
            'rewards': dict(self.rewards),
            'done': self.done
        }
    
    def render(self) -> Optional[Union[str, np.ndarray]]:
        """
        渲染环境
        
        Returns:
            根据render_mode返回渲染结果
        """
        if self.render_mode is None:
            return None
            
        if self.render_mode == 'ansi':
            return self._render_text()
            
        # human模式下直接打印
        print(self._render_text())
        return None
    
    def _render_text(self) -> str:
        """
        生成文本渲染
        
        Returns:
            文本渲染结果
        """
        if self.game_state is None:
            return "环境尚未初始化"
            
        lines = []
        lines.append(f"===== 狼人杀游戏 - 轮次 {self.game_state.round} =====")
        lines.append(f"当前阶段: {self.game_state.phase}")
        lines.append(f"当前玩家: {self.current_player_id}")
        
        # 玩家信息
        lines.append("\n--- 玩家信息 ---")
        for i, player in enumerate(self.game_state.players):
            role = "???" if i != self.current_player_id else player['original_role']
            lines.append(f"玩家 {i}: 初始角色={role}")
        
        # 游戏状态
        if self.game_state.phase == 'night':
            lines.append("\n--- 夜晚行动 ---")
            if self.current_player_id >= 0:
                role = self.game_state.players[self.current_player_id]['original_role']
                lines.append(f"玩家 {self.current_player_id} ({role}) 正在执行夜晚行动")
        
        elif self.game_state.phase == 'day':
            lines.append("\n--- 白天发言 ---")
            for speech in self.game_state.speech_history:
                lines.append(f"玩家 {speech['player_id']} 说: {speech['content']}")
            
            if self.current_player_id >= 0:
                lines.append(f"玩家 {self.current_player_id} 正在发言")
        
        elif self.game_state.phase == 'vote':
            lines.append("\n--- 投票阶段 ---")
            for voter, target in self.game_state.votes.items():
                lines.append(f"玩家 {voter} 投票给 玩家 {target}")
                
            if self.current_player_id >= 0:
                lines.append(f"玩家 {self.current_player_id} 正在投票")
        
        elif self.game_state.phase == 'end':
            lines.append("\n--- 游戏结束 ---")
            lines.append(f"胜利阵营: {self.game_state.game_result}")
            
            lines.append("\n最终角色:")
            for i, player in enumerate(self.game_state.players):
                lines.append(f"玩家 {i}: 初始角色={player['original_role']}, 最终角色={player['current_role']}")
        
        return "\n".join(lines)
    
    def close(self) -> None:
        """关闭环境"""
        pass


# 示例使用
if __name__ == "__main__":
    # 创建环境
    env = WerewolfEnv(render_mode="human")
    
    # 重置环境
    obs, info = env.reset()
    
    # 模拟一些随机动作
    done = False
    while not done:
        # 随机选择动作
        action = env.action_space.sample()
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 检查是否结束
        done = terminated or truncated
    
    # 关闭环境
    env.close() 