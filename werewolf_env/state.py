"""
狼人杀游戏状态表示
"""
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import copy
import random
from collections import defaultdict

from werewolf_env.roles import create_role
from utils.common import validate_state

class GameState:
    """游戏状态类，维护完整的游戏状态"""
    
    GAME_PHASES = ['init', 'night', 'day', 'vote', 'end']
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化游戏状态
        
        Args:
            config: 游戏配置字典
        """
        self.config = config
        self.num_players = config['num_players']
        self.num_center_cards = config['num_center_cards']
        
        # 游戏阶段
        self.phase = 'init'
        self.round = 0
        
        # 角色分配
        self.roles = copy.deepcopy(config['roles'])
        random.shuffle(self.roles)
        
        # 玩家状态
        self.players = []
        for i in range(self.num_players):
            self.players.append({
                'id': i,
                'original_role': self.roles[i],
                'current_role': self.roles[i],
                'known_info': [],
                'belief_states': defaultdict(lambda: {role: 1/len(self.roles) for role in set(self.roles)})
            })
        
        # 中央牌堆
        self.center_cards = self.roles[self.num_players:]
        
        # 角色实例
        self.role_instances = {}
        for i in range(self.num_players):
            self.role_instances[i] = create_role(self.roles[i], i)
        
        # 历史记录
        self.action_history = []
        self.speech_history = []
        
        # 投票结果
        self.votes = {}
        
        # 游戏结果
        self.game_result = None
        
        # 当前轮次玩家
        self.current_player = 0
        
        # 当前夜晚行动角色索引
        self.night_action_index = 0
        self.night_action_roles = self._get_night_action_roles()
        
    def _get_night_action_roles(self) -> List[int]:
        """获取有夜间行动的角色列表"""
        night_roles = []
        
        # 根据配置中的角色行动顺序确定夜间行动顺序
        role_order = self.config.get('role_action_order', [
            'werewolf', 'minion', 'seer', 'robber', 'troublemaker', 'insomniac'
        ])
        
        for role in role_order:
            for i, player in enumerate(self.players):
                if player['original_role'] == role:
                    night_roles.append(i)
        
        return night_roles
    
    def get_current_player(self) -> int:
        """获取当前行动玩家ID"""
        if self.phase == 'night':
            if self.night_action_index < len(self.night_action_roles):
                return self.night_action_roles[self.night_action_index]
            return -1
        elif self.phase in ['day', 'vote']:
            return self.current_player
        return -1
    
    def next_player(self) -> int:
        """移动到下一个玩家，返回新的当前玩家ID"""
        if self.phase == 'night':
            self.night_action_index += 1
            if self.night_action_index >= len(self.night_action_roles):
                # 夜晚阶段结束
                self.phase = 'day'
                self.current_player = 0
                return self.current_player
            return self.night_action_roles[self.night_action_index]
        
        elif self.phase == 'day':
            self.current_player = (self.current_player + 1) % self.num_players
            # 如果所有玩家都发言完毕，进入投票阶段
            if self.current_player == 0:
                self.phase = 'vote'
            return self.current_player
        
        elif self.phase == 'vote':
            self.current_player = (self.current_player + 1) % self.num_players
            # 如果所有玩家都投票完毕，结束游戏
            if self.current_player == 0 and len(self.votes) == self.num_players:
                self.phase = 'end'
                self._determine_winner()
            return self.current_player
            
        return -1
    
    def perform_night_action(self, action_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行夜晚行动
        
        Args:
            action_params: 行动参数
            
        Returns:
            行动结果
        """
        player_id = self.get_current_player()
        if player_id < 0 or self.phase != 'night':
            return {'success': False, 'message': 'Invalid action phase or player'}
        
        player_role = self.role_instances[player_id]
        result = player_role.night_action(self.to_dict(), action_params)
        
        # 记录行动历史
        action_record = {
            'player_id': player_id,
            'role': player_role.original_role_name,
            'action': result['action'],
            'params': action_params,
            'result': result['result']
        }
        self.action_history.append(action_record)
        
        # 更新游戏状态（如角色交换等）
        self._update_state_after_action(player_id, result)
        
        # 移动到下一个玩家
        self.next_player()
        
        return result
    
    def _update_state_after_action(self, player_id: int, action_result: Dict[str, Any]) -> None:
        """
        根据行动结果更新游戏状态
        
        Args:
            player_id: 行动玩家ID
            action_result: 行动结果
        """
        action = action_result['action']
        
        # 根据不同行动类型更新状态
        if action == 'swap_role':
            target_id = action_result.get('target')
            if target_id is not None and 0 <= target_id < self.num_players:
                # 交换角色
                self.players[player_id]['current_role'], self.players[target_id]['current_role'] = \
                    self.players[target_id]['current_role'], self.players[player_id]['current_role']
                
                # 更新角色实例
                self.role_instances[player_id] = create_role(self.players[player_id]['current_role'], player_id)
                self.role_instances[target_id] = create_role(self.players[target_id]['current_role'], target_id)
                
        elif action == 'swap_roles':
            targets = action_result.get('targets', [])
            if len(targets) == 2 and action_result.get('result') == True:
                target_id1, target_id2 = targets
                # 交换两个玩家的角色
                self.players[target_id1]['current_role'], self.players[target_id2]['current_role'] = \
                    self.players[target_id2]['current_role'], self.players[target_id1]['current_role']
                
                # 更新角色实例
                self.role_instances[target_id1] = create_role(self.players[target_id1]['current_role'], target_id1)
                self.role_instances[target_id2] = create_role(self.players[target_id2]['current_role'], target_id2)
    
    def record_speech(self, player_id: int, speech_content: Dict[str, Any]) -> None:
        """
        记录玩家发言
        
        Args:
            player_id: 发言玩家ID
            speech_content: 发言内容
        """
        if self.phase != 'day' or player_id != self.current_player:
            return
        
        speech_record = {
            'player_id': player_id,
            'content': speech_content,
            'round': self.round
        }
        self.speech_history.append(speech_record)
        
        # 移动到下一个玩家
        self.next_player()
    
    def record_vote(self, voter_id: int, target_id: int) -> None:
        """
        记录玩家投票
        
        Args:
            voter_id: 投票者ID
            target_id: 投票目标ID
        """
        if self.phase != 'vote' or voter_id != self.current_player:
            return
        
        if 0 <= target_id < self.num_players:
            self.votes[voter_id] = target_id
        
        # 移动到下一个玩家
        self.next_player()
    
    def _determine_winner(self) -> str:
        """
        确定游戏胜利方
        
        Returns:
            胜利阵营: 'werewolf_team' 或 'villager_team'
        """
        if not self.votes:
            self.game_result = None
            return None
        
        # 计算得票数
        vote_counts = defaultdict(int)
        for target_id in self.votes.values():
            vote_counts[target_id] += 1
        
        # 找出得票最多的玩家
        max_votes = max(vote_counts.values()) if vote_counts else 0
        most_voted = [player_id for player_id, count in vote_counts.items() if count == max_votes]
        
        # 如果有平票，按照规则判定
        if len(most_voted) > 1:
            # 检查是否有狼人在平票中
            werewolves_in_tie = any(self.players[pid]['current_role'] == 'werewolf' for pid in most_voted)
            if werewolves_in_tie:
                # 狼人与好人平票，村民胜利
                self.game_result = 'villager_team'
                return self.game_result
        
        # 判断被投出的玩家是否是狼人
        voted_out = most_voted[0] if most_voted else -1
        if voted_out >= 0:
            voted_role = self.players[voted_out]['current_role']
            if voted_role == 'werewolf':
                # 投出狼人，村民胜利
                self.game_result = 'villager_team'
            else:
                # 投出非狼人，狼人胜利
                self.game_result = 'werewolf_team'
        
        return self.game_result
    
    def get_observation(self, player_id: int) -> Dict[str, Any]:
        """
        获取指定玩家的观察
        
        Args:
            player_id: 玩家ID
            
        Returns:
            玩家观察字典
        """
        if player_id < 0 or player_id >= self.num_players:
            return {}
        
        # 公共信息
        obs = {
            'player_id': player_id,
            'num_players': self.num_players,
            'phase': self.phase,
            'round': self.round,
            'current_player': self.current_player,
            'original_role': self.players[player_id]['original_role'],
            'known_info': self.players[player_id]['known_info'].copy(),
        }
        
        # 根据不同阶段提供不同信息
        if self.phase == 'night':
            # 夜晚只知道自己的角色和行动
            pass
        
        elif self.phase in ['day', 'vote', 'end']:
            # 白天可以看到发言历史
            obs['speech_history'] = self.speech_history.copy()
            
            if self.phase in ['vote', 'end']:
                # 投票阶段可以看到当前投票情况
                obs['votes'] = self.votes.copy()
                
                if self.phase == 'end':
                    # 游戏结束可以看到最终结果
                    obs['game_result'] = self.game_result
                    # 游戏结束时可以看到所有玩家的最终角色
                    obs['final_roles'] = {p['id']: p['current_role'] for p in self.players}
        
        return obs
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将游戏状态转换为字典表示
        
        Returns:
            状态字典
        """
        return {
            'phase': self.phase,
            'round': self.round,
            'current_player': self.current_player,
            'players': self.players,
            'center_cards': self.center_cards,
            'action_history': self.action_history,
            'speech_history': self.speech_history,
            'votes': self.votes,
            'game_result': self.game_result
        }


class PlayerObservation:
    """玩家观察类，表示玩家可观察到的游戏状态"""
    
    def __init__(self, player_id: int, game_state: GameState):
        """
        初始化玩家观察
        
        Args:
            player_id: 玩家ID
            game_state: 游戏状态
        """
        self.player_id = player_id
        self.observation = game_state.get_observation(player_id)
        
    def to_vector(self) -> np.ndarray:
        """
        将观察转换为向量表示，用于神经网络输入
        
        Returns:
            观察向量
        """
        # 这里需要根据实际项目需求进行实现
        # 可以使用one-hot编码、词嵌入等方法
        
        # 示例：简单的向量化
        phase_map = {phase: i for i, phase in enumerate(GameState.GAME_PHASES)}
        
        # 基本信息
        vec = [
            self.player_id,
            phase_map.get(self.observation.get('phase', 'init'), 0),
            self.observation.get('round', 0),
            self.observation.get('current_player', 0),
        ]
        
        # 角色信息 (one-hot)
        # 在实际实现中需要根据所有可能的角色进行编码
        
        return np.array(vec, dtype=np.float32)

def process_state(state):
    validate_state(state)
    # 处理状态的其他逻辑 