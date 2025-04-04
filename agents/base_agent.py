"""
狼人杀游戏智能体基类
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import random
from abc import ABC, abstractmethod

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
    """智能体基类"""
    
    def __init__(self, player_id: int):
        """
        初始化智能体
        
        Args:
            player_id: 玩家ID
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
        初始化智能体状态
        
        Args:
            game_state: 游戏状态
        """
        self.game_state = game_state
        self.original_role = game_state.players[self.player_id]['original_role']
        self.current_role = game_state.players[self.player_id]['current_role']
        self.current_phase = game_state.phase
        
        # 创建角色特定的信念更新器
        self.belief_updater = create_belief_updater(self.player_id, game_state)
    
    def observe(self, observation: Dict[str, Any]) -> None:
        """
        接收观察信息
        
        Args:
            observation: 观察信息
        """
        # 更新当前角色和阶段
        self.current_role = observation.get('original_role', self.current_role)
        self.current_phase = observation.get('phase', self.current_phase)
        
        # 处理夜晚行动结果
        if 'action_result' in observation:
            self.belief_updater.update_with_night_action(observation['action_result'])
        
        # 处理发言历史
        if 'speech_history' in observation:
            for speech in observation['speech_history']:
                if speech not in self.action_history:  # 避免重复处理
                    self.action_history.append(speech)
                    self.belief_updater.update_with_speech(speech['player_id'], speech['content'])
        
        # 处理投票信息
        if 'votes' in observation:
            self.belief_updater.update_with_votes(observation['votes'])
    
    def act(self, observation: Dict[str, Any]) -> Action:
        """
        根据观察选择行动
        
        Args:
            observation: 观察信息
            
        Returns:
            行动
        """
        # 更新观察
        self.observe(observation)
        
        # 获取当前阶段
        phase = observation.get('phase', self.current_phase)
        
        # 根据不同阶段选择行动
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
        选择夜晚行动
        
        Args:
            observation: 观察信息
            
        Returns:
            夜晚行动
        """
        pass
    
    @abstractmethod
    def _day_action(self, observation: Dict[str, Any]) -> Action:
        """
        选择白天发言
        
        Args:
            observation: 观察信息
            
        Returns:
            白天发言
        """
        pass
    
    @abstractmethod
    def _vote_action(self, observation: Dict[str, Any]) -> Action:
        """
        选择投票目标
        
        Args:
            observation: 观察信息
            
        Returns:
            投票行动
        """
        pass
    
    def get_role_probabilities(self, player_id: int) -> Dict[str, float]:
        """
        获取某玩家的角色概率分布
        
        Args:
            player_id: 玩家ID
            
        Returns:
            角色概率分布
        """
        if self.belief_updater and player_id in self.belief_updater.belief_state.beliefs:
            return dict(self.belief_updater.belief_state.beliefs[player_id])
        return {}
    
    def get_most_suspected_werewolf(self) -> Tuple[int, float]:
        """
        获取最可能是狼人的玩家
        
        Returns:
            (玩家ID, 概率)
        """
        if not self.belief_updater:
            return -1, 0.0
        
        max_prob = -1
        max_player = -1
        
        for player_id in self.belief_updater.belief_state.beliefs:
            if player_id == self.player_id:
                continue  # 跳过自己
                
            werewolf_prob = self.belief_updater.belief_state.beliefs[player_id].get('werewolf', 0.0)
            if werewolf_prob > max_prob:
                max_prob = werewolf_prob
                max_player = player_id
        
        return max_player, max_prob
    
    def get_random_player_except_self(self) -> int:
        """
        随机选择一个非自己的玩家
        
        Returns:
            玩家ID
        """
        if not self.game_state:
            return random.randint(0, 5)  # 默认随机选择
            
        num_players = len(self.game_state.players)
        other_players = [i for i in range(num_players) if i != self.player_id]
        
        if other_players:
            return random.choice(other_players)
        return -1  # 如果只有自己，返回-1
    
    def get_action_probabilities(self, action_space: List[Any]) -> np.ndarray:
        """
        计算行动概率分布
        
        Args:
            action_space: 可用行动空间
            
        Returns:
            行动概率分布
        """
        if self.belief_updater:
            action_probs = self.belief_updater.get_action_probabilities(action_space)
            return np.array([action_probs.get(action, 0.0) for action in action_space])
        
        # 默认均匀分布
        return np.ones(len(action_space)) / len(action_space)

    def log_action(self, action: Action):
        # 公共的记录行为逻辑
        print(f"Agent {self.player_id} executes action {action}")


class RandomAgent(BaseAgent):
    """随机行动智能体"""
    
    def _night_action(self, observation: Dict[str, Any]) -> Action:
        """随机夜晚行动"""
        role = self.current_role
        
        # 获取角色可用的夜晚行动
        if role == 'werewolf':
            actions = ['check_other_werewolves', 'check_center_card']
            action_name = random.choice(actions)
            
            if action_name == 'check_center_card':
                card_index = random.randint(0, 2)  # 随机选择一张中央牌
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
                card_indices = random.sample(range(3), 2)  # 随机选择两张中央牌
                action = create_night_action(self.player_id, role, action_name, card_indices=card_indices)
                
        elif role == 'robber':
            target_id = self.get_random_player_except_self()
            action = create_night_action(self.player_id, role, 'swap_role', target_id=target_id)
            
        elif role == 'troublemaker':
            # 随机选择两个不同的玩家交换角色
            players = [i for i in range(len(self.game_state.players)) if i != self.player_id]
            if len(players) >= 2:
                target_id1, target_id2 = random.sample(players, 2)
                action = create_night_action(self.player_id, role, 'swap_roles', 
                                          target_id1=target_id1, target_id2=target_id2)
            
        elif role == 'minion':
            action = create_night_action(self.player_id, role, 'check_werewolves')
            
        elif role == 'insomniac':
            action = create_night_action(self.player_id, role, 'check_final_role')
            
        # 对于没有夜晚行动的角色，或者其他情况
        else:
            action = create_no_action(self.player_id)
        
        self.log_action(action)
        return action
    
    def _day_action(self, observation: Dict[str, Any]) -> Action:
        """随机白天发言"""
        # 随机选择发言类型
        speech_types = [t.name for t in SpeechType]
        speech_type = random.choice(speech_types)
        
        if speech_type == SpeechType.CLAIM_ROLE.name:
            # 声称自己的角色（可能是真实的，也可能是伪装的）
            possible_roles = ['villager', 'seer', 'robber', 'troublemaker', 'insomniac']
            role = random.choice(possible_roles)
            action = create_speech(self.player_id, speech_type, role=role)
            
        elif speech_type == SpeechType.CLAIM_ACTION_RESULT.name:
            # 声称行动结果
            claimed_role = self.current_role  # 通常声称自己的真实角色
            action = "查看" if claimed_role == 'seer' else "交换" if claimed_role in ['robber', 'troublemaker'] else "检查"
            target = f"玩家{random.randint(0, len(self.game_state.players)-1)}"
            result = random.choice(['villager', 'werewolf', 'seer'])
            
            action = create_speech(self.player_id, speech_type, 
                                role=claimed_role, action=action, target=target, result=result)
                                
        elif speech_type == SpeechType.ACCUSE.name:
            # 指控某人是狼人
            target_id = self.get_random_player_except_self()
            action = create_speech(self.player_id, speech_type, 
                                target_id=target_id, accused_role='werewolf')
                                
        elif speech_type == SpeechType.DEFEND.name:
            # 辩解自己不是狼人
            action = create_speech(self.player_id, speech_type, 
                                not_role='werewolf', reason="我是好人")
                                
        elif speech_type == SpeechType.VOTE_INTENTION.name:
            # 声明投票意向
            target_id = self.get_random_player_except_self()
            action = create_speech(self.player_id, speech_type, target_id=target_id)
        
        # 默认声称自己是村民
        else:
            action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role='villager')
        
        self.log_action(action)
        return action
    
    def _vote_action(self, observation: Dict[str, Any]) -> Action:
        """随机投票"""
        target_id = self.get_random_player_except_self()
        action = create_vote(self.player_id, target_id)
        self.log_action(action)
        return action


class HeuristicAgent(BaseAgent):
    """基于启发式规则的智能体"""
    
    def _night_action(self, observation: Dict[str, Any]) -> Action:
        """基于启发式规则的夜晚行动"""
        role = self.current_role
        
        if role == 'werewolf':
            # 如果知道有其他狼人，选择查看其他狼人
            # 如果是唯一的狼人，查看中央牌堆
            other_werewolves_exist = False
            for player in self.game_state.players:
                if player['id'] != self.player_id and player['original_role'] == 'werewolf':
                    other_werewolves_exist = True
                    break
            
            if other_werewolves_exist:
                action = create_night_action(self.player_id, role, 'check_other_werewolves')
            else:
                # 如果是唯一的狼人，随机查看一张中央牌
                card_index = random.randint(0, 2)
                action = create_night_action(self.player_id, role, 'check_center_card', card_index=card_index)
        
        elif role == 'seer':
            # 优先查验可疑玩家，如果没有可疑玩家，则随机查验或查看中央牌堆
            if random.random() < 0.7:  # 70%的概率查验玩家
                # 尝试找一个可疑的玩家
                suspected_player, prob = self.get_most_suspected_werewolf()
                if suspected_player >= 0 and prob > 0.3:
                    target_id = suspected_player
                else:
                    # 随机选择一个非自己的玩家
                    target_id = self.get_random_player_except_self()
                
                action = create_night_action(self.player_id, role, 'check_player', target_id=target_id)
            else:
                # 查看中央牌堆
                card_indices = random.sample(range(3), 2)
                action = create_night_action(self.player_id, role, 'check_center_cards', card_indices=card_indices)
        
        elif role == 'robber':
            # 尝试找一个看起来不是狼人的玩家偷取
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
                # 如果没有找到好的目标，随机选择
                target_id = self.get_random_player_except_self()
                action = create_night_action(self.player_id, role, 'swap_role', target_id=target_id)
        
        elif role == 'troublemaker':
            # 尝试交换两个玩家，优先考虑可能是狼人的玩家
            players = []
            for player_id in range(len(self.game_state.players)):
                if player_id == self.player_id:
                    continue
                
                werewolf_prob = self.get_role_probabilities(player_id).get('werewolf', 0.5)
                players.append((player_id, werewolf_prob))
            
            # 按狼人概率排序
            players.sort(key=lambda x: x[1], reverse=True)
            
            if len(players) >= 2:
                target_id1 = players[0][0]  # 最可能是狼人的玩家
                target_id2 = players[-1][0]  # 最不可能是狼人的玩家
                
                action = create_night_action(self.player_id, role, 'swap_roles', 
                                          target_id1=target_id1, target_id2=target_id2)
            else:
                action = create_no_action(self.player_id)
        
        elif role == 'minion':
            # 爪牙只能查看狼人
            action = create_night_action(self.player_id, role, 'check_werewolves')
        
        elif role == 'insomniac':
            # 失眠者只能查看自己的最终角色
            action = create_night_action(self.player_id, role, 'check_final_role')
        
        # 对于没有夜晚行动的角色，或者其他情况
        else:
            action = create_no_action(self.player_id)
        
        self.log_action(action)
        return action
    
    def _day_action(self, observation: Dict[str, Any]) -> Action:
        """基于启发式规则的白天发言"""
        role = self.current_role
        
        # 不同角色有不同的发言策略
        if role == 'werewolf':
            # 狼人可能会伪装成村民或特殊角色
            if random.random() < 0.6:  # 60%的概率伪装成村民
                action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role='villager')
            else:
                # 40%的概率伪装成特殊角色（通常不会声称自己是预言家，因为容易被反驳）
                special_roles = ['robber', 'troublemaker', 'insomniac']
                fake_role = random.choice(special_roles)
                
                if fake_role == 'robber':
                    # 编造一个偷取故事
                    target_id = self.get_random_player_except_self()
                    fake_stolen_role = random.choice(['villager', 'troublemaker', 'insomniac'])
                    
                    action = create_speech(self.player_id, SpeechType.CLAIM_ACTION_RESULT.name,
                                        role='robber', action='偷取', target=f"玩家{target_id}",
                                        result=fake_stolen_role)
                else:
                    # 简单声称角色
                    action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role=fake_role)
        
        elif role == 'seer':
            # 预言家通常会声称自己是预言家，并分享查验结果
            # 查看是否有夜晚行动结果
            action_results = [action for action in self.action_history 
                             if action.get('player_id') == self.player_id 
                             and action.get('action') == 'check_player']
            
            if action_results:
                # 有查验结果，分享
                action_result = action_results[-1]  # 最新的查验结果
                target_id = action_result.get('target')
                result = action_result.get('result')
                
                if target_id is not None and result:
                    action = create_speech(self.player_id, SpeechType.CLAIM_ACTION_RESULT.name,
                                        role='seer', action='查验', target=f"玩家{target_id}",
                                        result=result)
            
            # 如果没有查验结果或查验了中央牌堆，简单声称自己是预言家
            else:
                action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role='seer')
        
        elif role in ['robber', 'troublemaker', 'insomniac']:
            # 特殊角色通常会声称自己的角色和行动结果
            action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role=role)
        
        elif role == 'minion':
            # 爪牙需要保护狼人，通常伪装成村民
            action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role='villager')
        
        else:  # 村民
            # 村民通常声称自己是村民，或者指控可疑玩家
            if random.random() < 0.7:  # 70%的概率声称自己是村民
                action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role='villager')
            else:
                # 30%的概率指控可疑玩家
                suspected_player, prob = self.get_most_suspected_werewolf()
                if suspected_player >= 0 and prob > 0.3:
                    action = create_speech(self.player_id, SpeechType.ACCUSE.name,
                                        target_id=suspected_player, accused_role='werewolf')
                else:
                    # 如果没有足够可疑的玩家，声称自己是村民
                    action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role='villager')
        
        self.log_action(action)
        return action
    
    def _vote_action(self, observation: Dict[str, Any]) -> Action:
        """基于启发式规则的投票"""
        # 根据身份和信念状态决定投票
        role = self.current_role
        
        if role in ['werewolf', 'minion']:
            # 狼人阵营尝试投给看起来最像预言家的玩家
            best_target = -1
            highest_seer_prob = -1
            
            for player_id in range(len(self.game_state.players)):
                if player_id == self.player_id:
                    continue
                
                # 检查该玩家是否已知是狼人（对于狼人来说）
                if role == 'werewolf' and player_id in self.belief_updater.belief_state.certain_roles:
                    if self.belief_updater.belief_state.certain_roles[player_id] == 'werewolf':
                        continue  # 跳过队友
                
                seer_prob = self.get_role_probabilities(player_id).get('seer', 0.0)
                if seer_prob > highest_seer_prob:
                    highest_seer_prob = seer_prob
                    best_target = player_id
            
            if best_target >= 0 and highest_seer_prob > 0.3:
                action = create_vote(self.player_id, best_target)
            else:
                action = create_no_action(self.player_id)
        
        # 对于村民阵营或没有找到特定目标的狼人阵营
        else:
            suspected_player, prob = self.get_most_suspected_werewolf()
            if suspected_player >= 0 and prob > 0.3:
                action = create_vote(self.player_id, suspected_player)
            else:
                # 如果没有明确目标，随机投票
                target_id = self.get_random_player_except_self()
                action = create_vote(self.player_id, target_id)
        
        self.log_action(action)
        return action
        

# 工厂函数，创建指定类型的智能体
def create_agent(agent_type: str, player_id: int) -> BaseAgent:
    """
    创建指定类型的智能体
    
    Args:
        agent_type: 智能体类型
        player_id: 玩家ID
        
    Returns:
        智能体实例
    """
    if agent_type == 'random':
        return RandomAgent(player_id)
    elif agent_type == 'heuristic':
        return HeuristicAgent(player_id)
    else:
        # 默认返回随机智能体
        return RandomAgent(player_id) 