"""
基于启发式规则的狼人杀智能体
"""
from typing import Dict, List, Any, Tuple
import random
from collections import defaultdict

from werewolf_env.actions import (
    Action, 
    create_night_action, create_speech, create_vote, create_no_action,
    SpeechType
)
from agents.base_agent import BaseAgent


class HeuristicAgent(BaseAgent):
    """Rule-based heuristic agent"""
    
    def __init__(self, player_id: int):
        """
        Initialize the heuristic agent
        
        Args:
            player_id: Player ID
        """
        super().__init__(player_id)
        self.claimed_role = None  # 记录已声明的角色
        self.claimed_actions = []  # 记录已声明的行动

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
        speech_round = observation.get('speech_round', 0)
        
        # 如果已经有声明过的角色，保持一致性
        if self.claimed_role is not None:
            if speech_round > 0 and random.random() < 0.7:  # 在后续轮次中有70%概率重申角色
                # 重申已声明的角色
                return create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role=self.claimed_role)
        
        # 不同角色有不同的发言策略
        if role == 'werewolf':
            # 狼人可能假装成村民或特殊角色
            if self.claimed_role is None:  # 第一次声明角色
                if random.random() < 0.6:  # 60%概率假装成村民
                    self.claimed_role = 'villager'
                    action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role='villager')
                else:
                    # 40%概率假装成特殊角色（通常不会声称是预言家，因为容易被反驳）
                    special_roles = ['robber', 'troublemaker', 'insomniac']
                    self.claimed_role = random.choice(special_roles)
                    
                    if self.claimed_role == 'robber':
                        # 编造一个偷窃故事
                        target_id = self.get_random_player_except_self()
                        fake_stolen_role = random.choice(['villager', 'troublemaker', 'insomniac'])
                        
                        action = create_speech(self.player_id, SpeechType.CLAIM_ACTION_RESULT.name,
                                            role='robber', action='steal', target=f"player {target_id}",
                                            result=fake_stolen_role)
                        self.claimed_actions.append({"target": target_id, "result": fake_stolen_role})
                    else:
                        # 简单声明角色
                        action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role=self.claimed_role)
            else:
                # 已经声明过角色，采取一致的后续行动
                if self.claimed_role == 'robber' and random.random() < 0.5 and not self.claimed_actions:
                    # 编造一个偷窃故事
                    target_id = self.get_random_player_except_self()
                    fake_stolen_role = random.choice(['villager', 'troublemaker', 'insomniac'])
                    
                    action = create_speech(self.player_id, SpeechType.CLAIM_ACTION_RESULT.name,
                                        role='robber', action='steal', target=f"player {target_id}",
                                        result=fake_stolen_role)
                    self.claimed_actions.append({"target": target_id, "result": fake_stolen_role})
                elif speech_round > 1 and random.random() < 0.3:
                    # 在后期轮次有30%概率尝试指控其他玩家
                    suspected_player = self.get_random_player_except_self()
                    action = create_speech(self.player_id, SpeechType.ACCUSE.name,
                                        target_id=suspected_player, accused_role='werewolf')
                else:
                    # 重申自己的角色
                    action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role=self.claimed_role)
        
        elif role == 'seer':
            # 预言家通常声称是预言家并分享检查结果
            if self.claimed_role is None:
                self.claimed_role = 'seer'
            
            # 检查夜间行动结果
            action_results = [action for action in self.action_history 
                             if action.get('player_id') == self.player_id 
                             and action.get('action') == 'check_player']
            
            if action_results and random.random() < 0.7:
                # 有检查结果，分享它们
                action_result = action_results[-1]  # 最新的检查结果
                target_id = action_result.get('target')
                result = action_result.get('result')
                
                if target_id is not None and result:
                    action = create_speech(self.player_id, SpeechType.CLAIM_ACTION_RESULT.name,
                                        role='seer', action='check', target=f"player {target_id}",
                                        result=result)
                    self.claimed_actions.append({"target": target_id, "result": result})
                else:
                    action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role=self.claimed_role)
            else:
                # 如果没有检查结果或检查了中央牌，只需声明自己是预言家
                action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role=self.claimed_role)
        
        elif role in ['robber', 'troublemaker', 'insomniac']:
            # 特殊角色通常声明其角色和行动结果
            if self.claimed_role is None:
                self.claimed_role = role
            
            action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role=self.claimed_role)
        
        elif role == 'minion':
            # 爪牙需要保护狼人，通常假装是村民
            if self.claimed_role is None:
                self.claimed_role = 'villager'
            
            action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role=self.claimed_role)
        
        else:  # 村民
            # 村民通常声称是村民或指控可疑玩家
            if self.claimed_role is None:
                self.claimed_role = 'villager'
            
            if speech_round > 0 and random.random() < 0.3:  # 在后续轮次中有30%概率指控可疑玩家
                # 指控可疑玩家
                suspected_player, prob = self.get_most_suspected_werewolf()
                if suspected_player >= 0 and prob > 0.3:
                    action = create_speech(self.player_id, SpeechType.ACCUSE.name,
                                        target_id=suspected_player, accused_role='werewolf')
                else:
                    # 如果没有足够可疑的玩家，声称是村民
                    action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role=self.claimed_role)
            else:
                # 声称是村民
                action = create_speech(self.player_id, SpeechType.CLAIM_ROLE.name, role=self.claimed_role)
        
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
        
        # 获取动作
        action = self.get_action(game_state)
        
        # 为不同类型的动作生成理由
        reasoning = ""
        
        # 检查action类型
        if hasattr(action, 'action_type'):
            action_type = action.action_type
            
            if action_type == "NIGHT_ACTION":
                role = self.current_role
                action_name = action.action_name
                
                if role == 'werewolf':
                    if action_name == 'check_other_werewolves':
                        reasoning = "作为狼人，我想查看其他狼人以确定我的队友"
                    else:
                        reasoning = "作为狼人，我想查看中央牌以获取更多信息"
                elif role == 'seer':
                    if action_name == 'check_player':
                        reasoning = "作为预言家，我想检查可疑的玩家身份"
                    else:
                        reasoning = "作为预言家，我想查看中央牌中的角色"
                elif role == 'robber':
                    reasoning = "作为强盗，我想偷取其他玩家的角色"
                elif role == 'troublemaker':
                    reasoning = "作为捣蛋鬼，我想交换两名玩家的角色卡"
                elif role == 'insomniac':
                    reasoning = "作为失眠者，我想查看我最终的角色"
                else:
                    reasoning = "执行夜间动作"
                    
            elif action_type == "DAY_SPEECH":
                speech_type = action.content.get('type', '')
                
                if speech_type == 'CLAIM_ROLE':
                    role = action.content.get('role', '')
                    reasoning = f"声称自己是{role}角色"
                elif speech_type == 'CLAIM_ACTION_RESULT':
                    reasoning = "分享我在夜间执行动作的结果"
                elif speech_type == 'ACCUSE':
                    target_id = action.content.get('target_id', -1)
                    reasoning = f"指控玩家{target_id}是狼人"
                else:
                    reasoning = "发表演讲"
                    
            elif action_type == "VOTE":
                target_id = action.target_id
                reasoning = f"投票给玩家{target_id}，我认为他/她最有可能是狼人"
                
            else:
                reasoning = "执行动作"
        else:
            # 动作可能是字典格式
            action_type = action.get('action_type', '')
            
            if action_type == 'NIGHT_ACTION':
                role = self.current_role
                action_name = action.get('action_name', '')
                
                if role == 'werewolf':
                    if action_name == 'check_other_werewolves':
                        reasoning = "作为狼人，我想查看其他狼人以确定我的队友"
                    else:
                        reasoning = "作为狼人，我想查看中央牌以获取更多信息"
                # ... 其他角色的理由
                else:
                    reasoning = "执行夜间动作"
            elif action_type == 'DAY_SPEECH':
                speech_type = action.get('speech_type', '')
                
                if speech_type == 'CLAIM_ROLE':
                    reasoning = "声称自己的角色"
                else:
                    reasoning = "发表演讲"
            elif action_type == 'VOTE':
                target_id = action.get('target_id', -1)
                reasoning = f"投票给玩家{target_id}"
            else:
                reasoning = "执行动作"
        
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