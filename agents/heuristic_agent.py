import random

class HeuristicAgent:
    def __init__(self, player_id, belief_updater):
        self.player_id = player_id
        self.belief_updater = belief_updater

    def _choose_vote_target(self, game_state: GameState) -> int:
        """Choose who to vote for based on beliefs and strategy"""
        my_role = game_state.players[self.player_id]['role']
        
        # Initialize suspicion_scores with base suspicion from belief updater
        suspicion_scores = {}
        for player_id, beliefs in self.belief_updater.belief_state.beliefs.items():
            if player_id != self.player_id and player_id < len(game_state.players):
                # Base suspicion from belief probabilities
                wolf_belief = beliefs.get('werewolf', 0)
                wolf_belief_adjusted = wolf_belief
                
                # Calculate suspicion based on behavioral analysis
                player_consistency = self.belief_updater.player_consistency_score.get(player_id, 0.5)
                vote_suspicion = self.belief_updater.vote_suspicion_score.get(player_id, 0)
                speech_suspicion = self.belief_updater.speech_suspicion_score.get(player_id, 0)
                
                # Different behavior for werewolf vs village roles
                if my_role == 'werewolf':
                    # Werewolves try to mislead by voting for non-werewolves to appear genuine
                    # Prioritize voting for strongest village players
                    if 'werewolf' in beliefs and player_id not in self.belief_updater.belief_state.certain_roles and beliefs['werewolf'] < 0.3:
                        # More likely to vote for a non-werewolf
                        # Reduced efficiency of werewolf misleading strategy
                        suspicion_scores[player_id] = (1.0 - wolf_belief_adjusted) * 0.7 + player_consistency * 0.1 + vote_suspicion * 0.1 + speech_suspicion * 0.1
                    else:
                        # Less likely to vote for another werewolf
                        suspicion_scores[player_id] = 0.1 + player_consistency * 0.1 + vote_suspicion * 0.1
                else:
                    # Village roles try to identify werewolves
                    # Enhanced village voting strategy 
                    if 'werewolf' in beliefs:
                        # Boost werewolf suspicion to help village team
                        wolf_belief_adjusted = wolf_belief * 1.5
                        
                    # Weight behavioral factors more heavily for village team
                    behavior_weight = 0.6  
                    belief_weight = 0.4
                    
                    suspicion_scores[player_id] = (
                        wolf_belief_adjusted * belief_weight +  # Base suspicion from beliefs
                        (1.0 - player_consistency) * 0.3 * behavior_weight +  # Inconsistency factor
                        vote_suspicion * 0.2 * behavior_weight +  # Suspicious voting
                        speech_suspicion * 0.5 * behavior_weight   # Suspicious speech (weighted higher)
                    )
                    
                    # Extra penalty for suspicious claims
                    if player_id in self.belief_updater.player_role_claims:
                        claimed_role = self.belief_updater.player_role_claims[player_id]
                        if claimed_role in beliefs and beliefs[claimed_role] < 0.3:
                            # Claimed a role that seems unlikely - increase suspicion
                            suspicion_scores[player_id] += 0.3
        
        # Don't vote for myself or players already certain about
        for player_id in list(suspicion_scores.keys()):
            if player_id == self.player_id:
                suspicion_scores.pop(player_id)
            elif player_id in self.belief_updater.belief_state.certain_roles:
                # If I'm certain about their role
                certain_role = self.belief_updater.belief_state.certain_roles[player_id]
                if (my_role == 'werewolf' and certain_role == 'werewolf') or \
                   (my_role != 'werewolf' and certain_role != 'werewolf'):
                    # Don't vote for teammates
                    suspicion_scores.pop(player_id)
        
        # Random choice if no valid targets
        if not suspicion_scores:
            valid_targets = [p for p in range(len(game_state.players)) if p != self.player_id]
            return random.choice(valid_targets) if valid_targets else self.player_id
        
        # Choose the most suspicious player
        return max(suspicion_scores.items(), key=lambda x: x[1])[0] 