"""
Werewolf Game Agent Training Script
"""
import os
import numpy as np
import random
import time
from typing import List, Dict, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import json

from werewolf_env import WerewolfEnv
from agents import BaseAgent, RandomAgent, HeuristicAgent, create_agent
from config import DEFAULT_GAME_CONFIG, ROLE_TEAMS
from utils.visualizer import BeliefVisualizer


class Trainer:
    """Agent trainer"""
    
    def __init__(self, 
                 env_config: Dict[str, Any] = None, 
                 num_players: int = 6,
                 log_dir: str = './logs',
                 save_dir: str = './models',
                 visualize_dir: str = './visualizations'):
        """
        Initialize trainer
        
        Args:
            env_config: Environment configuration
            num_players: Number of players
            log_dir: Log directory
            save_dir: Model save directory
            visualize_dir: Visualization save directory
        """
        self.num_players = num_players
        self.env_config = env_config or DEFAULT_GAME_CONFIG
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.visualize_dir = visualize_dir
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(visualize_dir, exist_ok=True)
        
        # Create environment
        self.env = WerewolfEnv(self.env_config)
        
        # Statistics
        self.stats = {
            'werewolf_wins': 0,
            'villager_wins': 0,
            'total_games': 0,
            'game_lengths': [],
            'rewards': defaultdict(list)
        }
    
    def create_agents(self, agent_types: List[str]) -> List[BaseAgent]:
        """
        Create agents
        
        Args:
            agent_types: List of agent types
            
        Returns:
            List of agents
        """
        agents = []
        for i in range(self.num_players):
            agent_type = agent_types[i % len(agent_types)]
            agents.append(create_agent(agent_type, i))
        return agents
    
    def run_episode(self, agents: List[BaseAgent], render: bool = False, visualize: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run a single game
        
        Args:
            agents: List of agents
            render: Whether to render
            visualize: Whether to visualize belief states
            
        Returns:
            (Game result, Action history)
        """
        # Reset environment
        obs, _ = self.env.reset()
        
        # Initialize agents
        for i, agent in enumerate(agents):
            agent.initialize(self.env.game_state)
        
        done = False
        action_history = []
        total_rewards = defaultdict(float)
        
        # Main game loop
        while not done:
            # Render
            if render:
                self.env.render()
            
            # Visualize belief states
            if visualize and hasattr(agents[0], 'belief_updater') and agents[0].belief_updater:
                believer_id = 0
                belief_visualizer = BeliefVisualizer()
                timestamp = int(time.time())
                belief_visualizer.generate_belief_report(
                    agents[believer_id].belief_updater.belief_state,
                    self.env.game_state,
                    believer_id,
                    f"{self.visualize_dir}/belief_{timestamp}.png"
                )
            
            # Get current player
            current_player_id = self.env.current_player_id
            
            if current_player_id < 0:
                # Phase end or invalid state
                break
            
            # Get current player's agent
            agent = agents[current_player_id]
            
            # Agent decision
            action = agent.act(obs)
            
            # Record action
            action_info = {
                'player_id': current_player_id,
                'phase': self.env.game_state.phase,
                'speech_round': self.env.game_state.speech_round,  # Record speech round
                'action': action
            }
            action_history.append(action_info)
            
            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Accumulate rewards
            total_rewards[current_player_id] += reward
            
            # Check if game is over
            done = terminated or truncated
        
        # Game result
        result = {
            'winner': self.env.game_state.game_result,
            'game_length': len(action_history),
            'total_rewards': dict(total_rewards),
            'final_roles': {i: player['current_role'] for i, player in enumerate(self.env.game_state.players)},
            'votes': self.env.game_state.votes.copy() if hasattr(self.env.game_state, 'votes') else {}
        }
        
        # Update statistics
        self.stats['total_games'] += 1
        self.stats['game_lengths'].append(len(action_history))
        
        if self.env.game_state.game_result == 'werewolf':
            self.stats['werewolf_wins'] += 1
        elif self.env.game_state.game_result == 'villager':
            self.stats['villager_wins'] += 1
            
        for player_id, reward in total_rewards.items():
            self.stats['rewards'][player_id].append(reward)
        
        if render:
            print(f"Game over! Winner: {result['winner']}")
            print(f"Total rewards: {result['total_rewards']}")
            print(f"Voting results: {result['votes']}")
        
        return result, action_history
    
    def evaluate(self, 
                agent_types: List[str], 
                num_episodes: int = 100, 
                render: bool = False,
                visualize: bool = False) -> Dict[str, Any]:
        """
        Evaluate agents
        
        Args:
            agent_types: List of agent types
            num_episodes: Number of evaluation episodes
            render: Whether to render
            visualize: Whether to visualize
            
        Returns:
            Evaluation results
        """
        results = []
        werewolf_wins = 0
        villager_wins = 0
        
        # Run multiple games
        for episode in range(num_episodes):
            agents = self.create_agents(agent_types)
            result, _ = self.run_episode(agents, render=(render and episode < 5), visualize=visualize)
            results.append(result)
            
            # Count win rates
            if result['winner'] == 'werewolf':
                werewolf_wins += 1
            elif result['winner'] == 'villager':
                villager_wins += 1
        
        # Calculate win rates
        werewolf_win_rate = werewolf_wins / num_episodes
        villager_win_rate = villager_wins / num_episodes
        
        # Calculate average rewards
        avg_rewards = {}
        for player_id in range(self.num_players):
            rewards = [r['total_rewards'].get(player_id, 0) for r in results]
            avg_rewards[player_id] = sum(rewards) / len(rewards)
        
        # Calculate average game length
        avg_game_length = sum(r['game_length'] for r in results) / len(results)
        
        # Evaluation results
        eval_result = {
            'num_episodes': num_episodes,
            'werewolf_win_rate': werewolf_win_rate,
            'villager_win_rate': villager_win_rate,
            'avg_rewards': avg_rewards,
            'avg_game_length': avg_game_length
        }
        
        print(f"Evaluation results ({num_episodes} games):")
        print(f"Werewolf win rate: {werewolf_win_rate:.2f}")
        print(f"Villager win rate: {villager_win_rate:.2f}")
        print(f"Average game length: {avg_game_length:.2f} rounds")
        
        return eval_result
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """
        Save evaluation results
        
        Args:
            results: Evaluation results
            filename: Filename
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save as JSON
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        # If enough data, create win rate over time plot
        if 'win_rates' in results and len(results['win_rates']) > 1:
            werewolf_rates = [r[0] for r in results['win_rates']]
            villager_rates = [r[1] for r in results['win_rates']]
            
            plt.figure(figsize=(10, 6))
            plt.plot(werewolf_rates, label='Werewolf Win Rate')
            plt.plot(villager_rates, label='Villager Win Rate')
            plt.xlabel('Generation')
            plt.ylabel('Win Rate')
            plt.title('Win Rates Over Generations')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{os.path.splitext(filename)[0]}_winrates.png")
            plt.close()
    
    def train_self_play(self, 
                   initial_agent_types: List[str], 
                   num_generations: int = 10,
                   episodes_per_generation: int = 100,
                   render: bool = False,
                   visualize: bool = False):
        """
        Self-play training
        
        Args:
            initial_agent_types: Initial agent types
            num_generations: Number of training generations
            episodes_per_generation: Episodes per generation
            render: Whether to render
            visualize: Whether to visualize
        """
        # Initialize agents
        agent_types = initial_agent_types.copy()
        
        win_rates = []
        
        # Start training
        for generation in range(num_generations):
            print(f"Generation {generation+1}/{num_generations}")
            
            # Create agents
            agents = self.create_agents(agent_types)
            
            # Self-play
            for episode in range(episodes_per_generation):
                print(f"Episode {episode+1}/{episodes_per_generation}", end='\r')
                self.run_episode(agents, render=(render and episode < 3), visualize=visualize)
                
                # Update agents (e.g., neural network training)
                for agent in agents:
                    if hasattr(agent, 'update') and callable(agent.update):
                        agent.update()
            
            # Evaluate results
            eval_result = self.evaluate(agent_types, num_episodes=50)
            win_rates.append((eval_result['werewolf_win_rate'], eval_result['villager_win_rate']))
            
            # Save models and results
            for i, agent in enumerate(agents):
                if hasattr(agent, 'save') and callable(agent.save):
                    agent.save(f"{self.save_dir}/agent_{i}_gen_{generation}.pt")
            
            # Save evaluation results
            results = {
                'generations': generation + 1,
                'win_rates': win_rates,
                'final_eval': eval_result
            }
            self.save_results(results, f"{self.log_dir}/training_results.json")
            
            print(f"Completed Generation {generation+1}, Werewolf win rate: {eval_result['werewolf_win_rate']:.2f}, Villager win rate: {eval_result['villager_win_rate']:.2f}")
            
        return agents


def main():
    """Main function"""
    trainer = Trainer()
    
    # Create agents
    agents = trainer.create_agents(['random'] * 6)
    
    # Run a single game
    result, _ = trainer.run_episode(agents, render=True)
    
    print("Game result:", result)
    

if __name__ == "__main__":
    main() 