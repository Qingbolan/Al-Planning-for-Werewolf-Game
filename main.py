"""
Werewolf Game Main Program
"""
import argparse
import os
import torch
import numpy as np
import random
import time
from typing import List, Dict, Any

from werewolf_env import WerewolfEnv
from agents import (
    BaseAgent, RandomAgent, HeuristicAgent, RLAgent,
    create_agent, create_rl_agent, HAS_RL_AGENT
)
from utils.visualizer import BeliefVisualizer
from config.default_config import DEFAULT_GAME_CONFIG


def set_seed(seed: int):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Werewolf Game')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='play',
                       choices=['play', 'train', 'evaluate'],
                       help='Run mode: play (play game), train (training), evaluate (evaluation)')
    
    # Game configuration
    parser.add_argument('--num_players', type=int, default=6,
                       help='Number of players')
    parser.add_argument('--render', action='store_true',
                       help='Whether to render the game')
    parser.add_argument('--max_speech_rounds', type=int, default=3, 
                       help='Number of speech rounds (3-round speech mechanism)')
    parser.add_argument('--reverse_vote_rules', action='store_true', default=True,
                       help='Whether to use reversed voting rules (villagers voting leads to werewolf victory, and vice versa)')
    
    # Agent configuration
    parser.add_argument('--agent_types', type=str, nargs='+', 
                       default=['random'],
                       choices=['random', 'heuristic', 'rl'],
                       help='Types of agents to use')
    
    # Training configuration
    parser.add_argument('--train_episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--num_generations', type=int, default=10,
                       help='Number of training generations')
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to load model')
    parser.add_argument('--save_model', type=str, default='./models/rl_agent.pt',
                       help='Path to save model')
    
    # Evaluation configuration
    parser.add_argument('--eval_episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--visualize', action='store_true',
                       help='Whether to visualize belief states')
    
    # Other configuration
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Run device (cpu or cuda)')
    
    return parser.parse_args()


def play_game(args):
    """Run a single game"""
    # Create configuration
    config = DEFAULT_GAME_CONFIG.copy()
    config.update({
        'num_players': args.num_players,
        'max_speech_rounds': args.max_speech_rounds,
        'reverse_vote_rules': args.reverse_vote_rules
    })
    
    print("Game configuration:", config)
    
    # Create environment
    env = WerewolfEnv(config, render_mode="ansi" if args.render else None)
    
    # Create agents
    agents = []
    for i in range(args.num_players):
        agent_type = args.agent_types[i % len(args.agent_types)]
        if agent_type == 'rl':
            # Check if RL agent module exists
            if not HAS_RL_AGENT:
                print(f"Warning: RL agent module not implemented, using random agent instead")
                agent = create_agent('random', i)
            else:
                agent = create_rl_agent(i, args.device)
                if args.load_model:
                    agent.load_model(args.load_model)
        else:
            agent = create_agent(agent_type, i)
        agents.append(agent)
        print(f"Created agent {i}: {agent_type}")
    
    # Reset environment
    obs, info = env.reset()
    print("\nGame started!")
    print("Initial information:", info)
    
    # Initialize agents
    for i, agent in enumerate(agents):
        agent.initialize(env.game_state)
    
    # Main game loop
    done = False
    action_history = []
    step_count = 0
    
    while not done and step_count < 1000:  # Add maximum step limit
        # Render
        if args.render:
            rendered = env.render()
            if rendered:
                print(rendered)
            time.sleep(0.5)  # Slow display
        
        # Visualize belief states
        if args.visualize and hasattr(agents[0], 'belief_updater') and agents[0].belief_updater:
            believer_id = 0
            belief_visualizer = BeliefVisualizer()
            timestamp = int(time.time())
            belief_visualizer.generate_belief_report(
                agents[believer_id].belief_updater.belief_state,
                env.game_state,
                believer_id,
                f"./visualizations/belief_{timestamp}.png"
            )
        
        # Get current player
        current_player_id = env.current_player_id
        
        if current_player_id < 0:
            print("Warning: Invalid current player ID")
            break
        
        # Get current player's agent
        agent = agents[current_player_id]
        
        # Agent decision
        action = agent.act(obs)
        
        # Record action
        action_info = {
            'player_id': current_player_id,
            'phase': env.game_state.phase,
            'speech_round': env.game_state.speech_round if env.game_state.phase == 'day' else 0,
            'action': action
        }
        action_history.append(action_info)
        print(f"\nStep {step_count}:")
        print(f"Current player: {current_player_id}")
        print(f"Game phase: {env.game_state.phase}")
        print(f"Action: {action}")
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward}")
        print(f"Info: {info}")
        
        # Check if game is over
        done = terminated or truncated
        step_count += 1
    
    # Game over
    if args.render:
        rendered = env.render()
        if rendered:
            print(rendered)
        print(f"\nGame over! Total steps: {step_count}")
        print(f"Winner: {env.game_state.game_result}")
        print(f"Voting results: {env.game_state.votes}")
    
    return {
        'winner': env.game_state.game_result,
        'num_rounds': env.game_state.round,
        'speech_rounds': env.game_state.speech_round if hasattr(env.game_state, 'speech_round') else 0,
        'votes': env.game_state.votes if hasattr(env.game_state, 'votes') else {},
        'final_roles': {i: player['current_role'] for i, player in enumerate(env.game_state.players)},
        'total_steps': step_count
    }


def main():
    """Main function"""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Ensure directories exist
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./visualizations', exist_ok=True)
    
    # Create game configuration
    game_config = DEFAULT_GAME_CONFIG.copy()
    game_config.update({
        'num_players': args.num_players,
        'max_speech_rounds': args.max_speech_rounds,
        'reverse_vote_rules': args.reverse_vote_rules
    })
    
    # Check if RL agent exists and give warning if needed
    if 'rl' in args.agent_types and not HAS_RL_AGENT:
        print("Warning: RL agent module not implemented, will use random agent instead")
        args.agent_types = ['random' if t == 'rl' else t for t in args.agent_types]
    
    if args.mode == 'play':
        # Run single game
        result = play_game(args)
        print(f"Game result: {result}")
        
    elif args.mode == 'train':
        print("Warning: Training functionality not implemented")
        print("Please use run_training.py for training")
        
    elif args.mode == 'evaluate':
        print("Warning: Evaluation functionality not implemented")
        print("Please use run_training.py for evaluation")


if __name__ == "__main__":
    main() 