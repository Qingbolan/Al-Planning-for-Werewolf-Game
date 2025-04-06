"""
Werewolf Game Reinforcement Learning Training Script
"""
import argparse
import torch
import random
import numpy as np
import os

from train import Trainer
from config.default_config import DEFAULT_GAME_CONFIG


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Werewolf Game Reinforcement Learning Training')
    
    # Basic training parameters
    parser.add_argument('--num_episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--num_players', type=int, default=6, help='Number of players')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--evaluate_every', type=int, default=100, help='Evaluate every N episodes')
    parser.add_argument('--save_every', type=int, default=500, help='Save model every N episodes')
    parser.add_argument('--render_every', type=int, default=200, help='Render every N episodes')
    
    # Model parameters
    parser.add_argument('--obs_dim', type=int, default=128, help='Observation space dimension')
    parser.add_argument('--action_dim', type=int, default=100, help='Action space dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    
    # Optimizer parameters
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--value_coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Gradient clipping norm')
    
    # Environment parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_speech_rounds', type=int, default=3, help='Number of speech rounds')
    parser.add_argument('--reverse_vote_rules', action='store_true', default=True, help='Whether to use reversed voting rules')
    
    # Path parameters
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--save_dir', type=str, default='./models/saved', help='Model save directory')
    parser.add_argument('--visualize_dir', type=str, default='./visualizations', help='Visualization save directory')
    
    # Other parameters
    parser.add_argument('--use_cuda', action='store_true', help='Whether to use CUDA')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb for experiment logging')
    parser.add_argument('--continue_training', action='store_true', help='Whether to continue previous training')
    parser.add_argument('--model_path', type=str, default=None, help='Model path to load when continuing training')
    parser.add_argument('--agent_types', type=str, nargs='+', default=['random'], 
                       help='List of agent types to use, options: random, heuristic, rl')
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.visualize_dir, exist_ok=True)
    
    # Modify game configuration to support new game rules
    game_config = DEFAULT_GAME_CONFIG.copy()
    game_config.update({
        'num_players': args.num_players,
        'max_speech_rounds': args.max_speech_rounds,
        'reverse_vote_rules': args.reverse_vote_rules
    })
    
    # Create trainer
    trainer = Trainer(
        env_config=game_config,
        num_players=args.num_players,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        visualize_dir=args.visualize_dir
    )
    
    # Start training or evaluation
    if args.num_episodes > 0:
        print("Starting training...")
        agents = trainer.train_self_play(
            initial_agent_types=args.agent_types,
            num_generations=args.num_episodes // args.evaluate_every,
            episodes_per_generation=args.evaluate_every,
            render=(args.render_every > 0),
            visualize=args.use_wandb
        )
    else:
        # Evaluation only
        print("Starting evaluation...")
        eval_result = trainer.evaluate(
            agent_types=args.agent_types,
            num_episodes=100,
            render=True,
            visualize=args.use_wandb
        )
        print("\nEvaluation results:")
        print(f"Werewolf win rate: {eval_result['werewolf_win_rate']:.2f}")
        print(f"Villager win rate: {eval_result['villager_win_rate']:.2f}")
        print(f"Average game length: {eval_result['avg_game_length']:.2f} rounds")
    
    print("\nTraining/Evaluation completed!")


if __name__ == "__main__":
    main() 