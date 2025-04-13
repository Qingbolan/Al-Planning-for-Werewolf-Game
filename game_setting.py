"""
Game settings configuration module
"""
from typing import Dict, List, Any, Optional
import os
from dataclasses import dataclass
from enum import Enum

class AgentType(Enum):
    RANDOM = "random"
    HEURISTIC = "heuristic"
    RL = "rl"

@dataclass
class AgentConfig:
    """Configuration for a single agent"""
    agent_type: AgentType
    model_path: Optional[str] = None
    is_training: bool = False
    device: str = "cpu"

@dataclass
class GameConfig:
    """Main game configuration"""
    num_players: int
    max_speech_rounds: int
    reverse_vote_rules: bool
    render: bool = False
    visualize: bool = False
    seed: int = 42
    device: str = "cpu"
    agent_configs: List[AgentConfig] = None

    def __post_init__(self):
        if self.agent_configs is None:
            # Default to all random agents if not specified
            self.agent_configs = [
                AgentConfig(AgentType.RANDOM) for _ in range(self.num_players)
            ]
        elif len(self.agent_configs) != self.num_players:
            raise ValueError(f"Number of agent configs ({len(self.agent_configs)}) "
                           f"does not match number of players ({self.num_players})")

class GameSettings:
    """Main class for managing game settings"""
    
    @staticmethod
    def from_args(args) -> GameConfig:
        """Create game configuration from command line arguments"""
        agent_configs = []
        for i in range(args.num_players):
            agent_type = AgentType(args.agent_types[i % len(args.agent_types)])
            config = AgentConfig(
                agent_type=agent_type,
                model_path=args.load_model if agent_type == AgentType.RL else None,
                is_training=args.mode == 'train',
                device=args.device
            )
            agent_configs.append(config)
        
        return GameConfig(
            num_players=args.num_players,
            max_speech_rounds=args.max_speech_rounds,
            reverse_vote_rules=args.reverse_vote_rules,
            render=args.render,
            visualize=args.visualize,
            seed=args.seed,
            device=args.device,
            agent_configs=agent_configs
        )
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> GameConfig:
        """Create game configuration from dictionary"""
        agent_configs = []
        for agent_dict in config_dict.get('agent_configs', []):
            config = AgentConfig(
                agent_type=AgentType(agent_dict['agent_type']),
                model_path=agent_dict.get('model_path'),
                is_training=agent_dict.get('is_training', False),
                device=agent_dict.get('device', 'cpu')
            )
            agent_configs.append(config)
        
        return GameConfig(
            num_players=config_dict['num_players'],
            max_speech_rounds=config_dict['max_speech_rounds'],
            reverse_vote_rules=config_dict['reverse_vote_rules'],
            render=config_dict.get('render', False),
            visualize=config_dict.get('visualize', False),
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', 'cpu'),
            agent_configs=agent_configs
        )
    
    @staticmethod
    def create_default_config() -> GameConfig:
        """Create default game configuration"""
        return GameConfig(
            num_players=6,
            max_speech_rounds=3,
            reverse_vote_rules=True,
            render=False,
            visualize=False,
            seed=42,
            device='cpu',
            agent_configs=[AgentConfig(AgentType.RANDOM) for _ in range(6)]
        ) 