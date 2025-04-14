"""
Game Manager
Manages game instances with stateless design
"""

import logging
import time
import uuid
from typing import Optional, Dict, List, Any, Union
from threading import Lock
from dataclasses import dataclass, field
import sys
import os
import random

# Setup logging
logger = logging.getLogger("game_manager")

GAME_CACHE = {}
cache_lock = Lock()

# Add project root to sys.path to ensure imports work correctly 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

@dataclass
class PlayerConfig:
    """Player configuration for game setup"""
    is_human: bool = False
    name: str = ""
    agent_type: Optional[str] = "heuristic"
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "is_human": self.is_human,
            "name": self.name,
            "agent_type": self.agent_type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlayerConfig':
        """Create from dictionary"""
        return cls(
            is_human=data.get("is_human", False),
            name=data.get("name", ""),
            agent_type=data.get("agent_type", "heuristic")
        )

@dataclass
class GameConfig:
    """Game configuration schema matching the expected structure"""
    num_players: int = 6
    roles: List[str] = field(default_factory=lambda: ["werewolf", "werewolf", "minion", "seer", "robber", "troublemaker", "villager", "villager", "insomniac"])
    center_card_count: int = 3
    max_speech_rounds: int = 3
    seed: Optional[int] = None
    players: Dict[str, PlayerConfig] = field(default_factory=dict)
    required_player_roles: List[str] = field(default_factory=list)
    enforce_required_roles: bool = False
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "num_players": self.num_players,
            "roles": self.roles,
            "center_card_count": self.center_card_count,
            "max_speech_rounds": self.max_speech_rounds,
            "seed": self.seed,
            "players": {k: v.dict() for k, v in self.players.items()},
        }
        
        # Add optional fields if they have values
        if self.required_player_roles:
            result["required_player_roles"] = self.required_player_roles
        if self.enforce_required_roles:
            result["enforce_required_roles"] = self.enforce_required_roles
            
        return result
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'GameConfig':
        """Create GameConfig from JSON data"""
        config = cls(
            num_players=json_data.get("num_players", 6),
            roles=json_data.get("all_roles", []),
            center_card_count=json_data.get("center_card_count", 3),
            max_speech_rounds=json_data.get("max_speech_rounds", 3),
            seed=json_data.get("seed"),
            required_player_roles=json_data.get("required_player_roles", []),
            enforce_required_roles=json_data.get("enforce_required_roles", False)
        )
        
        # Add players if specified
        if "players" in json_data:
            for player_id, player_data in json_data["players"].items():
                config.players[player_id] = PlayerConfig(
                    is_human=player_data.get("is_human", False),
                    name=player_data.get("name", f"Player {player_id}"),
                    agent_type=player_data.get("agent_type")
                )
                
        return config


class GameManager:
    """
    Stateless Game Manager
    
    Instead of keeping game state in memory, this manager uses a cache (in-memory for
    development, but should be replaced with Redis or similar for production) to store
    game state between requests.
    """
    
    # Class attribute to store game states
    game_states = {}
    
    @staticmethod
    def create_game(config: GameConfig) -> dict:
        """
        Create a new game with the specified configuration and 
        immediately return the full initial game state
        """
        try:
            # Generate a unique game ID
            game_id = str(uuid.uuid4())[:8]
            
            # Validate that num_players is positive and valid
            if config.num_players <= 0:
                return {
                    "success": False,
                    "message": "Number of players must be positive"
                }
                
            # 修复：自动扩展角色列表，而不是直接返回错误
            if len(config.roles) < config.num_players:
                additional_roles_needed = config.num_players - len(config.roles)
                config.roles.extend(["villager"] * additional_roles_needed)
                logger.warning(f"Not enough roles provided. Adding {additional_roles_needed} villager roles.")
            
            # Log configuration for debugging
            logger.info(f"Creating game with ID: {game_id}")
            logger.info(f"Number of players: {config.num_players}")
            logger.info(f"Roles: {config.roles}")
            logger.info(f"Players: {config.players}")
            
            # Create initial game environment config
            env_config = {
                "num_players": config.num_players,
                "roles": config.roles[:config.num_players + config.center_card_count],  # 包括中央牌的角色
                "center_card_count": config.center_card_count,
                "max_speech_rounds": config.max_speech_rounds,
                "seed": config.seed
            }
            
            # Build player configurations for the environment
            player_configs = {}
            for i in range(config.num_players):
                player_id_str = str(i)
                
                # Default configuration if not provided
                player_config = {
                    "is_human": False,
                    "name": f"Player {i}",
                    "agent_type": "heuristic"
                }
                
                # Check if this player is configured in the request
                if player_id_str in config.players:
                    player = config.players.get(player_id_str)
                    if player:
                        player_config = {
                            "is_human": player.is_human,
                            "name": player.name,
                            "agent_type": player.agent_type if not player.is_human else None
                        }
                
                # Add to player configs
                player_configs[i] = player_config
                
            logger.info(f"Player configs prepared: {player_configs}")
            
            # Preprocess the environment config
            processed_config = preprocess_env_config(env_config)
            
            try:
                # Create the environment and get initial state
                env, initial_state = create_game_environment(processed_config, player_configs)
                
                # Update players in the initial state
                for player_id, config in player_configs.items():
                    found = False
                    for player in initial_state["players"]:
                        if player["player_id"] == player_id:
                            player["is_human"] = config["is_human"]
                            player["name"] = config["name"]
                            player["agent_type"] = config["agent_type"]
                            found = True
                            break
                            
                    if not found:
                        logger.warning(f"Player {player_id} not found in initial state")
                
                # 确保初始状态包含game_id
                if "game_id" not in initial_state:
                    initial_state["game_id"] = game_id
                else:
                    # 使用环境生成的game_id
                    game_id = initial_state["game_id"]
                    
                # 确保初始状态包含max_speech_rounds
                if "max_speech_rounds" not in initial_state:
                    initial_state["max_speech_rounds"] = config.max_speech_rounds
                
                # Store the game state in cache
                game_state = {
                    "game_id": game_id,
                    "config": {
                        "num_players": config.num_players,
                        "roles": config.roles,
                        "center_card_count": config.center_card_count,
                        "max_speech_rounds": config.max_speech_rounds,
                        "seed": config.seed,
                        "players": {pid: pc.dict() if hasattr(pc, 'dict') else pc for pid, pc in config.players.items()}
                    },
                    "state": initial_state,
                    "history": [],
                    "created_at": time.time()
                }
                
                with cache_lock:
                    GAME_CACHE[game_id] = game_state
                
                logger.info(f"Game created successfully: {game_id}")
                
                # Return the game ID and initial state with correct format
                return {
                    "game_id": game_id,
                    "message": "Game created successfully",
                    "success": True,
                    "state": initial_state
                }
            except Exception as e:
                logger.error(f"Error creating environment: {str(e)}")
                return {
                    "success": False,
                    "message": f"Error creating environment: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Error creating game: {str(e)}")
            return {
                "success": False,
                "message": f"Error creating game: {str(e)}"
            }
    
    @staticmethod
    def create_test_game(test_game_type="heuristic", num_players=3, seed=None):
        """
        Create a test game with preconfigured settings
        
        Args:
            test_game_type: Type of test game to create (default: "heuristic")
            num_players: Number of players in the test game (default: 3)
            seed: Optional random seed for reproducibility (default: None)
            
        Returns:
            dict: Game creation result with game ID and initial state
        """
        try:
            logger.info(f"Creating test game with type={test_game_type}, num_players={num_players}, seed={seed}")
            
            # Validate num_players
            if num_players <= 0:
                raise ValueError("Number of players must be positive")
                
            if num_players > 10:
                raise ValueError("Maximum number of players is 10")
            
            # Generate unique game ID
            game_id = str(uuid.uuid4())[:8]
            logger.info(f"Generated game ID: {game_id}")
            
            # Create roles configuration
            roles = ["werewolf"]  # At least one werewolf
            
            # Add other roles based on number of players
            additional_roles = []
            if num_players >= 3:
                additional_roles.extend(["seer", "robber"])
            if num_players >= 5:
                additional_roles.extend(["troublemaker", "drunk"])
            if num_players >= 7:
                additional_roles.extend(["insomniac", "mason", "mason"])
            if num_players >= 10:
                additional_roles.extend(["minion", "hunter"])
                
            # Fill remaining slots with villagers
            villager_count = max(0, num_players - 1 - len(additional_roles))
            additional_roles.extend(["villager"] * villager_count)
            
            # Shuffle roles and select required number
            random.shuffle(additional_roles)
            roles.extend(additional_roles[:num_players - 1])
            
            logger.info(f"Roles for game: {roles}")
            
            # Create player configurations
            player_configs = {}
            for i in range(num_players):
                is_human = (i == 0)  # First player is human by default
                player_configs[i] = {
                    "is_human": is_human,
                    "name": f"Player {i}" if not is_human else "You",
                    "agent_type": test_game_type
                }
            
            logger.info(f"Player configs: {player_configs}")
            
            # Create environment config
            env_config = {
                "num_players": num_players,
                "roles": roles,
                "center_card_count": 3,
                "max_speech_rounds": 2,
                "seed": seed
            }
            
            # Create game environment
            env, initial_state = create_game_environment(env_config, player_configs)
            
            # Store game state in cache
            game_state = {
                "game_id": game_id,
                "environment": env,
                "state": initial_state,
                "history": [initial_state],
                "config": env_config,
                "created_at": time.time()
            }
            
            with cache_lock:
                GAME_CACHE[game_id] = game_state
            
            logger.info(f"Test game created successfully with ID: {game_id}")
            return {
                "game_id": game_id,
                "message": "Test game created successfully",
                "success": True,
                "state": initial_state
            }
            
        except Exception as e:
            error_msg = f"Error creating test game: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {
                "success": False,
                "message": error_msg
            }
    
    @staticmethod
    def get_game_state(game_id: str, player_id: Optional[int] = None) -> dict:
        """
        Get the current game state
        
        If player_id is provided, return only the information visible to that player
        """
        with cache_lock:
            game_data = GAME_CACHE.get(game_id)
        
        if not game_data:
            return {"success": False, "message": "Game not found"}
        
        # Filter state information based on player_id if needed
        state = game_data["state"]
        
        if player_id is not None:
            # Create a filtered state with only information visible to the player
            # This will depend on game rules and player's role
            filtered_state = GameManager._filter_state_for_player(state, player_id)
            return filtered_state
        
        return {
            "success": True,
            "game_id": game_id,
            "state": state
        }
    
    @staticmethod
    def perform_action(game_id: str, player_id: int, action: dict) -> dict:
        """
        Execute a player's action and update the game state
        """
        with cache_lock:
            game_data = GAME_CACHE.get(game_id)
        
        if not game_data:
            return {"success": False, "message": "Game not found"}
        
        # Get the current game state
        current_state = game_data["state"]
        
        # Verify it's the correct player's turn
        if current_state.get("current_player") != player_id:
            return {
                "success": False, 
                "message": f"Not your turn. Current player is {current_state.get('current_player')}"
            }
        
        # Apply the action and get the new state
        action_result, new_state = apply_action(
            current_state,
            player_id, 
            action
        )
        
        # Update history
        game_data["history"].append({
            "player_id": player_id,
            "action": action,
            "result": action_result,
            "timestamp": time.time()
        })
        
        # Update the game state in cache
        game_data["state"] = new_state
        
        with cache_lock:
            GAME_CACHE[game_id] = game_data
        
        # Return the action result and state update
        return {
            "success": True,
            "message": "Action executed successfully",
            "action_result": action_result,
            "state_update": {
                "phase": new_state.get("phase"),
                "round": new_state.get("round"),
                "speech_round": new_state.get("speech_round"),
                "current_player": new_state.get("current_player"),
                "cumulative_rewards": new_state.get("cumulative_rewards", {})
            }
        }
    
    @staticmethod
    def get_ai_decision(game_id: str, player_id: int, game_state: Optional[dict] = None) -> dict:
        """
        Get a decision from an AI player
        
        If game_state is provided, use that instead of fetching from cache
        (useful for frontend-managed state)
        """
        state_to_use = game_state
        
        if not state_to_use:
            with cache_lock:
                game_data = GAME_CACHE.get(game_id)
            
            if not game_data:
                return {"success": False, "message": "Game not found"}
            
            state_to_use = game_data["state"]
        
        # Find the player in the state
        player = None
        for p in state_to_use.get("players", []):
            if p.get("player_id") == player_id:
                player = p
                break
        
        if not player:
            return {"success": False, "message": "Player not found"}
        
        if player.get("is_human", False):
            return {"success": False, "message": "Cannot get AI decision for human player"}
        
        # Create an AI agent and get its decision
        agent_type = player.get("agent_type", "heuristic")
        agent = create_agent(agent_type, player_id, state_to_use)
        
        # Get the decision
        action, reasoning = agent.decide_action(state_to_use)
        
        return {
            "success": True,
            "player_id": player_id,
            "action": action,
            "reasoning": reasoning
        }
    
    @staticmethod
    def step_game(game_id: str) -> dict:
        """
        Advance the game by one step (for automated testing)
        """
        with cache_lock:
            game_data = GAME_CACHE.get(game_id)
        
        if not game_data:
            return {"success": False, "message": "Game not found"}
        
        # Get the current game state
        current_state = game_data["state"]
        
        # Get the current player
        current_player_id = current_state.get("current_player")
        
        if current_player_id is None:
            return {"success": False, "message": "No current player"}
        
        # Find the player in the state
        player = None
        for p in current_state.get("players", []):
            if p.get("player_id") == current_player_id:
                player = p
                break
        
        if not player:
            return {"success": False, "message": "Current player not found"}
        
        # Get AI decision
        agent_type = player.get("agent_type", "heuristic")
        agent = create_agent(agent_type, current_player_id, current_state)
        
        # Get the decision
        action, reasoning = agent.decide_action(current_state)
        
        # Apply the action
        action_result, new_state = apply_action(
            current_state,
            current_player_id, 
            action
        )
        
        # Update history
        game_data["history"].append({
            "player_id": current_player_id,
            "action": action,
            "result": action_result,
            "timestamp": time.time(),
            "step": len(game_data["history"])
        })
        
        # Update the game state in cache
        game_data["state"] = new_state
        
        with cache_lock:
            GAME_CACHE[game_id] = game_data
        
        # Return the step information
        return {
            "success": True,
            "step": len(game_data["history"]) - 1,
            "action": {
                "player_id": current_player_id,
                "player_role": player.get("current_role"),
                "action_type": action.get("action_type"),
                "action_name": action.get("action_name"),
                "action_params": action.get("action_params", {})
            },
            "state_update": {
                "phase": new_state.get("phase"),
                "round": new_state.get("round"),
                "speech_round": new_state.get("speech_round"),
                "current_player": new_state.get("current_player"),
                "cumulative_rewards": new_state.get("cumulative_rewards", {})
            }
        }
    
    @staticmethod
    def get_game_result(game_id: str) -> dict:
        """
        Get the complete results after a game has ended
        """
        with cache_lock:
            game_data = GAME_CACHE.get(game_id)
        
        if not game_data:
            return {"success": False, "message": "Game not found"}
        
        # Get the current game state
        current_state = game_data["state"]
        
        # Check if the game is over
        if not current_state.get("game_over", False):
            return {
                "success": False, 
                "message": "Game is not over yet",
                "state": {
                    "phase": current_state.get("phase"),
                    "round": current_state.get("round"),
                    "current_player": current_state.get("current_player")
                }
            }
        
        # Compile game results
        winner = current_state.get("winner")
        voting_results = current_state.get("voting_results", {})
        
        # Get role allocation
        role_allocation = []
        for player in current_state.get("players", []):
            role_allocation.append(player.get("original_role"))
        
        # Add center cards
        for card in current_state.get("center_cards", []):
            role_allocation.append(card)
        
        # Compile statistics
        total_steps = len(game_data["history"])
        
        # Count steps per phase
        steps_per_phase = {
            "night": 0,
            "day": 0,
            "vote": 0
        }
        
        for entry in game_data["history"]:
            action = entry.get("action", {})
            action_type = action.get("action_type", "")
            
            if action_type == "NIGHT_ACTION":
                steps_per_phase["night"] += 1
            elif action_type == "DAY_SPEECH":
                steps_per_phase["day"] += 1
            elif action_type == "VOTE":
                steps_per_phase["vote"] += 1
        
        return {
            "success": True,
            "game_id": game_id,
            "winner": winner,
            "game_over": True,
            "voting_results": voting_results,
            "role_allocation": role_allocation,
            "player_info": current_state.get("players", []),
            "center_cards": current_state.get("center_cards", []),
            "statistics": {
                "total_game_steps": total_steps,
                "steps_per_phase": steps_per_phase
            },
            "game_summary": f"{winner} team wins! Game completed in {total_steps} steps."
        }
    
    @staticmethod
    def _filter_state_for_player(state: dict, player_id: int) -> dict:
        """
        Filter the game state to only include information visible to the specified player
        """
        # Create a copy of the state to modify
        filtered_state = state.copy()
        
        # Find the player's information
        player = None
        for p in state.get("players", []):
            if p.get("player_id") == player_id:
                player = p
                break
        
        if not player:
            return {"success": False, "message": "Player not found"}
        
        # Hide information based on the player's role and game phase
        phase = state.get("phase")
        player_role = player.get("original_role")
        
        # Always hide other players' original roles unless werewolf seeing other werewolves
        werewolf_indices = state.get("werewolf_indices", [])
        is_werewolf = player_id in werewolf_indices
        
        # Create a list of players with hidden information
        filtered_players = []
        for p in state.get("players", []):
            p_copy = p.copy()
            
            # Don't hide the player's own information
            if p.get("player_id") == player_id:
                filtered_players.append(p_copy)
                continue
            
            # Hide original role unless player is werewolf and other player is also werewolf
            if not (is_werewolf and p.get("player_id") in werewolf_indices):
                if "original_role" in p_copy:
                    p_copy["original_role"] = "?"
                
                if "current_role" in p_copy and phase != "game_over":
                    p_copy["current_role"] = "?"
            
            filtered_players.append(p_copy)
        
        filtered_state["players"] = filtered_players
        
        # Hide center cards unless they've been viewed
        if "center_cards" in filtered_state and phase != "game_over":
            known_center_cards = state.get("known_center_cards", {}).get(str(player_id), {})
            
            filtered_center_cards = []
            for i, card in enumerate(state.get("center_cards", [])):
                if str(i) in known_center_cards:
                    filtered_center_cards.append(card)
                else:
                    filtered_center_cards.append("?")
            
            filtered_state["center_cards"] = filtered_center_cards
        
        # Add visible roles based on player's knowledge
        visible_roles = state.get("visible_roles", {}).get(str(player_id), {})
        filtered_state["visible_roles"] = visible_roles
        
        # 修复：确保game_id存在，如果state中没有，则使用参数中的game_id
        game_id = state.get("game_id", filtered_state.get("game_id", ""))
        
        # 修复：确保max_speech_rounds有一个默认值
        max_speech_rounds = state.get("max_speech_rounds", 3)
        
        return {
            "success": True,
            "game_id": game_id,  # 确保game_id存在
            "phase": phase,
            "current_player_id": state.get("current_player"),
            "current_role": player.get("current_role"),
            "players": filtered_players,
            "player_count": len(state.get("players", [])),
            "center_cards": filtered_state.get("center_cards", []),
            "known_center_cards": state.get("known_center_cards", {}).get(str(player_id), {}),
            "visible_roles": visible_roles,
            "turn": state.get("round", 0),
            "action_order": state.get("action_order", []),
            "valid_actions": state.get("valid_actions", {}).get(str(player_id), []),
            "speech_round": state.get("speech_round"),
            "max_speech_rounds": max_speech_rounds,  # 使用默认值确保字段存在
            "votes": state.get("votes"),
            "winner": state.get("winner") if state.get("game_over") else None,
            "game_over": state.get("game_over", False),
            "history": state.get("history", []),
            "message": None
        }

# Import necessary modules for game environment and agents
from werewolf_env import WerewolfEnv
from agents import RandomAgent, HeuristicAgent
from config.default_config import ROLE_TEAMS

def preprocess_env_config(env_config):
    """
    Preprocesses the environment configuration to ensure it's valid
    
    Args:
        env_config: The raw environment configuration
        
    Returns:
        dict: The processed environment configuration
    """
    # Make a copy of the config to avoid modifying the original
    config = env_config.copy()
    
    # Ensure num_players is positive
    if config.get("num_players", 0) <= 0:
        config["num_players"] = 6  # Default to 6 players
        
    # Ensure we have enough roles
    num_players = config["num_players"]
    roles = config.get("roles", [])
    
    if len(roles) < num_players:
        # Add more roles if needed (default to villager)
        additional_roles_needed = num_players - len(roles)
        roles.extend(["villager"] * additional_roles_needed)
        config["roles"] = roles
        
    # Ensure center_card_count is valid (default to 3)
    if "center_card_count" not in config or config["center_card_count"] < 0:
        config["center_card_count"] = 3
        
    # Ensure max_speech_rounds is valid
    if "max_speech_rounds" not in config or config["max_speech_rounds"] < 0:
        config["max_speech_rounds"] = 3
        
    return config

def create_game_environment(env_config, player_configs):
    """
    Create a new game environment with the specified configuration
    
    Args:
        env_config: Game environment configuration
        player_configs: Player configurations
        
    Returns:
        tuple: (environment, initial_state)
    """
    try:
        # Validate num_players
        num_players = env_config.get("num_players", 0)
        if num_players <= 0:
            raise ValueError("Number of players must be positive")
            
        # Validate roles
        roles = env_config.get("roles", [])
        if len(roles) < num_players:
            # 修复: 自动扩展角色列表，而不是抛出错误
            additional_roles_needed = num_players - len(roles)
            roles.extend(["villager"] * additional_roles_needed)
            logger.warning(f"Not enough roles provided. Adding {additional_roles_needed} villager roles.")
        
        # 使用环境配置中的种子来保证可重复性
        seed = env_config.get("seed")
        if seed is not None:
            import random
            random.seed(seed)
            
        # 生成游戏ID
        import uuid
        game_id = str(uuid.uuid4())[:8]
        
        # Create a simplified environment for testing
        logger.info("Creating game environment")
        logger.info(f"Environment config: {env_config}")
        logger.info(f"Player configs: {player_configs}")
        
        # Create initial state structure
        initial_state = {
            "game_id": game_id,  # 添加game_id字段
            "phase": "night",
            "round": 0,
            "speech_round": 0,
            "current_player": 0,
            "players": [],
            "center_cards": ["villager", "tanner", "hunter"],  # 示例中心牌
            "werewolf_indices": [],
            "villager_indices": [],
            "action_order": [
                "werewolf", "minion", "mason", "seer", 
                "robber", "troublemaker", "drunk", "insomniac"
            ],
            "max_speech_rounds": env_config.get("max_speech_rounds", 3)  # 添加max_speech_rounds字段
        }
        
        # Assign roles to players
        for i in range(num_players):
            role = roles[i] if i < len(roles) else "villager"
            player_info = {
                "player_id": i,
                "name": f"Player {i}",
                "is_human": False,
                "original_role": role,
                "current_role": role,
                "team": "villager" if role != "werewolf" and role != "minion" else "werewolf",
                "agent_type": "heuristic"
            }
            
            # Apply player config if available
            if i in player_configs:
                config = player_configs[i]
                player_info["is_human"] = config.get("is_human", False)
                player_info["name"] = config.get("name", f"Player {i}")
                player_info["agent_type"] = config.get("agent_type", "heuristic")
                
            # Add to player list
            initial_state["players"].append(player_info)
            
            # Track werewolf and villager indices
            if role == "werewolf":
                initial_state["werewolf_indices"].append(i)
            else:
                initial_state["villager_indices"].append(i)
                
        # Mock environment
        class MockEnv:
            def __init__(self, config):
                self.config = config
                
            def reset(self):
                return initial_state
                
        return MockEnv(env_config), initial_state
        
    except Exception as e:
        logger.error(f"Error creating game environment: {str(e)}")
        raise


def apply_action(current_state, player_id, action):
    """
    Apply an action to the current state and get the new state
    
    Args:
        current_state: Current game state
        player_id: Player ID performing the action
        action: Action to perform
        
    Returns:
        tuple: (action_result, new_state)
    """
    # Create a temporary environment from the current state
    env = WerewolfEnv()
    env.game_state = current_state
    
    # Apply the action
    next_state, reward, terminated, truncated, info = env.step(action)
    
    # Return the action result and new state
    action_result = {
        "success": info.get("success", True),
        "message": info.get("message", "Action applied successfully"),
        "terminated": terminated,
        "truncated": truncated,
        "reward": reward
    }
    
    return action_result, next_state


def create_agent(agent_type, player_id, state):
    """
    Create an AI agent of the specified type
    
    Args:
        agent_type: Type of agent ("random", "heuristic", etc.)
        player_id: Player ID
        state: Current game state
        
    Returns:
        Agent: An AI agent
    """
    # Get player role from state for context
    player_role = None
    for player in state.get("players", []):
        if player.get("player_id") == player_id:
            player_role = player.get("current_role")
            break
    
    # Create agent based on type
    if agent_type == "random":
        agent = RandomAgent(player_id)
    elif agent_type == "heuristic":
        agent = HeuristicAgent(player_id)
    else:
        # Default to heuristic agent
        agent = HeuristicAgent(player_id)
    
    # Set agent's current role if available
    if hasattr(agent, "current_role") and player_role:
        agent.current_role = player_role
    
    # Initialize the agent with the current state
    if hasattr(agent, "initialize"):
        agent.initialize(state)
    
    return agent 