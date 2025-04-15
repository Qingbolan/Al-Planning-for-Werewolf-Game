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
from .models import GameState, NightAction, NightActionResponse

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

    # 添加一个StateWrapper类，用于将字典转换为对象形式
    class StateWrapper:
        def __init__(self, state_dict):
            # 将字典中的所有键值对复制到对象属性
            for key, value in state_dict.items():
                setattr(self, key, value)
            
            # 计算并添加num_players属性（玩家数量）
            if hasattr(self, 'players') and isinstance(self.players, list):
                self.num_players = len(self.players)
            else:
                self.num_players = 0
                
            # 添加更多可能需要的属性
            # 角色列表
            if not hasattr(self, 'roles'):
                if hasattr(self, 'players') and isinstance(self.players, list):
                    self.roles = [p.get('original_role', 'unknown') for p in self.players]
                else:
                    self.roles = []
                    
            # 狼人列表
            if not hasattr(self, 'werewolf_indices'):
                self.werewolf_indices = []
                if hasattr(self, 'players') and isinstance(self.players, list):
                    for i, player in enumerate(self.players):
                        if player.get('original_role') == 'werewolf' or player.get('team') == 'werewolf':
                            self.werewolf_indices.append(player.get('player_id', i))
            
            # 中央牌数量
            if hasattr(self, 'center_cards') and isinstance(self.center_cards, list):
                self.center_card_count = len(self.center_cards)
            elif hasattr(self, 'config') and isinstance(self.config, dict) and 'center_card_count' in self.config:
                self.center_card_count = self.config.get('center_card_count', 3)
            else:
                self.center_card_count = 3
                
            # 添加num_center_cards属性，与center_card_count相同
            self.num_center_cards = self.center_card_count
        
        def __getattr__(self, name):
            """处理所有未找到的属性访问，特别是get_xxx方法调用"""
            # 处理特殊的方法调用
            if name == 'get_observation':
                def get_observation(player_id=None):
                    """返回当前游戏状态的观察结果"""
                    # 如果没有提供player_id，使用当前玩家
                    if player_id is None:
                        player_id = getattr(self, 'current_player', 0)
                    
                    # 创建观察结果
                    observation = {
                        'phase': getattr(self, 'phase', None),
                        'round': getattr(self, 'round', 0),
                        'current_player': getattr(self, 'current_player', None),
                        'speech_round': getattr(self, 'speech_round', 0),
                        'players': getattr(self, 'players', []),
                        'center_cards': getattr(self, 'center_cards', []),
                        'action_order': getattr(self, 'action_order', []),
                        'votes': getattr(self, 'votes', {})
                    }
                    
                    return observation
                
                return get_observation
            
            # 添加get_player_role方法
            elif name == 'get_player_role':
                def get_player_role(player_id):
                    """获取指定玩家的角色"""
                    players = getattr(self, 'players', [])
                    for player in players:
                        if player.get('player_id') == player_id:
                            return player.get('current_role', 'unknown')
                    return 'unknown'
                
                return get_player_role
            
            # 添加get_valid_actions方法
            elif name == 'get_valid_actions':
                def get_valid_actions(player_id=None):
                    """获取有效动作列表"""
                    # 如果没有提供player_id，使用当前玩家
                    if player_id is None:
                        player_id = getattr(self, 'current_player', 0)
                        
                    # 尝试从valid_actions字典中获取
                    valid_actions = getattr(self, 'valid_actions', {})
                    return valid_actions.get(str(player_id), [])
                
                return get_valid_actions
            
            # 添加get_num_center_cards方法
            elif name == 'get_num_center_cards':
                return lambda: getattr(self, 'num_center_cards', 3)
            
            # 添加get_center_cards方法
            elif name == 'get_center_cards':
                return lambda: getattr(self, 'center_cards', [])
            
            # 添加get_original_role_for_player方法
            elif name == 'get_original_role_for_player':
                def get_original_role(pid):
                    players = getattr(self, 'players', [])
                    for p in players:
                        if p.get('player_id') == pid:
                            return p.get('original_role', 'unknown')
                    return 'unknown'
                return get_original_role
            
            # 添加get_current_role_for_player方法
            elif name == 'get_current_role_for_player':
                def get_current_role(pid):
                    players = getattr(self, 'players', [])
                    for p in players:
                        if p.get('player_id') == pid:
                            return p.get('current_role', 'unknown')
                    return 'unknown'
                return get_current_role
            
            # 处理所有get_xxx方法调用
            elif name.startswith('get_'):
                # 从方法名提取属性名（去掉get_前缀）
                attr_name = name[4:]
                
                # 默认情况：返回一个函数，调用时返回属性值（如果存在）或适当的默认值
                return lambda *args, **kwargs: getattr(self, attr_name, None)
                
            # 对于任何其他属性访问，尝试直接返回相应的属性
            # 这对于belief_updater等可能直接访问属性的代码很有用
            try:
                # 直接从字典属性中找
                return getattr(self.__dict__, name, None)
            except:
                pass
                
            # 如果所有尝试都失败，才抛出AttributeError
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
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
    def create_test_game(test_game_type="heuristic", num_players=6, seed=42):
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
            roles = ["werewolf", "werewolf", "minion", "seer", "robber", "troublemaker", "villager", "villager", "insomniac"]  # At least one werewolf
            
            logger.info(f"Roles for game: {roles}")
            
            # Create player configurations
            player_configs = {}
            for i in range(num_players):
                # 测试游戏中所有玩家都应该是AI玩家
                player_configs[i] = {
                    "is_human": False,
                    "agent_type": test_game_type
                }
            
            logger.info(f"Player configs: {player_configs}")
            
            # Create environment config
            env_config = {
                "num_players": num_players,
                "roles": roles,
                "center_card_count": 3,
                "max_speech_rounds": 3,
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
        try:
            state_to_use = game_state
            
            if not state_to_use:
                with cache_lock:
                    game_data = GAME_CACHE.get(game_id)
                
                if not game_data:
                    return {"success": False, "message": "Game not found"}
                
                state_to_use = game_data["state"]
            
            logger.info(f"Getting AI decision for player {player_id} in game {game_id}")
            
            # Find the player in the state
            player = None
            for p in state_to_use.get("players", []):
                if p.get("player_id") == player_id:
                    player = p
                    break
            
            if not player:
                logger.warning(f"Player {player_id} not found in game {game_id}")
                return {"success": False, "message": "Player not found"}
            
            if player.get("is_human", False):
                logger.warning(f"Player {player_id} is human, cannot get AI decision")
                return {"success": False, "message": "Cannot get AI decision for human player"}
            
            # Check if it's this player's turn
            current_player = state_to_use.get("current_player")
            if current_player != player_id:
                logger.warning(f"Not player {player_id}'s turn (current player is {current_player})")
                return {
                    "success": False, 
                    "message": f"Not this player's turn. Current player is {current_player}"
                }
            
            # Create an AI agent and get its decision
            agent_type = player.get("agent_type", "heuristic")
            logger.info(f"Creating {agent_type} agent for player {player_id}")
            
            try:
                agent = create_agent(agent_type, player_id, state_to_use)
                
                # 将字典状态转换为对象状态，以便代理可以使用.属性方式访问
                # 即使create_agent内部已经使用了StateWrapper，这里也再次包装
                # 是为了确保与step_game方法保持一致的调用模式
                wrapped_state = GameManager.StateWrapper(state_to_use)
                
                # Get the decision
                logger.info(f"Agent {agent_type} deciding action for player {player_id}")
                action, reasoning = agent.decide_action(wrapped_state)
                
                logger.info(f"Agent decision: {action} with reasoning: {reasoning}")
                
                return {
                    "success": True,
                    "player_id": player_id,
                    "action": action,
                    "reasoning": reasoning
                }
            except Exception as agent_error:
                logger.error(f"Error creating or using agent: {str(agent_error)}")
                logger.exception(agent_error)
                return {
                    "success": False,
                    "message": f"Error with AI agent: {str(agent_error)}"
                }
        except Exception as e:
            logger.error(f"Error in get_ai_decision: {str(e)}")
            logger.exception(e)
            return {
                "success": False,
                "message": f"Internal error: {str(e)}"
            }
    
    @staticmethod
    def step_game(game_id: str) -> dict:
        """
        Advance the game by one step (for automated testing)
        """
        try:
            logger.info(f"Stepping game {game_id}")
            
            with cache_lock:
                game_data = GAME_CACHE.get(game_id)
            
            if not game_data:
                logger.warning(f"Game {game_id} not found")
                return {"success": False, "message": "Game not found"}
            
            # Get the current game state
            current_state = game_data["state"]
            
            # Check if game is over
            if current_state.get("game_over", False):
                logger.info(f"Game {game_id} is over, cannot step further")
                return {
                    "success": False, 
                    "message": "Game is already over", 
                    "game_over": True,
                    "winner": current_state.get("winner")
                }
            
            # Get the current player
            current_player_id = current_state.get("current_player")
            
            if current_player_id is None:
                logger.warning(f"Game {game_id} has no current player")
                return {"success": False, "message": "No current player"}
            
            logger.info(f"Current player is {current_player_id}")
            
            # Find the player in the state
            player = None
            for p in current_state.get("players", []):
                if p.get("player_id") == current_player_id:
                    player = p
                    break
            
            if not player:
                logger.warning(f"Current player {current_player_id} not found in game {game_id}")
                return {"success": False, "message": "Current player not found"}
            
            # Check if player is human - cannot step automatically for human players
            if player.get("is_human", False):
                logger.warning(f"Player {current_player_id} is human, cannot auto-step")
                return {
                    "success": False, 
                    "message": "Cannot automatically step for human player",
                    "player_id": current_player_id,
                    "is_human": True
                }
            
            try:
                # Get AI decision
                agent_type = player.get("agent_type", "heuristic")
                logger.info(f"Creating {agent_type} agent for player {current_player_id}")
                
                # 创建代理，现在create_agent函数内部会使用StateWrapper
                agent = create_agent(agent_type, current_player_id, current_state)
                
                # Get the decision
                logger.info(f"Agent deciding action for player {current_player_id}")
                # 创建代理时已经使用了StateWrapper，这里重新包装是为了保持与get_ai_decision方法一致
                wrapped_state = GameManager.StateWrapper(current_state)
                action, reasoning = agent.decide_action(wrapped_state)
                
                logger.info(f"Agent decision: {action} with reasoning: {reasoning}")
                
                # Apply the action
                logger.info(f"Applying action for player {current_player_id}")
                action_result, new_state = apply_action(
                    current_state,
                    current_player_id, 
                    action
                )
                
                # Update history
                history_entry = {
                    "player_id": current_player_id,
                    "action": action,
                    "result": action_result,
                    "timestamp": time.time(),
                    "step": len(game_data["history"])
                }
                game_data["history"].append(history_entry)
                
                # Update the game state in cache
                game_data["state"] = new_state
                
                with cache_lock:
                    GAME_CACHE[game_id] = game_data
                
                logger.info(f"Step completed for game {game_id}, player {current_player_id}")
                
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
            except Exception as action_error:
                logger.error(f"Error during step action: {str(action_error)}")
                logger.exception(action_error)
                return {
                    "success": False,
                    "message": f"Error during step: {str(action_error)}"
                }
        except Exception as e:
            logger.error(f"Error in step_game: {str(e)}")
            logger.exception(e)
            return {
                "success": False,
                "message": f"Internal error: {str(e)}"
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

    @staticmethod
    def auto_night_action(player_id: int, game_state: GameState, role: str) -> NightActionResponse:
        """
        Executes a night action for a player automatically based on their role.
        
        Args:
            player_id: The player performing the action
            game_state: Current game state
            role: The role of the player
            
        Returns:
            NightActionResponse with action result and updated game state
        """
        import random  # Import random once at the start of the function
        
        success = False
        action = None
        
        # Execute night action based on player's role
        if role == 'werewolf':
            # Werewolf can check other werewolves or view a center card
            other_werewolves = [p.player_id for p in game_state.players if p.current_role == "werewolf" and p.player_id != player_id]
            if other_werewolves:
                target_id = other_werewolves[0]
                action = NightAction(action_name="werewolf_check", action_params={"target": target_id})
            else:
                # If no other werewolf is available, check a center card (index 0)
                action = NightAction(action_name="werewolf_check", action_params={"center_card": 0})
            success = True
            
        elif role == 'seer':
            # Seer can check a player's role or view two center cards
            player_count = len(game_state.players) if hasattr(game_state, 'players') else 0
            if random.random() < 0.5 and player_count > 1:
                # Choose a player (excluding self) to check their role
                target_id = random.choice([p.player_id for p in game_state.players if p.player_id != player_id])
                action = NightAction(action_name="seer_check", action_params={"target": target_id})
            else:
                # Otherwise, view two center cards
                action = NightAction(action_name="seer_check", action_params={"center_cards": [0, 1]})
            success = True
            
        elif role == 'robber':
            # Robber swaps roles with another player
            player_count = len(game_state.players) if hasattr(game_state, 'players') else 0
            if player_count > 1:
                target_id = random.choice([p.player_id for p in game_state.players if p.player_id != player_id])
                action = NightAction(action_name="robber_swap", action_params={"target": target_id})
                success = True
                
        elif role == 'troublemaker':
            # Troublemaker swaps roles between two other players
            other_players = [p for p in game_state.players if p.player_id != player_id]
            if len(other_players) >= 2:
                target1, target2 = random.sample(other_players, 2)
                action = NightAction(
                    action_name="troublemaker_swap",
                    action_params={
                        "target1": target1.player_id,
                        "target2": target2.player_id
                    }
                )
                success = True
                    
        elif role == 'insomniac':
            # Insomniac checks their own current role
            action = NightAction(action_name="insomniac_check", action_params={})
            success = True
            
        elif role == 'minion':
            # Minion has no active night action
            action = NightAction(action_name="minion_sleep", action_params={})
            success = True
            
        elif role == 'villager':
            # Villager has no night action but still returns a successful response
            action = NightAction(action_name="villager_sleep", action_params={})
            success = True
            
        return NightActionResponse(
            success=success,
            action=action if action else NightAction(action_name="unknown", action_params={}),
            game_state=game_state
        )

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

        # Get role configuration options
        use_complete_roles = env_config.get("use_complete_roles", False)
        
        # Implement role assignment using the test_agents.py approach
        if use_complete_roles:
            # Use the complete standard role set (consistent with ONUW game)
            all_roles = [
                'villager', 'villager',  # 2 villagers
                'werewolf', 'werewolf',  # 2 werewolves
                'minion',                # 1 minion
                'seer',                  # 1 seer
                'troublemaker',          # 1 troublemaker
                'robber',                # 1 robber
                'insomniac'              # 1 insomniac
            ]
            # Calculate center card count (total roles - player count)
            center_card_count = 3
            roles = all_roles
        else:
            # Check if roles are explicitly provided in env_config
            roles = env_config.get("roles", [])
            
            if not roles:
                # Use simple villager/werewolf distribution if no roles provided
                num_werewolves = max(1, num_players // 3)
                num_villagers = num_players - num_werewolves
                roles = ['werewolf'] * num_werewolves + ['villager'] * num_villagers
            
            # Ensure enough roles for all players
            if len(roles) < num_players:
                additional_roles_needed = num_players - len(roles)
                roles.extend(["villager"] * additional_roles_needed)
            
            # Default center card count
            center_card_count = env_config.get("center_card_count", 3)
        
        # Use seed from environment config for reproducibility
        seed = env_config.get("seed")
        if seed is not None:
            import random
            random.seed(seed)
            
            # Shuffle roles if seed is provided
            random.shuffle(roles)
            
        # Generate game ID
        import uuid
        game_id = str(uuid.uuid4())[:8]
        
        # Create initial state structure
        initial_state = {
            "game_id": game_id,
            "phase": "night",
            "round": 0,
            "speech_round": 0,
            "current_player": 0,
            "players": [],
            "roles": roles,  # Store all roles
            "werewolf_indices": [],
            "villager_indices": [],
            "action_order": [
                "werewolf", "minion",  "seer", 
                "robber", "troublemaker", "insomniac"
            ],
            "max_speech_rounds": env_config.get("max_speech_rounds", 3)
        }
        
        # Prioritize critical roles (werewolf x2 and minion) from the existing roles
        # Count how many of each critical role we have in the original role list
        role_counts = {}
        for role in roles:
            role_counts[role] = role_counts.get(role, 0) + 1
            
        # Check if we have enough critical roles
        werewolf_count = role_counts.get('werewolf', 0)
        minion_count = role_counts.get('minion', 0)
        
        if werewolf_count < 2 or minion_count < 1:
            logger.warning(f"Not enough critical roles in the original role list. Found {werewolf_count} werewolves and {minion_count} minions.")
            logger.warning("Standard game requires 2 werewolves and 1 minion.")
            
        # Reorganize roles to prioritize critical roles
        critical_roles = []
        non_critical_roles = []
        
        # Extract werewolves (up to 2)
        werewolf_extracted = 0
        for i, role in enumerate(roles):
            if role == 'werewolf' and werewolf_extracted < 2:
                critical_roles.append(role)
                werewolf_extracted += 1
            else:
                non_critical_roles.append(role)
                
        # Extract minion (1)
        minion_extracted = 0
        for i, role in enumerate(non_critical_roles):
            if role == 'minion' and minion_extracted < 1:
                critical_roles.append(role)
                minion_extracted += 1
                non_critical_roles.remove(role)
                break
        
        # Reorganize roles with critical roles first
        reorganized_roles = critical_roles + non_critical_roles
        
        # Ensure we have enough roles for all players and center cards
        total_roles_needed = num_players + center_card_count
        if len(reorganized_roles) < total_roles_needed:
            additional_roles_needed = total_roles_needed - len(reorganized_roles)
            reorganized_roles.extend(["villager"] * additional_roles_needed)
            
        # Set up center cards - now the critical roles will be in front and assigned to players
        player_roles = reorganized_roles[:num_players]
        center_cards = reorganized_roles[num_players:num_players+center_card_count] if center_card_count > 0 else []
        
        # Add center cards to initial state
        initial_state["center_cards"] = center_cards
        initial_state["roles"] = reorganized_roles  # Update the roles list
        
        # Assign roles to players
        for i in range(num_players):
            role = player_roles[i] if i < len(player_roles) else "villager"
            
            # Determine team based on role
            team = "werewolf" if role in ["werewolf", "minion"] else "villager"
            
            # Default agent type
            agent_type = "heuristic"
            
            # Apply player config if available
            is_human = False
            custom_name = None
            
            if i in player_configs:
                config = player_configs[i]
                is_human = config.get("is_human", False)
                custom_name = config.get("name")
                agent_type = config.get("agent_type", "heuristic")
            
            # Generate player name based on agent type if no custom name provided
            if not custom_name:
                if is_human:
                    player_name = "You"  # 人类玩家显示为"You"
                else:
                    # Map agent type to readable name
                    agent_type_names = {
                        "random": "Random",
                        "heuristic": "Heuristic",
                        "rl": "RL",
                        "mixed": "Mixed"
                    }
                    agent_type_display = agent_type_names.get(agent_type, agent_type.capitalize())
                    player_name = f"{agent_type_display} {i}"  # 更简洁的格式，去掉"Player"一词
            else:
                player_name = custom_name
            
            player_info = {
                "player_id": i,
                "name": player_name,
                "is_human": is_human,
                "original_role": role,
                "current_role": role,
                "team": team,
                "agent_type": agent_type
            }
                
            # Add to player list
            initial_state["players"].append(player_info)
            
            # Track werewolf and villager indices
            if team == "werewolf":
                initial_state["werewolf_indices"].append(i)
            else:
                initial_state["villager_indices"].append(i)
                
        # Create role map for easy access
        initial_state["role_map"] = {player["player_id"]: player["current_role"] for player in initial_state["players"]}
        
        # Create team map for easy access 
        initial_state["team_map"] = {player["player_id"]: player["team"] for player in initial_state["players"]}
                
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
    try:
        # 创建一个临时环境
        env = WerewolfEnv()
        
        # 设置游戏状态
        env.game_state = current_state
        
        # 设置当前玩家ID
        env.current_player_id = player_id
        
        # 日志记录
        logger.info(f"Applying action for player {player_id}: {action}")
        
        # 直接解析动作并执行，而不是调用step方法
        from werewolf_env.actions import Action, create_night_action, create_speech, create_vote, create_no_action
        
        # 如果action已经是Action对象，直接使用
        if isinstance(action, Action):
            action_obj = action
        else:
            # 解析动作
            if isinstance(action, dict):
                action_type = action.get('action_type')
                
                if action_type == 'NIGHT_ACTION':
                    # 夜晚动作
                    action_name = action.get('action_name', '')
                    action_params = action.get('action_params', {})
                    
                    # 获取玩家角色
                    player_role = None
                    for player in current_state.get('players', []):
                        if player.get('player_id') == player_id:
                            player_role = player.get('original_role')
                            break
                            
                    if not player_role:
                        action_obj = create_no_action(player_id)
                    else:
                        # 创建夜晚动作，使用**action_params将参数作为关键字参数传递
                        action_obj = create_night_action(player_id, player_role, action_name, **action_params)
                        
                elif action_type == 'DAY_SPEECH':
                    # 白天发言
                    speech_type = action.get('speech_type', 'GENERAL')
                    content = action.get('content', '')
                    action_obj = create_speech(player_id, speech_type, content=content)
                    
                elif action_type == 'VOTE':
                    # 投票
                    target_id = action.get('target_id', 0)
                    action_obj = create_vote(player_id, target_id)
                    
                else:
                    # 默认无动作
                    action_obj = create_no_action(player_id)
            else:
                # 默认无动作
                action_obj = create_no_action(player_id)
        
        # 直接执行动作
        result = env._execute_action(action_obj)
        
        # 更新当前玩家ID
        if isinstance(env.game_state, dict) and 'current_player' in env.game_state:
            next_player = env.game_state['current_player']
        else:
            # 如果无法获取下一个玩家，使用默认的下一个玩家
            next_player = (player_id + 1) % len(env.game_state.get('players', []))
            
        # 更新状态
        new_state = env.game_state
        
        # 构造动作结果
        action_result = {
            "success": result.get('success', True),
            "message": result.get('message', "Action executed successfully"),
            "result": result.get('result', None)
        }
        
        logger.info(f"Action result: {action_result}")
        return action_result, new_state
        
    except Exception as e:
        logger.error(f"Error applying action: {str(e)}")
        logger.exception(e)
        
        # 返回错误结果和原始状态
        error_result = {
            "success": False,
            "message": f"Error applying action: {str(e)}",
            "error": str(e)
        }
        return error_result, current_state


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
    try:
        # 导入agent_factory
        from agents.agent_factory import create_agent as factory_create_agent
        
        logger.info(f"创建代理: 类型={agent_type}, 玩家ID={player_id}")
        
        # 使用agent_factory创建代理
        agent = factory_create_agent(agent_type, player_id=player_id)
        
        # 将状态转换为StateWrapper对象
        wrapped_state = GameManager.StateWrapper(state)
        
        # 初始化代理，如果有initialize方法
        if hasattr(agent, "initialize"):
            logger.info(f"初始化代理: 玩家ID={player_id}")
            # 使用wrapped_state而不是原始state进行初始化
            agent.initialize(wrapped_state)
            
        # 检查代理是否有decide_action方法
        if not hasattr(agent, "decide_action"):
            raise AttributeError(f"代理缺少决策方法: decide_action")
            
        logger.info(f"代理创建成功: 玩家ID={player_id}, 类型={agent_type}")
        return agent
        
    except Exception as e:
        logger.error(f"创建代理失败: {str(e)}")
        logger.exception(e)
        raise 