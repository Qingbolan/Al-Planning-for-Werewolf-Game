"""
Werewolf Game Agent GPU Accelerated Testing Script
Supports multi-threading and GPU for agent performance testing
"""

import argparse
import os
import torch
import numpy as np
import random
import time
import json
import traceback
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple, Optional

from werewolf_env import WerewolfEnv
from werewolf_env.roles import Role
from agents import RandomAgent, HeuristicAgent, create_agent
from models.rl_agent import RLAgent, WerewolfNetwork
from config.default_config import DEFAULT_GAME_CONFIG, ROLE_TEAMS

# 确保日志目录存在
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/game_histories", exist_ok=True)
os.makedirs("logs/summaries", exist_ok=True)

# 设置日志处理器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/game_logs.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Werewolf Game Agent Testing (GPU Accelerated)')
    
    # Testing parameters
    parser.add_argument('--agent_type', type=str, default='compare', 
                        choices=['random', 'heuristic', 'mixed', 'rl', 'compare', 'scenario', 
                                'random_villager_heuristic_werewolf', 'heuristic_villager_random_werewolf', 'random_mix'], 
                        help='Type of agent to test')
    parser.add_argument('--num_games', type=int, default=100, help='Number of games to test')
    parser.add_argument('--num_players', type=int, default=6, help='Number of players')
    parser.add_argument('--model_path', type=str, default=None, help='Path to RL model')
    parser.add_argument('--render', action='store_true', help='Whether to render the game')
    parser.add_argument('--output_file', type=str, default=None, help='Output file path')
    
    # GPU related parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Computing device')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel worker threads')
    
    # Logging related parameters
    parser.add_argument('--log_detail', action='store_true', help='Whether to log detailed information')
    parser.add_argument('--log_file', type=str, default='logs/game_logs.log', help='Log file path')
    
    # Specific scenario testing parameters
    parser.add_argument('--test_scenario', type=str, default='all_scenarios', 
                       choices=['random_vs_heuristic', 'random_villager_heuristic_werewolf', 
                               'heuristic_villager_random_werewolf', 'random_mix', 'all_scenarios'], 
                       help='Specific test scenario')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 添加角色配置参数
    parser.add_argument('--role_config', type=str, default=None, 
                        help='Path to role configuration JSON file')
    parser.add_argument('--use_complete_roles', action='store_true',
                        help='Use complete set of roles instead of just villagers and werewolves')
    
    # 添加时间戳参数,用于标识这次实验
    parser.add_argument('--timestamp', type=str, default=time.strftime('%Y%m%d_%H%M%S'), 
                        help='Timestamp for this experiment run')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_agents(agent_type, env, model_path=None, device="cpu", werewolf_indices=None, villager_indices=None):
    """
    Create a list of agents
    
    Args:
        agent_type: Agent type ('random', 'heuristic', 'mixed', 'rl')
        env: Game environment
        model_path: Path to RL model
        device: Use device ("cpu" or "cuda")
        werewolf_indices: Specified werewolf player indices (optional)
        villager_indices: Specified villager player indices (optional)
        
    Returns:
        list: List of agents
    """
    agents = []
    num_players = env.config['num_players']
    
    # Record werewolf and villager indices
    werewolf_ids = []
    villager_ids = []
    
    # Check if there are pre-specified role indices
    if werewolf_indices is None or villager_indices is None:
        # Assign roles to each player (if not pre-specified)
        roles = env.game_state.roles if hasattr(env.game_state, 'roles') else []
        
        # Determine werewolf and villager indices based on roles in the environment
        if isinstance(roles, list) and len(roles) > 0:
            for i, role in enumerate(roles):
                if i >= num_players:  # 只考虑实际玩家
                    break
                if role == 'werewolf':
                    werewolf_ids.append(i)
                elif role == 'villager':
                    villager_ids.append(i)
        elif isinstance(roles, dict) and len(roles) > 0:
            for player_id, role in roles.items():
                if int(player_id) >= num_players:  # 只考虑实际玩家
                    continue
                if role == 'werewolf':
                    werewolf_ids.append(int(player_id))
                elif role == 'villager':
                    villager_ids.append(int(player_id))
        else:
            # If there is no role information in the environment, allocate roles by fixed ratio
            num_werewolves = max(1, num_players // 3)
            werewolf_ids = random.sample(range(num_players), num_werewolves)
            villager_ids = [i for i in range(num_players) if i not in werewolf_ids]
    else:
        # Use pre-specified role indices
        werewolf_ids = werewolf_indices
        villager_ids = villager_indices
    
    # Create agents
    if agent_type == 'random':
        # All players are random agents
        for i in range(num_players):
            agents.append(RandomAgent(i))
    
    elif agent_type == 'heuristic':
        # All players are heuristic agents
        for i in range(num_players):
            agents.append(HeuristicAgent(i))
    
    elif agent_type == 'mixed':
        # Werewolves are heuristic agents, villagers are random agents
        for i in range(num_players):
            if i in werewolf_ids:
                agents.append(HeuristicAgent(i))
            else:
                agents.append(RandomAgent(i))
    
    elif agent_type == 'random_villager_heuristic_werewolf':
        # Villagers use random agents, werewolves use heuristic agents
        roles = {}
        
        # Ensure we have correct role information
        if hasattr(env.game_state, 'role_map') and env.game_state.role_map:
            # Use role_map if exists
            for player_id, role in env.game_state.role_map.items():
                roles[player_id] = role
        elif hasattr(env.game_state, 'roles') and env.game_state.roles:
            # Or use roles attribute
            for player_id, role in enumerate(env.game_state.roles):
                roles[player_id] = role
                
        # Print role allocation for debugging
        logger.info(f"Role allocation: {roles}")
        
        for i in range(num_players):
            # Get player role
            role = roles.get(i)
            team = ROLE_TEAMS.get(role, None) if role else None
            
            # Record allocation
            if team == 'werewolf' or i in werewolf_ids:  # Werewolf team uses heuristic
                agents.append(HeuristicAgent(i))
                logger.info(f"Player {i} is in werewolf team, using heuristic agent")
            else:  # Villager team uses random
                agents.append(RandomAgent(i))
                logger.info(f"Player {i} is in villager team, using random agent")
                
    elif agent_type == 'heuristic_villager_random_werewolf':
        # Villagers use heuristic agents, werewolves use random agents
        roles = {}
        
        # Ensure we have correct role information
        if hasattr(env.game_state, 'role_map') and env.game_state.role_map:
            # Use role_map if exists
            for player_id, role in env.game_state.role_map.items():
                roles[player_id] = role
        elif hasattr(env.game_state, 'roles') and env.game_state.roles:
            # Or use roles attribute
            for player_id, role in enumerate(env.game_state.roles):
                roles[player_id] = role
                
        # Print role allocation for debugging
        logger.info(f"Role allocation: {roles}")
        
        for i in range(num_players):
            # Get player role
            role = roles.get(i)
            team = ROLE_TEAMS.get(role, None) if role else None
            
            if team == 'werewolf' or i in werewolf_ids:  # Werewolf team uses random
                agents.append(RandomAgent(i))
                logger.info(f"Player {i} is in werewolf team, using random agent")
            else:  # Villager team uses heuristic
                agents.append(HeuristicAgent(i))
                logger.info(f"Player {i} is in villager team, using heuristic agent")
                
    elif agent_type == 'random_mix':
        # Each player randomly assigns agent type
        for i in range(num_players):
            if random.random() < 0.5:  # 50% probability to choose random or heuristic
                agents.append(RandomAgent(i))
                logger.info(f"Player {i} randomly assigned to random agent")
            else:
                agents.append(HeuristicAgent(i))
                logger.info(f"Player {i} randomly assigned to heuristic agent")
    
    elif agent_type == 'rl':
        # Select one player as RL agent, others are heuristic agents
        if not model_path:
            raise ValueError("RL agent needs model path")
        
        rl_index = random.randint(0, num_players - 1)
        
        for i in range(num_players):
            if i == rl_index:
                # Create RL agent and load model
                model = WerewolfNetwork(
                    observation_dim=128,
                    action_dim=env.action_space.n,
                    num_players=num_players
                ).to(device)
                
                try:
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    logger.info(f"Successfully loaded model: {model_path}")
                    agents.append(RLAgent(i, model=model, device=device))
                except Exception as e:
                    logger.error(f"Error loading model: {e}")
                    # Fallback to heuristic agent
                    agents.append(HeuristicAgent(i))
            else:
                # Other players use heuristic agents
                agents.append(HeuristicAgent(i))
    
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    return agents


def setup_logging_directories(config_name, timestamp):
    """设置基于配置名称和时间戳的日志目录结构
    
    Args:
        config_name: 配置文件名称 (不含路径和扩展名)
        timestamp: 时间戳
        
    Returns:
        dict: 包含各种日志路径的字典
    """
    # 如果配置名称为None，使用'default'作为名称
    if config_name is None:
        config_name = 'default'
    else:
        # 如果是路径,只保留文件名(不含扩展名)
        config_name = os.path.basename(config_name)
        config_name = os.path.splitext(config_name)[0]
    
    # 创建基本目录结构
    base_dir = f"logs/{config_name}/{timestamp}"
    dirs = {
        'base': base_dir,
        'game_histories': f"{base_dir}/game_histories",
        'summaries': f"{base_dir}/summaries",
        'metadata': f"{base_dir}/metadata",
    }
    
    # 创建所有目录
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # 设置主日志文件
    main_log_path = f"{base_dir}/game_logs.log"
    
    # 更新日志处理器
    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logging.root.removeHandler(handler)
    
    # 添加新的文件处理器
    file_handler = logging.FileHandler(main_log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.root.addHandler(file_handler)
    
    # 返回路径字典
    return dirs


def run_game_worker(agent_type, num_players=6, model_path=None, render=False, device="cpu", seed=None, env_config=None, agents=None, log_dirs=None):
    """Run single game worker function
    
    Args:
        agent_type: Agent type
        num_players: Number of players
        model_path: Model path
        render: Whether to render
        device: Computing device
        seed: Random seed
        env_config: Environment configuration
        agents: Pre-defined agent list
        log_dirs: 日志目录字典
        
    Returns:
        dict: Game result
    """
    start_time = time.time()
    result = {
        'error': None,
        'winner': None,
        'game_length': 0
    }
    
    # 创建独立的游戏日志记录器
    game_id = int(time.time() * 1000) % 10000000  # 生成唯一的游戏ID
    
    # 使用新的日志目录结构
    if log_dirs and 'game_histories' in log_dirs:
        game_log_path = f"{log_dirs['game_histories']}/game_{agent_type}_seed{seed}_{game_id}.log"
    else:
        # 向后兼容
        game_log_path = f"logs/game_histories/game_{agent_type}_seed{seed}_{game_id}.log"
    
    os.makedirs(os.path.dirname(game_log_path), exist_ok=True)
    
    # 创建文件处理器
    game_log_handler = logging.FileHandler(game_log_path)
    game_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 创建游戏专用logger
    game_logger = logging.getLogger(f"game_{game_id}")
    game_logger.setLevel(logging.INFO)
    game_logger.addHandler(game_log_handler)
    game_logger.propagate = False  # 避免日志重复输出
    
    # 记录游戏初始信息
    game_logger.info(f"===== GAME START =====")
    game_logger.info(f"Agent Type: {agent_type}")
    game_logger.info(f"Num Players: {num_players}")
    game_logger.info(f"Seed: {seed}")
    game_logger.info(f"Device: {device}")
    
    try:
        # Set random seed
        if seed is not None:
            set_seed(seed)
        
        # Create environment configuration
        if env_config is None:
            env_config = DEFAULT_GAME_CONFIG.copy()
            env_config.update({
                'num_players': num_players,
                'max_speech_rounds': 3,  # 固定发言轮数为3轮
            })
            
            # Adjust configuration based on agent_type
            if "werewolf" in agent_type or "villager" in agent_type:
                # 如果是特定角色测试，确保角色分配固定
                # 计算狼人数量，通常是玩家总数的1/3
                num_werewolves = max(1, num_players // 3)
                # 计算村民数量
                num_villagers = num_players - num_werewolves
                
                # 创建明确的角色列表
                roles = ['werewolf'] * num_werewolves + ['villager'] * num_villagers
                # 打乱角色列表
                random.shuffle(roles)
                
                # 更新环境配置
                env_config['roles'] = roles
                game_logger.info(f"Created fixed role configuration for test scenario {agent_type}: {roles}")
        
        # Create environment
        env = WerewolfEnv(config=env_config)
        obs, info = env.reset()
        
        # Detailed logging of initial game state
        game_logger.info("========== Game Start ==========")
        
        # Record role allocation
        player_roles = []
        for i in range(num_players):
            if hasattr(env.game_state, 'players') and i < len(env.game_state.players):
                role = env.game_state.players[i]['original_role']
                player_roles.append(role)
                game_logger.info(f"Player {i} assigned role: {role}")
        
        # Record center card roles
        if hasattr(env.game_state, 'center_cards'):
            game_logger.info(f"Center card roles: {env.game_state.center_cards}")
        
        # Record important environment state information
        if hasattr(env.game_state, 'roles'):
            game_logger.info(f"Game role allocation: {env.game_state.roles}")
        if hasattr(env.game_state, 'role_map'):
            game_logger.info(f"Game role mapping: {env.game_state.role_map}")
        
        # Get werewolf and villager indices for agent creation
        werewolf_indices = []
        villager_indices = []
        
        # Try to get role information from game state
        if hasattr(env.game_state, 'role_map') and env.game_state.role_map:
            for player_id, role in env.game_state.role_map.items():
                if int(player_id) >= num_players:  # 只考虑实际玩家
                    continue
                if 'werewolf' in role:
                    werewolf_indices.append(int(player_id))
                else:
                    villager_indices.append(int(player_id))
        elif hasattr(env.game_state, 'roles') and isinstance(env.game_state.roles, list):
            for i, role in enumerate(env.game_state.roles):
                if i >= num_players:  # 只考虑实际玩家
                    break
                if 'werewolf' in role:
                    werewolf_indices.append(i)
                else:
                    villager_indices.append(i)
        elif hasattr(env.game_state, 'players'):
            for i, player in enumerate(env.game_state.players):
                role = player.get('original_role', '')
                if 'werewolf' in role:
                    werewolf_indices.append(i)
                else:
                    villager_indices.append(i)
        else:
            # Use default allocation
            num_werewolves = max(1, num_players // 3)
            werewolf_indices = random.sample(range(num_players), num_werewolves)
            villager_indices = [i for i in range(num_players) if i not in werewolf_indices]
            
        game_logger.info(f"Werewolf player indices: {werewolf_indices}")
        game_logger.info(f"Villager player indices: {villager_indices}")
        
        # Create agents
        if agents is None:
            agents = create_agents(agent_type, env, model_path, device, werewolf_indices, villager_indices)
        
        # Record agent type allocation
        agent_types = [type(agent).__name__ for agent in agents]
        game_logger.info(f"Agent type allocation: {agent_types}")
        
        # Initialize agents
        for i, agent in enumerate(agents):
            if hasattr(agent, 'initialize'):
                agent.initialize(env.game_state)
        
        # Set step limit and invalid action count
        max_steps = 100  # Set reasonable maximum steps to avoid infinite loops
        consecutive_invalid_actions = 0
        max_invalid_actions = 20  # Maximum number of consecutive invalid actions
        
        # Record game process data
        game_log = {
            'phases': [],
            'actions': [],
            'speeches': [],
            'votes': {},
            'step_by_phase': {'night': 0, 'day': 0, 'vote': 0},  # 按阶段记录步骤
        }
        
        # Game loop
        terminated = False
        truncated = False
        step = 0
        
        game_logger.info("\n----- Game Starting -----\n")
        
        while not (terminated or truncated) and step < max_steps:
            player_idx = env.current_player_id
            
            # Ensure player index is valid
            if player_idx < 0 or player_idx >= len(agents):
                error_msg = f"Invalid player index: {player_idx}, index range should be 0-{len(agents)-1}"
                game_logger.error(error_msg)
                logger.error(error_msg)  # 同时记录到主日志
                break
            
            # Get current agent
            agent = agents[player_idx]
            
            # Record current phase and player
            current_phase = env.game_state.phase
            # 记录各阶段的步骤数
            if current_phase in game_log['step_by_phase']:
                game_log['step_by_phase'][current_phase] += 1
                
            if len(game_log['phases']) == 0 or game_log['phases'][-1] != current_phase:
                game_log['phases'].append(current_phase)
                game_logger.info(f"\n=== Entering {current_phase} phase ===\n")
            
            # Get observation
            obs = env.game_state.get_observation(player_idx)
            
            # Agent chooses action
            try:
                action = agent.act(obs)
                
                # 创建自定义的log_action方法，将日志输出到游戏专用logger
                def log_agent_action(agent_action):
                    """记录代理动作到专用日志文件"""
                    action_type = agent_action.action_type.name if hasattr(agent_action, 'action_type') else "UNKNOWN"
                    
                    game_logger.info(f"Step {step}: Player {player_idx} ({agent.current_role}) performs {action_type}")
                    
                    if action_type == "NIGHT_ACTION" and hasattr(agent_action, 'action_name'):
                        game_logger.info(f"  Night action: {agent_action.action_name}")
                        if hasattr(agent_action, 'action_params'):
                            game_logger.info(f"  Parameters: {agent_action.action_params}")
                    
                    elif action_type == "DAY_SPEECH" and hasattr(agent_action, 'speech_type'):
                        speech_type = agent_action.speech_type
                        content = agent_action.content if hasattr(agent_action, 'content') else {}
                        game_logger.info(f"  Speech type: {speech_type}")
                        game_logger.info(f"  Content: {content.get('text', '')}")
                        
                    elif action_type == "VOTE" and hasattr(agent_action, 'target_id'):
                        game_logger.info(f"  Vote target: Player {agent_action.target_id}")
                
                # 记录动作
                log_agent_action(action)
                
                # Record action details (already logged through agent's log_action)
                if hasattr(action, 'action_type'):
                    action_type = action.action_type.name
                    
                    if action_type == 'DAY_SPEECH':
                        if hasattr(action, 'content'):
                            speech_content = action.content
                            game_log['speeches'].append({
                                'player_id': player_idx,
                                'content': speech_content,
                                'phase': current_phase
                            })
                    
                    elif action_type == 'VOTE':
                        if hasattr(action, 'target_id'):
                            game_log['votes'][player_idx] = action.target_id
                    
                    game_log['actions'].append({
                        'step': step,
                        'player_id': player_idx,
                        'action_type': action_type,
                        'phase': current_phase
                    })
                    
            except Exception as e:
                error_msg = f"Agent action selection failed: {e}"
                game_logger.error(error_msg)
                logger.error(error_msg)
                traceback.print_exc()
                result['error'] = error_msg
                break
            
            # Execute action
            try:
                next_obs, reward, terminated, truncated, info = env.step(action)
                game_logger.info(f"  Result: {info}")
                if reward != 0:
                    game_logger.info(f"  Reward: {reward}")
            except Exception as e:
                error_msg = f"Environment failed to execute action: {e}"
                game_logger.error(error_msg)
                logger.error(error_msg)
                traceback.print_exc()
                result['error'] = error_msg
                break
                
            # Check if action is valid
            if 'success' in info and not info['success']:
                consecutive_invalid_actions += 1
                game_logger.warning(f"Invalid action detected. Consecutive invalid actions: {consecutive_invalid_actions}")
                if consecutive_invalid_actions >= max_invalid_actions:
                    warning_msg = f"Consecutive invalid actions reached limit {max_invalid_actions}, game terminated early"
                    game_logger.warning(warning_msg)
                    logger.warning(warning_msg)
                    break
            else:
                consecutive_invalid_actions = 0
            
            # Render game
            if render:
                rendered_output = env.render()
                if rendered_output:
                    game_logger.info(f"Game display:\n{rendered_output}")
                
            # Increase step
            step += 1
            
        # 计算标准化的游戏长度：
        # 夜晚阶段：每个玩家一步（有夜晚动作的角色）
        night_steps = len([i for i in range(num_players) if agents[i].current_role in ['werewolf', 'seer', 'robber', 'troublemaker', 'insomniac', 'minion']])
        # 白天阶段：每个玩家3轮发言，每轮一步
        day_steps = num_players * 3  # 3轮发言
        # 投票阶段：每个玩家一步
        vote_steps = num_players
        # 总步数
        standardized_steps = night_steps + day_steps + vote_steps
        
        result['game_length'] = standardized_steps
        result['actual_step_count'] = step  # 记录实际步数作为参考
        result['step_by_phase'] = game_log['step_by_phase']  # 记录每个阶段的步数
        
        # Collect game statistics
        if not result['error']:
            # Get winner
            if terminated:
                # Game ended naturally
                game_result = env.game_state.game_result
                if game_result == 'werewolf':
                    result['winner'] = "werewolf"
                    # Record detailed information
                    game_logger.info(f"\nGame over: Werewolf team wins! Game steps: {step}")
                    
                elif game_result == 'villager':
                    result['winner'] = "villager"
                    # Record detailed information
                    game_logger.info(f"\nGame over: Villager team wins! Game steps: {step}")
                    
                else:
                    result['winner'] = "unknown"
                    game_logger.warning(f"Game over, but winner is unclear: {game_result}")
            else:
                result['winner'] = "unknown"
                game_logger.warning("Game did not end normally, unable to determine winner")
            
            # Record final results
            game_logger.info("\n----- Game Statistics -----")
            
            # Record total game steps
            game_logger.info(f"Total game steps: {step}")
            
            # Record steps per phase
            phase_counts = {phase: game_log['step_by_phase'][phase] for phase in game_log['step_by_phase']}
            game_logger.info(f"Steps per phase: {phase_counts}")
            
            # Record voting information
            if game_log['votes']:
                game_logger.info("Voting results:")
                for voter, target in game_log['votes'].items():
                    voter_role = env.game_state.players[voter]['current_role'] if hasattr(env.game_state, 'players') else "unknown"
                    target_role = env.game_state.players[target]['current_role'] if hasattr(env.game_state, 'players') else "unknown"
                    game_logger.info(f"  Player {voter}({voter_role}) voted for Player {target}({target_role})")
                
            # Record final role and state for each player
            if hasattr(env.game_state, 'roles'):
                game_logger.info(f"Role allocation: {env.game_state.roles}")
                
            # Record agent performance
            game_logger.info("\nFinal player information:")
            for i, agent in enumerate(agents):
                agent_type = type(agent).__name__
                
                # Detailed player information
                if hasattr(env.game_state, 'players') and i < len(env.game_state.players):
                    player_data = env.game_state.players[i]
                    player_role = player_data.get('current_role', "unknown")
                    original_role = player_data.get('original_role', "unknown")
                    
                    # Ensure correct team information from config
                    player_team = ROLE_TEAMS.get(player_role, "unknown")
                    
                    game_logger.info(f"Player {i} - Original role: {original_role}, Current role: {player_role}, " 
                               f"Team: {player_team}, Agent type: {agent_type}")
                else:
                    player_role = env.game_state.get_player_role(i) if hasattr(env.game_state, 'get_player_role') else "unknown"
                    player_team = ROLE_TEAMS.get(player_role, "unknown") if player_role != "unknown" else "unknown"
                    game_logger.info(f"Player {i} - Role: {player_role}, Team: {player_team}, Agent type: {agent_type}")
                    
            game_logger.info("========== Game End ==========")
            game_logger.info(f"Game log saved to: {game_log_path}")
                
    except Exception as e:
        error_msg = f"Game execution error: {e}"
        game_logger.error(error_msg)
        logger.error(error_msg)
        traceback.print_exc()
        result['error'] = error_msg
    
    # 记录运行时间
    run_time = time.time() - start_time
    result['run_time'] = run_time
    game_logger.info(f"Total run time: {run_time:.2f} seconds")
    
    # 关闭游戏日志处理器
    for handler in game_logger.handlers:
        handler.close()
    game_logger.handlers = []
    
    # 记录日志文件路径
    result['log_file'] = game_log_path
    
    return result


def test_agent_type(agent_type, num_games=100, num_players=6, model_path=None, render=False, 
                   device="cpu", num_workers=1, random_seed=42, role_config=None, use_complete_roles=False, log_dirs=None):
    """Test specific type of agent
    
    Args:
        agent_type: Agent type ('random', 'heuristic', 'mixed', 'rl')
        num_games: Number of test games
        num_players: Number of players per game
        model_path: Path to RL model
        render: Whether to render game screen
        device: Computing device
        num_workers: Number of parallel worker threads
        random_seed: Random seed
        role_config: Path to role configuration file
        use_complete_roles: Whether to use complete set of roles
        log_dirs: 日志目录字典
        
    Returns:
        dict: Test result statistics
    """
    logger.info(f"Starting test for {agent_type} agent, {num_games} games total")
    logger.info(f"Player number: {num_players}, Computing device: {device}, Parallel threads: {num_workers}")
    
    # 创建汇总日志CSV文件
    if log_dirs and 'summaries' in log_dirs:
        summary_log_path = f"{log_dirs['summaries']}/summary_{agent_type}_{num_games}games.csv"
    else:
        # 向后兼容
        summary_log_path = f"logs/summary_{agent_type}_{num_games}games_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(summary_log_path, 'w') as summary_file:
        # 写入CSV头
        summary_file.write("game_id,agent_type,seed,winner,game_length,actual_steps,night_steps,day_steps,vote_steps,werewolf_count,villager_count,run_time,log_file\n")
    
    # Set total random seed
    set_seed(random_seed)
    
    # 创建一个基础环境配置，确保所有游戏使用相同的配置
    base_env_config = DEFAULT_GAME_CONFIG.copy()
    base_env_config.update({
        'num_players': num_players,
        'max_speech_rounds': 3,  # 固定发言轮数为3轮
    })
    
    # 加载角色配置或使用完整角色集
    if role_config:
        try:
            with open(role_config, 'r') as f:
                role_settings = json.load(f)
            logger.info(f"Loaded role configuration from {role_config}")
            
            # 获取所有角色和中央牌数量
            all_roles = role_settings.get('all_roles', [])
            center_card_count = role_settings.get('center_card_count', 3)
            
            # 检查是否有必须分配给玩家的角色
            required_player_roles = role_settings.get('required_player_roles', [])
            enforce_required_roles = role_settings.get('enforce_required_roles', False)
            
            if enforce_required_roles and required_player_roles:
                # 如果启用强制分配必要角色
                logger.info(f"Enforcing required player roles: {required_player_roles}")
                
                # 设置环境配置
                base_env_config['roles'] = all_roles
                base_env_config['center_card_count'] = center_card_count
                base_env_config['required_player_roles'] = required_player_roles
                base_env_config['enforce_required_roles'] = True
                
                logger.info(f"Using role configuration with required player roles: {required_player_roles}")
            else:
                # 常规角色配置
                base_env_config['roles'] = all_roles
                base_env_config['center_card_count'] = center_card_count
                logger.info(f"Using role configuration: {all_roles} with {center_card_count} center cards")
        except Exception as e:
            logger.error(f"Failed to load role configuration: {e}")
            logger.info("Falling back to default role configuration")
            use_complete_roles = True  # 回退到完整角色集配置
    
    # 使用完整角色集（预设的标准角色分配）
    if use_complete_roles:
        # 使用完整的标准角色集（与图片一致）
        all_roles = [
            'villager', 'villager',  # 2个村民
            'werewolf', 'werewolf',  # 2个狼人
            'minion',                # 1个爪牙
            'seer',                  # 1个预言家
            'troublemaker',          # 1个捣蛋鬼
            'robber',                # 1个强盗
            'insomniac'              # 1个失眠者
        ]
        # 计算中央牌数量 (总角色数 - 玩家数)
        center_card_count = max(0, len(all_roles) - num_players)
        base_env_config['roles'] = all_roles
        base_env_config['center_card_count'] = center_card_count
        logger.info(f"Using complete role set: {all_roles} with {center_card_count} center cards")
    # 如果没有特殊配置，使用简单的村民/狼人分配
    elif not role_config:
        # 配置固定的狼人数量，避免随机性
        num_werewolves = max(1, num_players // 3)
        num_villagers = num_players - num_werewolves
        
        # 创建固定的角色列表，但每局游戏会使用不同的随机种子打乱分配
        base_roles = ['werewolf'] * num_werewolves + ['villager'] * num_villagers
        base_env_config['roles'] = base_roles.copy()  # 使用复制避免引用问题
        base_env_config['center_card_count'] = 0  # 没有中央牌
        logger.info(f"Using simple role configuration: {num_werewolves} werewolves, {num_villagers} villagers, 0 center cards")
    
    # 手动调整角色顺序，使关键角色更可能分配给玩家
    if role_config and 'simple_roles' in role_config:
        # 通过使用多个重复的必要角色加大它们被分配给玩家的概率
        all_roles = base_env_config.get('roles', [])
        # 确保狼人和爪牙出现在列表前面，增加它们被分配给玩家的概率
        critical_roles = []
        other_roles = []
        for role in all_roles:
            if role in ['werewolf', 'minion']:
                critical_roles.append(role)
            else:
                other_roles.append(role)
        # 重新排序角色列表
        base_env_config['roles'] = critical_roles + other_roles
        logger.info(f"Reordered roles to prioritize critical roles: {base_env_config['roles']}")
    
    # Prepare argument list
    args_list = []
    for i in range(num_games):
        # 为每局游戏使用不同的随机种子，但角色分配规则不变
        game_seed = random_seed + i
        game_env_config = base_env_config.copy()
        
        args_list.append((agent_type, num_players, model_path, render and i == 0, device, game_seed, game_env_config, None, log_dirs))
        
    # Run games
    start_time = time.time()
    results = []
    error_count = 0
    
    if num_workers > 1:
        # Multi-threaded run
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Create Future objects
            futures = []
            for args in args_list:
                future = executor.submit(run_game_worker, *args)
                futures.append(future)
                
            # Process results
            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), 
                                           desc=f"Testing {agent_type} agent")):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 写入汇总日志
                    append_result_to_summary(summary_log_path, result, agent_type)
                    
                    if result['error']:
                        error_count += 1
                        logger.warning(f"Game {i} error: {result['error']}")
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error processing game {i} result: {e}")
                    traceback.print_exc()
    else:
        # Single-threaded run
        for i, args in enumerate(tqdm(args_list, desc=f"Testing {agent_type} agent")):
            try:
                result = run_game_worker(*args)
                results.append(result)
                
                # 写入汇总日志
                append_result_to_summary(summary_log_path, result, agent_type)
                
                if result['error']:
                    error_count += 1
                    logger.warning(f"Game {i} error: {result['error']}")
            except Exception as e:
                error_count += 1
                logger.error(f"Game {i} execution error: {e}")
                traceback.print_exc()
    
    # Calculate statistics
    werewolf_wins = 0
    villager_wins = 0
    game_lengths = []
    
    for result in results:
        if result['winner'] == 'werewolf':
            werewolf_wins += 1
        elif result['winner'] == 'villager':
            villager_wins += 1
            
        game_lengths.append(result['game_length'])
    
    # Calculate final statistics
    total_games = len(results)
    if total_games > 0:
        werewolf_win_rate = werewolf_wins / total_games
        villager_win_rate = villager_wins / total_games
        avg_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0
    else:
        werewolf_win_rate = villager_win_rate = avg_game_length = 0
    
    # Record results
    logger.info(f"Test completed, time: {time.time() - start_time:.2f} seconds")
    logger.info(f"Total games: {total_games}, Valid games: {total_games - error_count}")
    logger.info(f"Werewolf win rate: {werewolf_win_rate:.4f}")
    logger.info(f"Villager win rate: {villager_win_rate:.4f}")
    logger.info(f"Average game length: {avg_game_length:.2f}")
    logger.info(f"Error count: {error_count}")
    logger.info(f"Summary log saved to: {summary_log_path}")
    
    # Return statistics results
    return {
        "agent_type": agent_type,
        "num_games": total_games,
        "werewolf_win_rate": werewolf_win_rate,
        "villager_win_rate": villager_win_rate,
        "avg_game_length": avg_game_length,
        "error_count": error_count,
        "summary_log": summary_log_path
    }


def append_result_to_summary(summary_path, result, agent_type):
    """将单个游戏结果追加到汇总日志中
    
    Args:
        summary_path: 汇总日志路径
        result: 游戏结果字典
        agent_type: 代理类型
    """
    # 从日志文件路径提取游戏ID
    log_file = result.get('log_file', '')
    game_id = 'unknown'
    if log_file:
        try:
            game_id = log_file.split('_')[-1].split('.')[0]
        except:
            pass
    
    # 提取步骤信息
    night_steps = result.get('step_by_phase', {}).get('night', 0)
    day_steps = result.get('step_by_phase', {}).get('day', 0)
    vote_steps = result.get('step_by_phase', {}).get('vote', 0)
    
    # 从结果中提取信息
    winner = result.get('winner', 'unknown')
    game_length = result.get('game_length', 0)
    actual_steps = result.get('actual_step_count', 0)
    run_time = result.get('run_time', 0)
    
    # 狼人数量为玩家数量的1/3（四舍五入）
    num_players = max(6, game_length // 4)
    werewolf_count = max(1, num_players // 3)
    villager_count = num_players - werewolf_count
    
    # 获取种子信息
    seed = 'unknown'
    if log_file:
        try:
            # 格式：game_{agent_type}_seed{seed}_{game_id}.log
            seed_part = log_file.split('_seed')[1].split('_')[0]
            seed = seed_part
        except:
            pass
    
    # 创建CSV行
    row = f"{game_id},{agent_type},{seed},{winner},{game_length},{actual_steps},{night_steps},{day_steps},{vote_steps},{werewolf_count},{villager_count},{run_time:.2f},{log_file}\n"
    
    # 追加到汇总文件
    with open(summary_path, 'a') as summary_file:
        summary_file.write(row)


def test_specific_scenarios(num_games, num_players, device="cpu", num_workers=4, render=False, scenario='all_scenarios', role_config=None, use_complete_roles=False, log_dirs=None):
    """Test specific scenario
    
    Test different agent combinations for specified scenarios:
    
    Args:
        num_games: Number of tests per scenario
        num_players: Number of players
        device: Computing device
        num_workers: Number of parallel worker threads
        render: Whether to render
        scenario: Specified test scenario
        role_config: Path to role configuration file
        use_complete_roles: Whether to use complete set of roles
        log_dirs: 日志目录字典
        
    Returns:
        Test result statistics
    """
    print("\n======= Test Specific Scenario =======")
    
    # Set werewolf number
    num_werewolves = max(1, num_players // 3)
    results = {}
    
    # Test based on specified scenario
    if scenario == 'all_scenarios' or scenario == 'random_vs_heuristic':
        # Test random vs heuristic
        print("\nTest: All random vs All heuristic agents")
        random_stats = test_agent_type('random', num_games, num_players, None, render, device, num_workers, role_config=role_config, use_complete_roles=use_complete_roles, log_dirs=log_dirs)
        heuristic_stats = test_agent_type('heuristic', num_games, num_players, None, render, device, num_workers, role_config=role_config, use_complete_roles=use_complete_roles, log_dirs=log_dirs)
        results['random'] = random_stats
        results['heuristic'] = heuristic_stats
    
    if scenario == 'all_scenarios' or scenario == 'random_villager_heuristic_werewolf':
        # Scenario 1: Villagers use random agents, werewolves use heuristic agents
        print("\nScenario: Villagers use random agents, werewolves use heuristic agents")
        scenario1_results = test_agent_type(
            'random_villager_heuristic_werewolf',
            num_games,
            num_players,
            None,  # model_path
            render,
            device,
            num_workers,
            role_config=role_config,
            use_complete_roles=use_complete_roles,
            log_dirs=log_dirs
        )
        results['random_villager_heuristic_werewolf'] = scenario1_results
    
    if scenario == 'all_scenarios' or scenario == 'heuristic_villager_random_werewolf':
        # Scenario 2: Villagers use heuristic agents, werewolves use random agents
        print("\nScenario: Villagers use heuristic agents, werewolves use random agents")
        scenario2_results = test_agent_type(
            'heuristic_villager_random_werewolf',
            num_games,
            num_players,
            None,  # model_path
            render,
            device,
            num_workers,
            role_config=role_config,
            use_complete_roles=use_complete_roles,
            log_dirs=log_dirs
        )
        results['heuristic_villager_random_werewolf'] = scenario2_results
        
    if scenario == 'all_scenarios' or scenario == 'random_mix':
        # Scenario 3: Random combination of agent allocation
        print("\nScenario: Random combination of agent allocation")
        random_mix_results = test_agent_type(
            'random_mix',
            num_games,
            num_players,
            None,  # model_path
            render,
            device,
            num_workers,
            role_config=role_config,
            use_complete_roles=use_complete_roles,
            log_dirs=log_dirs
        )
        results['random_mix'] = random_mix_results
    
    # 创建汇总元数据
    metadata = {
        'scenario': scenario,
        'num_games': num_games,
        'num_players': num_players,
        'device': device,
        'summary_logs': []
    }
    
    # 收集所有汇总日志路径
    for agent_type, result in results.items():
        if 'summary_log' in result:
            metadata['summary_logs'].append(result['summary_log'])
            
    # 保存测试元数据
    if log_dirs and 'metadata' in log_dirs:
        metadata_path = f"{log_dirs['metadata']}/scenario_test_metadata_{scenario}_{num_games}.json"
    else:
        metadata_path = f"logs/scenario_test_metadata_{scenario}_{num_games}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nScenario test metadata saved to {metadata_path}")
    
    return results


def compare_agent_types(num_games, num_players, device="cpu", num_workers=4, render=False, role_config=None, use_complete_roles=False, log_dirs=None):
    """Compare different types of agents
    
    Args:
        num_games: Number of tests per type
        num_players: Number of players
        device: Computing device
        num_workers: Number of parallel worker threads
        render: Whether to render
        role_config: Path to role configuration file
        use_complete_roles: Whether to use complete set of roles
        log_dirs: 日志目录字典
        
    Returns:
        Comparison result
    """
    print("\n======= Compare Different Type Agents =======")
    
    results = {}
    
    # 1. Test all players are random agents
    print("\nTest scenario 1: All players are random agents")
    random_stats = test_agent_type('random', num_games, num_players, None, render, device, num_workers, role_config=role_config, use_complete_roles=use_complete_roles, log_dirs=log_dirs)
    results['random'] = random_stats
    
    # 2. Test all players are heuristic agents
    print("\nTest scenario 2: All players are heuristic agents")
    heuristic_stats = test_agent_type('heuristic', num_games, num_players, None, render, device, num_workers, role_config=role_config, use_complete_roles=use_complete_roles, log_dirs=log_dirs)
    results['heuristic'] = heuristic_stats
    
    # 3. Test villagers use random agents, werewolves use heuristic agents
    print("\nTest scenario 3: Villagers use random agents, werewolves use heuristic agents")
    random_villager_heuristic_werewolf_stats = test_agent_type('random_villager_heuristic_werewolf', num_games, num_players, 
                                                      None, render, device, num_workers, role_config=role_config, use_complete_roles=use_complete_roles, log_dirs=log_dirs)
    results['random_villager_heuristic_werewolf'] = random_villager_heuristic_werewolf_stats
    
    # 4. Test villagers use heuristic agents, werewolves use random agents
    print("\nTest scenario 4: Villagers use heuristic agents, werewolves use random agents")
    heuristic_villager_random_werewolf_stats = test_agent_type('heuristic_villager_random_werewolf', num_games, num_players,
                                                     None, render, device, num_workers, role_config=role_config, use_complete_roles=use_complete_roles, log_dirs=log_dirs)
    results['heuristic_villager_random_werewolf'] = heuristic_villager_random_werewolf_stats
    
    # 5. Test any random combination of agent allocation
    print("\nTest scenario 5: Any random combination of agent allocation")
    random_mix_stats = test_agent_type('random_mix', num_games, num_players, None, render, device, num_workers, role_config=role_config, use_complete_roles=use_complete_roles, log_dirs=log_dirs)
    results['random_mix'] = random_mix_stats
    
    # 创建汇总元数据
    metadata = {
        'comparison_type': 'agent_types',
        'num_games': num_games,
        'num_players': num_players,
        'device': device,
        'summary_logs': []
    }
    
    # 收集所有汇总日志路径
    for agent_type, result in results.items():
        if 'summary_log' in result:
            metadata['summary_logs'].append(result['summary_log'])
            
    # 保存测试元数据
    if log_dirs and 'metadata' in log_dirs:
        metadata_path = f"{log_dirs['metadata']}/compare_agents_metadata_{num_games}.json"
    else:
        metadata_path = f"logs/compare_agents_metadata_{num_games}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nComparison test metadata saved to {metadata_path}")
    
    return results


def test_rl_agent(model_path, num_games, num_players, device="cpu", num_workers=4, render=False, role_config=None, use_complete_roles=False, log_dirs=None):
    """Test RL agent
    
    Args:
        model_path: Model path
        num_games: Number of tests
        num_players: Number of players
        device: Computing device
        num_workers: Number of parallel worker threads
        render: Whether to render
        role_config: Path to role configuration file
        use_complete_roles: Whether to use complete set of roles
        log_dirs: 日志目录字典
        
    Returns:
        Test result
    """
    return test_agent_type('rl', num_games, num_players, model_path, render, device, num_workers, role_config=role_config, use_complete_roles=use_complete_roles, log_dirs=log_dirs)


def print_results(results, title="Test result"):
    """Print test result
    
    Args:
        results: Test result
        title: Title
    """
    print(f"\n======= {title} =======")
    
    if isinstance(results, dict) and 'agent_type' in results:
        # Single agent result
        agent_type = results['agent_type']
        print(f"Agent type: {agent_type}")
        print(f"Number of tests: {results['num_games']}")
        print(f"Werewolf win rate: {results['werewolf_win_rate']:.2f}")
        print(f"Villager win rate: {results['villager_win_rate']:.2f}")
        print(f"Average game length: {results['avg_game_length']:.2f}")
        print(f"Error count: {results['error_count']}")
    
    elif isinstance(results, dict) and 'scenario1_random_villager_heuristic_werewolf' in results:
        # Specific scenario test result
        print("\nScenario 1: Villagers use random agents, werewolves use heuristic agents")
        scenario1 = results['scenario1_random_villager_heuristic_werewolf']
        print(f"Number of tests: {scenario1['num_games']}")
        print(f"Werewolf win rate: {scenario1['werewolf_win_rate']:.2f} (Heuristic agent)")
        print(f"Villager win rate: {scenario1['villager_win_rate']:.2f} (Random agent)")
        print(f"Average game length: {scenario1['avg_game_length']:.2f}")
        
        print("\nScenario 2: Villagers use heuristic agents, werewolves use random agents")
        scenario2 = results['scenario2_heuristic_villager_random_werewolf']
        print(f"Number of tests: {scenario2['num_games']}")
        print(f"Werewolf win rate: {scenario2['werewolf_win_rate']:.2f} (Random agent)")
        print(f"Villager win rate: {scenario2['villager_win_rate']:.2f} (Heuristic agent)")
        print(f"Average game length: {scenario2['avg_game_length']:.2f}")
        
        # Add comparison analysis
        print("\n===== Scenario Comparison Analysis =====")
        werewolf_win_diff = scenario1['werewolf_win_rate'] - scenario2['werewolf_win_rate']
        print(f"Heuristic werewolf vs Random werewolf win rate difference: {werewolf_win_diff:.2f}")
        
        villager_win_diff = scenario2['villager_win_rate'] - scenario1['villager_win_rate']
        print(f"Heuristic villager vs Random villager win rate difference: {villager_win_diff:.2f}")
        
        if werewolf_win_diff > 0 and villager_win_diff > 0:
            print("Conclusion: Heuristic agents perform better in both werewolf and villager roles")
        elif werewolf_win_diff > 0:
            print("Conclusion: Heuristic agents perform better in werewolf role")
        elif villager_win_diff > 0:
            print("Conclusion: Heuristic agents perform better in villager role")
        else:
            print("Conclusion: Random agents show unexpected advantage")
    
    elif isinstance(results, dict):
        # Multiple agent comparison result
        agent_types = list(results.keys())
        
        # Table header
        print(f"{'Agent type':<30} {'Number of tests':<8} {'Werewolf win rate':^10} {'Villager win rate':^10} {'Average game length':^12}")
        print("-" * 80)
        
        # Table content
        for agent_type in agent_types:
            r = results[agent_type]
            # Optimize agent type name display
            display_name = agent_type
            if agent_type == 'random':
                display_name = "All random agents"
            elif agent_type == 'heuristic':
                display_name = "All heuristic agents"
            elif agent_type == 'random_villager_heuristic_werewolf':
                display_name = "Villager random + Werewolf heuristic"
            elif agent_type == 'heuristic_villager_random_werewolf':
                display_name = "Villager heuristic + Werewolf random"
            elif agent_type == 'random_mix':
                display_name = "Random combination agents"
            
            print(f"{display_name:<30} {r['num_games']:<8} {r['werewolf_win_rate']:^10.2f} {r['villager_win_rate']:^10.2f} {r['avg_game_length']:^12.2f}")
        
        # Comparison analysis
        print("\n===== Agent Comparison Analysis =====")
        if 'random' in results and 'heuristic' in results:
            print("\n1. Random agents vs Heuristic agents")
            wolf_win_diff = results['heuristic']['werewolf_win_rate'] - results['random']['werewolf_win_rate']
            village_win_diff = results['heuristic']['villager_win_rate'] - results['random']['villager_win_rate']
            
            print(f"Heuristic vs Random agent werewolf win rate difference: {wolf_win_diff:.2f}")
            print(f"Heuristic vs Random agent villager win rate difference: {village_win_diff:.2f}")
            
            if wolf_win_diff > 0 and village_win_diff > 0:
                print("Conclusion: Heuristic agents overall perform better than Random agents")
            elif wolf_win_diff > 0:
                print("Conclusion: Heuristic agents perform better in werewolf role")
            elif village_win_diff > 0:
                print("Conclusion: Heuristic agents perform better in villager role")
            else:
                print("Conclusion: Random agents show unexpected advantage")
        
        # New analysis: Villager/Werewolf team comparison
        if 'random_villager_heuristic_werewolf' in results and 'heuristic_villager_random_werewolf' in results:
            print("\n2. Team comparison analysis")
            scenario1 = results['random_villager_heuristic_werewolf']
            scenario2 = results['heuristic_villager_random_werewolf']
            
            werewolf_win_diff = scenario1['werewolf_win_rate'] - scenario2['werewolf_win_rate']
            print(f"Heuristic werewolf vs Random werewolf win rate difference: {werewolf_win_diff:.2f}")
            
            villager_win_diff = scenario2['villager_win_rate'] - scenario1['villager_win_rate']
            print(f"Heuristic villager vs Random villager win rate difference: {villager_win_diff:.2f}")
            
            if werewolf_win_diff > 0 and villager_win_diff > 0:
                print("Conclusion: Heuristic agents perform better in both werewolf and villager roles")
            elif werewolf_win_diff > 0:
                print("Conclusion: Heuristic agents perform better in werewolf role")
            elif villager_win_diff > 0:
                print("Conclusion: Heuristic agents perform better in villager role")
            else:
                print("Conclusion: Random agents show unexpected advantage")
        
        # Random combination analysis
        if 'random_mix' in results:
            print("\n3. Random combination agent analysis")
            random_mix = results['random_mix']
            print(f"Random combination agent werewolf win rate: {random_mix['werewolf_win_rate']:.2f}")
            print(f"Random combination agent villager win rate: {random_mix['villager_win_rate']:.2f}")
            
            # Compare with other scenarios
            if 'random' in results and 'heuristic' in results:
                print(f"Werewolf win rate difference compared to all random: {random_mix['werewolf_win_rate'] - results['random']['werewolf_win_rate']:.2f}")
                print(f"Werewolf win rate difference compared to all heuristic: {random_mix['werewolf_win_rate'] - results['heuristic']['werewolf_win_rate']:.2f}")


def main():
    """Main function"""
    try:
        # Parse arguments
        args = parse_args()
        
        # Set random seed
        set_seed(args.seed)
        
        # 创建基于配置文件和时间戳的日志目录结构
        log_dirs = setup_logging_directories(args.role_config, args.timestamp)
        logger.info(f"Log directories setup at: {log_dirs['base']}")
        
        # 初始化结果信息
        start_time = time.time()  # 添加开始时间记录
        test_results = {
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'agent_type': args.agent_type,
            'num_games': args.num_games,
            'num_players': args.num_players,
            'device': args.device,
            'summary_logs': [],
            'log_base_dir': log_dirs['base']
        }
        
        # Based on parameter, select test type
        if args.agent_type == 'compare':
            # Compare different types of agents
            results = compare_agent_types(
                args.num_games, 
                args.num_players, 
                args.device, 
                args.num_workers, 
                args.render,
                role_config=args.role_config,
                use_complete_roles=args.use_complete_roles,
                log_dirs=log_dirs
            )
            # 收集所有汇总日志路径
            if isinstance(results, dict):
                for agent_type, result in results.items():
                    if 'summary_log' in result:
                        test_results['summary_logs'].append(result['summary_log'])
        elif args.agent_type == 'scenario':
            # Test specific scenario
            results = test_specific_scenarios(
                args.num_games, 
                args.num_players, 
                args.device, 
                args.num_workers, 
                args.render,
                args.test_scenario,
                role_config=args.role_config,
                use_complete_roles=args.use_complete_roles,
                log_dirs=log_dirs
            )
            # 收集所有汇总日志路径
            if isinstance(results, dict):
                for scenario, result in results.items():
                    if 'summary_log' in result:
                        test_results['summary_logs'].append(result['summary_log'])
        elif args.agent_type == 'rl' and args.model_path:
            # Test RL agent
            results = test_rl_agent(
                args.model_path, 
                args.num_games, 
                args.num_players, 
                args.device, 
                args.num_workers, 
                args.render,
                role_config=args.role_config,
                use_complete_roles=args.use_complete_roles,
                log_dirs=log_dirs
            )
            if 'summary_log' in results:
                test_results['summary_logs'].append(results['summary_log'])
        else:
            # Test specific type of agent
            results = test_agent_type(
                args.agent_type, 
                args.num_games, 
                args.num_players, 
                args.model_path, 
                args.render,
                args.device,
                args.num_workers,
                args.seed,
                role_config=args.role_config,
                use_complete_roles=args.use_complete_roles,
                log_dirs=log_dirs
            )
            if 'summary_log' in results:
                test_results['summary_logs'].append(results['summary_log'])
        
        # Print results
        print_results(results)
        
        # 记录总结信息
        test_results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        test_results['total_duration'] = time.time() - start_time
        
        # 保存测试元数据
        if 'metadata' in log_dirs:
            metadata_path = f"{log_dirs['metadata']}/test_metadata_{args.agent_type}_{args.num_games}.json"
        else:
            metadata_path = f"logs/test_metadata_{args.agent_type}_{args.num_games}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"\nTest metadata saved to {metadata_path}")
        
        # 打印汇总日志路径
        if test_results['summary_logs']:
            print("\nSummary logs:")
            for log_path in test_results['summary_logs']:
                print(f"- {log_path}")
        
        # Save results to file (if specified)
        if args.output_file:
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output_file}")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("\nDetailed error information:")
        traceback.print_exc()
        
        # Update debug log
        with open("documents/debug_log.md", "a") as f:
            f.write(f"\n\n## Run time error ({time.strftime('%Y-%m-%d %H:%M:%S')})\n\n")
            f.write(f"```\n{traceback.format_exc()}\n```\n")


if __name__ == "__main__":
    main() 