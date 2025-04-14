# Werewolf Game Testing Guide

## Overview

This document provides a comprehensive guide for testing the Werewolf Game, with a focus on evaluating different agent types and their performance. The testing framework is implemented in `test_agents.py` and supports various test scenarios, agent configurations, and performance metrics.

## Setup Requirements

Before running tests, ensure you have the following dependencies installed:

```bash
pip install torch numpy tqdm fastapi pydantic
```

The testing framework also requires the game environment and agent implementations:

- `werewolf_env`: The Werewolf Game environment
- `agents`: Agent implementations (RandomAgent, HeuristicAgent, RLAgent)

## Running Tests

### Basic Usage

The test script can be run from the command line with various arguments:

```bash
python test_agents.py --agent_type=heuristic --num_games=100 --num_players=6
```

### Command Line Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--agent_type` | Type of agents to test | `compare` | `random`, `heuristic`, `mixed`, `rl`, `compare`, `scenario`, `random_villager_heuristic_werewolf`, `heuristic_villager_random_werewolf`, `random_mix` |
| `--num_games` | Number of games to test | `100` | Any positive integer |
| `--num_players` | Number of players per game | `6` | Any positive integer (recommended: 4-10) |
| `--model_path` | Path to RL model (required for RL agent) | `None` | Path to model file |
| `--render` | Render the game (for visualization) | `False` | Flag (no value needed) |
| `--device` | Computing device | `cuda` if available, else `cpu` | `cuda`, `cpu` |
| `--num_workers` | Number of parallel workers | `1` | Any positive integer |
| `--log_detail` | Log detailed information | `False` | Flag (no value needed) |
| `--test_scenario` | Specific test scenario | `all_scenarios` | `random_vs_heuristic`, `random_villager_heuristic_werewolf`, `heuristic_villager_random_werewolf`, `random_mix`, `all_scenarios` |
| `--seed` | Random seed | `42` | Any integer |
| `--role_config` | Path to role configuration | `None` | Path to JSON file |
| `--use_complete_roles` | Use complete set of roles | `False` | Flag (no value needed) |

### Test Agent Types

#### Random Agent Test

Tests games with all random agents:

```bash
python test_agents.py --agent_type=random --num_games=100
```

#### Heuristic Agent Test

Tests games with all heuristic agents:

```bash
python test_agents.py --agent_type=heuristic --num_games=100
```

#### Mixed Agent Configuration Tests

Tests specific role distribution with different agent types:

```bash
# Test with random villagers and heuristic werewolves
python test_agents.py --agent_type=random_villager_heuristic_werewolf --num_games=100

# Test with heuristic villagers and random werewolves
python test_agents.py --agent_type=heuristic_villager_random_werewolf --num_games=100

# Test with random mix of agent types
python test_agents.py --agent_type=random_mix --num_games=100
```

#### Reinforcement Learning Agent Test

Tests a trained RL agent against heuristic agents:

```bash
python test_agents.py --agent_type=rl --num_games=50 --model_path=models/werewolf_rl_model.pt
```

#### Compare All Agent Types

Runs tests for all agent types and compares their performance:

```bash
python test_agents.py --agent_type=compare --num_games=100
```

### Scenario Testing

Tests specific game scenarios to analyze agent behavior:

```bash
python test_agents.py --agent_type=scenario --test_scenario=random_vs_heuristic --num_games=100
```

### Role Configuration

You can specify a custom role configuration using a JSON file:

```json
{
  "all_roles": [
    "werewolf", "werewolf", 
    "villager", "villager", 
    "seer", "robber", "troublemaker", "insomniac", "minion"
  ],
  "center_card_count": 3,
  "required_player_roles": ["werewolf", "seer"],
  "enforce_required_roles": true
}
```

Then use it in your test:

```bash
python test_agents.py --agent_type=heuristic --role_config=config/custom_roles.json
```

## Test Result Output

The test results are displayed in the console and saved to log files:

```
======= Compare Different Type Agents =======

Agent type               Number of tests Werewolf win rate Villager win rate Average game length
--------------------------------------------------------------------------------
All random agents        100             0.45          0.55          36.21
All heuristic agents     100             0.32          0.68          38.45
Villager random + Werewolf heuristic 100             0.57          0.43          37.82
Villager heuristic + Werewolf random 100             0.28          0.72          37.64
Random combination agents 100             0.42          0.58          37.12

===== Agent Comparison Analysis =====

1. Random agents vs Heuristic agents
Heuristic vs Random agent werewolf win rate difference: -0.13
Heuristic vs Random agent villager win rate difference: 0.13
Conclusion: Heuristic agents perform better in villager role

2. Team comparison analysis
Heuristic werewolf vs Random werewolf win rate difference: 0.29
Heuristic villager vs Random villager win rate difference: 0.29
Conclusion: Heuristic agents perform better in both werewolf and villager roles
```

## Log Files

Test results are saved in the following directories:

```
logs/
├── [config_name]/
│   └── [timestamp]/
│       ├── game_histories/   # Detailed game logs
│       │   └── game_[agent_type]_seed[seed]_[game_id].log
│       ├── summaries/        # Summary statistics
│       │   └── summary_[agent_type]_[num_games]games.csv
│       └── metadata/         # Test metadata
│           └── test_metadata_[agent_type]_[num_games].json
```

### Game History Log Format

Each game log includes:
- Initial game state and role distribution
- Player actions and results for each phase
- Game outcome and statistics

### Summary CSV Format

The summary CSV includes the following fields for each game:
- `game_id`: Unique game identifier
- `agent_type`: Type of agents used
- `seed`: Random seed used
- `winner`: Winning team
- `game_length`: Number of game steps
- `actual_steps`: Actual steps executed
- `night_steps`, `day_steps`, `vote_steps`: Steps in each phase
- `werewolf_count`, `villager_count`: Number of players on each team
- `run_time`: Game execution time in seconds
- `log_file`: Path to the detailed game log

## Analyzing Test Results

### Win Rate Analysis

To analyze which agent type performs better, compare the win rates:
- **Werewolf win rate**: Percentage of games won by the werewolf team
- **Villager win rate**: Percentage of games won by the villager team

### Role Effectiveness

By comparing different scenarios (e.g., random_villager_heuristic_werewolf vs. heuristic_villager_random_werewolf), you can determine which agent type performs better in specific roles.

### Game Length Analysis

The average game length provides insights into how efficiently agents make decisions.

## Advanced Testing Techniques

### GPU Acceleration

For faster testing, use GPU acceleration if available:

```bash
python test_agents.py --agent_type=compare --num_games=500 --device=cuda --num_workers=4
```

### Parallel Testing

To run tests in parallel (multi-threading):

```bash
python test_agents.py --agent_type=compare --num_games=100 --num_workers=8
```

### Custom Logging

For detailed logging of game actions:

```bash
python test_agents.py --agent_type=heuristic --num_games=10 --log_detail
```

## Example Testing Workflow

A typical testing workflow might include:

1. Test baseline performance with random agents:
   ```bash
   python test_agents.py --agent_type=random --num_games=100
   ```

2. Test heuristic agent performance:
   ```bash
   python test_agents.py --agent_type=heuristic --num_games=100
   ```

3. Compare role-specific effectiveness:
   ```bash
   python test_agents.py --agent_type=scenario --test_scenario=all_scenarios --num_games=100
   ```

4. Test a new agent implementation against existing agents:
   ```bash
   python test_agents.py --agent_type=compare --num_games=100
   ```

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `num_workers` or `num_games`
2. **CUDA errors**: Try `--device=cpu`
3. **Long execution time**: Increase `num_workers` or use GPU with `--device=cuda`

### Debug Logs

Error logs are saved in `logs/game_logs.log` and detailed game logs in the `logs/game_histories/` directory.

## Conclusion

The testing framework provides a comprehensive way to evaluate agent performance in the Werewolf Game. By analyzing the results, you can identify strengths and weaknesses of different agent types and improve their strategies.

For more information on agent implementation, refer to the [Agent System Documentation](agent_system.md). 