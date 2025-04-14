# Werewolf Game Reinforcement Learning Project

This project implements a reinforcement learning environment and agents for the Werewolf game. The project uses the Gymnasium framework for the game environment and implements reinforcement learning algorithms based on PyTorch to train the agents.

## Project Structure

```
├── agents/               # Agent implementations
│   ├── base_agent.py     # Base agent, random agent, and heuristic agent
├── config/               # Configuration files
│   ├── default_config.py # Default game configuration
├── models/               # Model implementations
│   ├── rl_agent.py       # Reinforcement learning agent and neural network model
├── train/                # Training related code
│   ├── rl_trainer.py     # Reinforcement learning trainer
├── utils/                # Utility functions
│   ├── belief_updater.py # Belief updater
├── werewolf_env/         # Game environment
│   ├── env.py            # Main environment class
│   ├── state.py          # Game state
│   ├── actions.py        # Action definitions
│   ├── roles.py          # Role definitions
├── main.py               # Main game program
├── run_training.py       # Training entry point
├── test_env.py           # Environment testing
├── requirements.txt      # Dependencies
```

## Dependencies

The project depends on the following Python packages:

```
gymnasium>=0.28.1
numpy>=1.24.0
torch>=2.0.0
stable-baselines3>=2.0.0
pettingzoo>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
wandb>=0.15.0
```

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage


### Test Agents

```bash
python test_agents.py --agent_type compare --num_games 1000 --role_config .\config\6_players_roles.json
```

### Training Agents

Start training with the following command:

```bash
python run_training.py --num_episodes 5000 --batch_size 4
```

Main parameters:

- `--num_episodes`: Number of training episodes
- `--batch_size`: Batch size
- `--learning_rate`: Learning rate
- `--gamma`: Discount factor
- `--use_cuda`: Whether to use GPU for training
- `--use_wandb`: Whether to use wandb for experiment logging

More parameters can be viewed with `python run_training.py --help`.

### Continue Training

To continue training from a saved model:

```bash
python run_training.py --continue_training --model_path ./models/saved/model_episode_1000.pt
```

### Custom Training

You can also customize the training process by importing the trainer:

```python
from train import RLTrainer

trainer = RLTrainer(
    num_players=6,
    obs_dim=128,
    action_dim=100,
    learning_rate=0.0003
)

# Start training
trainer.train(num_episodes=5000)
```

## Game Rules

The project implements a simplified version of the Werewolf game with the following roles:

| Role                   | Team          | Ability                                                                                                                      | Goal                                                                                              | Strategy                                                                                                                                                   |
| ---------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Werewolf**     | Werewolf Camp | All werewolves open their eyes at night to see teammates. If only one werewolf in game, can check one card from center pile. | Avoid being voted out. If a villager or non-werewolf receives the most votes, werewolf team wins. | Conceal identity by claiming to be another role; deflect suspicion to villagers.                                                                           |
| **Villager**     | Villager Camp | No special abilities or night actions.                                                                                       | Work with other village team members to identify and vote out werewolves.                         | Observe carefully, participate in discussions, use deductive reasoning to identify werewolves.                                                             |
| **Seer**         | Villager Camp | Can check one other player's card OR two cards from the center pile during night phase.                                      | Use information to help identify werewolves.                                                      | May reveal information to help villagers, but risks becoming a target for werewolves.                                                                      |
| **Robber**       | Villager Camp | Can steal another player's card, swap it with own, and check new identity.                                                   | Use gathered information to help the village, while adapting to potentially new role.             | Can claim knowledge about another player's role (the one they robbed), but must be careful if they've become a werewolf after the swap.                    |
| **Troublemaker** | Villager Camp | Can swap cards between any two other players without looking at them.                                                        | Create confusion among werewolves and gather information from reactions.                          | By announcing which players they switched, can observe reactions and help deduce roles.                                                                    |
| **Insomniac**    | Villager Camp | Wakes up at end of night phase to check final identity to see if it was switched.                                            | Provide confirmed information about at least one player (themselves).                             | Can validate or contradict claims by Robber or Troublemaker if their card was affected.                                                                    |
| **Minion**       | Werewolf Camp | Opens eyes during night to see who werewolves are, but werewolves don't know who minion is.                                  | Protect werewolves and help them win, even at personal cost.                                      | Often claims to be a villager or another role to deflect suspicion from werewolves. If minion is eliminated instead of werewolf, werewolf team still wins. |

The game consists of the following phases:

1. Night Phase: Each role performs their special abilities in order
2. Day Phase: Players discuss and make speeches
3. Voting Phase: All players vote to eliminate one player
4. Resolution Phase: Determine game outcome

## Agents

The project implements three types of agents:

1. Random Agent: Selects actions randomly from valid options
2. Heuristic Agent: Selects actions based on rule-based strategies
3. Reinforcement Learning Agent: Learns strategies using neural network models

## Environment Features

The game environment includes:

1. Multi-agent environment
2. Partially observable state
3. Discrete action space
4. Modified rules:
   - Three rounds of sequential speech during day
   - Single round of voting at night
   - Reversed victory conditions

## Neural Network Model

The reinforcement learning agent uses a neural network model with the following components:

1. Feature extractor
2. Role embedding layer
3. Player embedding layer
4. LSTM layer for processing historical information
5. Policy head for action probability output
6. Value head for state value output

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests to improve this project.

## License

MIT License
