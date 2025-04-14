# Werewolf Game Agent System Documentation

## Overview

This document describes the agent system implemented in the Werewolf Game, including different agent types, their decision-making processes, and how they interact with the game environment. The agent system is designed to simulate player behavior in an AI-driven environment or to provide opponents for human players.

## Agent Architecture

All agents in the system implement a common interface that enables them to:
1. Receive observations from the game environment
2. Process game state information
3. Select and execute valid actions
4. Update their internal state based on game progression

## Agent Types

### 1. Random Agent

The Random Agent represents the simplest AI implementation, making decisions entirely based on random selection from available legal actions.

#### Implementation Details

```python
class RandomAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        self.current_role = None
        
    def initialize(self, game_state):
        # Initialize agent with role information
        self.current_role = game_state.get_player_role(self.player_id)
        
    def act(self, observation):
        # Select a random action from valid actions
        valid_actions = observation.get('valid_actions', [])
        if not valid_actions:
            return None
        return random.choice(valid_actions)
```

#### Characteristics

- **Decision Making**: Completely random selection from available actions
- **Predictability**: Unpredictable behavior that doesn't follow any strategy
- **Effectiveness**: Provides a baseline performance level for comparison
- **Use Case**: Testing basic game mechanics and serving as a benchmark

### 2. Heuristic Agent

The Heuristic Agent employs role-specific strategies and rules to make more intelligent decisions based on available information.

#### Implementation Details

```python
class HeuristicAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        self.current_role = None
        self.knowledge_base = {}
        self.suspicion_levels = {}
        
    def initialize(self, game_state):
        self.current_role = game_state.get_player_role(self.player_id)
        # Initialize knowledge base and suspicion levels
        
    def act(self, observation):
        # Get current phase
        phase = observation.get('phase')
        
        if phase == 'night':
            return self._night_action(observation)
        elif phase == 'day':
            return self._day_action(observation)
        elif phase == 'vote':
            return self._vote_action(observation)
        
        return None
        
    def _night_action(self, observation):
        # Role-specific night actions
        if self.current_role == 'werewolf':
            return self._werewolf_night_action(observation)
        elif self.current_role == 'seer':
            return self._seer_night_action(observation)
        # ... other roles
        
    def _day_action(self, observation):
        # Implement day speech strategy
        # May include claiming roles, accusing others, etc.
        
    def _vote_action(self, observation):
        # Implement voting strategy based on suspicion levels
```

#### Characteristics

- **Decision Making**: Based on predefined heuristics specific to each role
- **Predictability**: More predictable than random agents but with some variability
- **Effectiveness**: Generally better than random agents but limited by fixed rules
- **Use Case**: Providing moderate challenge for human players and testing game balance

### 3. Reinforcement Learning Agent

The RL Agent uses trained neural networks to make decisions based on patterns learned through thousands of game simulations.

#### Implementation Details

```python
class RLAgent:
    def __init__(self, player_id, model=None, device="cpu"):
        self.player_id = player_id
        self.current_role = None
        self.model = model
        self.device = device
        
    def initialize(self, game_state):
        self.current_role = game_state.get_player_role(self.player_id)
        
    def act(self, observation):
        # Convert observation to tensor representation
        state_tensor = self._process_observation(observation)
        
        # Pass through neural network to get action probabilities
        with torch.no_grad():
            action_probs = self.model(state_tensor)
            
        # Select action based on probabilities
        valid_actions = observation.get('valid_actions', [])
        return self._select_action(action_probs, valid_actions)
        
    def _process_observation(self, observation):
        # Convert observation dictionary to tensor format for the neural network
        
    def _select_action(self, action_probs, valid_actions):
        # Select highest probability valid action
```

#### Characteristics

- **Decision Making**: Based on patterns learned from extensive gameplay
- **Predictability**: Can develop nuanced strategies that may be difficult to predict
- **Effectiveness**: Potentially superior to heuristic agents when properly trained
- **Use Case**: Providing challenging opponents and exploring optimal strategies

### 4. Mixed Strategy Agent

The Mixed Strategy Agent combines multiple decision-making approaches, selecting between them based on the game context.

#### Implementation Details

```python
class MixedStrategyAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        self.current_role = None
        self.random_agent = RandomAgent(player_id)
        self.heuristic_agent = HeuristicAgent(player_id)
        
    def initialize(self, game_state):
        self.current_role = game_state.get_player_role(self.player_id)
        self.random_agent.initialize(game_state)
        self.heuristic_agent.initialize(game_state)
        
    def act(self, observation):
        # Determine which strategy to use based on game state
        phase = observation.get('phase')
        
        if phase == 'night':
            # Use heuristic for night actions
            return self.heuristic_agent.act(observation)
        elif phase == 'day':
            # Mix random and heuristic for day speeches
            if random.random() < 0.3:  # 30% chance to use random
                return self.random_agent.act(observation)
            else:
                return self.heuristic_agent.act(observation)
        elif phase == 'vote':
            # Always use heuristic for voting
            return self.heuristic_agent.act(observation)
```

#### Characteristics

- **Decision Making**: Combines multiple strategies adaptively
- **Predictability**: Less predictable due to strategic switching
- **Effectiveness**: Can overcome limitations of individual agent types
- **Use Case**: Creating more human-like behavior with occasional suboptimal decisions

## Agent Decision-Making by Role

### Werewolf Role

#### Night Phase Strategy
- **Random Agent**: Randomly decides whether to look at a center card
- **Heuristic Agent**: 
  - Identifies other werewolves and adds them to allies list
  - If no other werewolves, checks center card to gather information
  - Records knowledge for use in day phase

#### Day Phase Strategy
- **Random Agent**: Makes random speech actions
- **Heuristic Agent**:
  - Claims non-werewolf role (often villager or seer)
  - Avoids accusing other werewolves
  - Strategically accuses players based on observed night actions

#### Vote Phase Strategy
- **Random Agent**: Votes randomly
- **Heuristic Agent**: 
  - Avoids voting for team members
  - Targets players who claimed powerful roles or accused werewolves

### Seer Role

#### Night Phase Strategy
- **Random Agent**: Randomly checks player card or center cards
- **Heuristic Agent**:
  - Prioritizes checking suspected werewolves
  - If no clear suspects, checks center cards to gain information

#### Day Phase Strategy
- **Random Agent**: Makes random speech actions
- **Heuristic Agent**:
  - Usually claims seer role and reveals information
  - Accuses players identified as werewolves
  - Defends players confirmed as villagers

#### Vote Phase Strategy
- **Random Agent**: Votes randomly
- **Heuristic Agent**: Votes for players identified as werewolves during night phase

## Agent Performance Testing

The testing framework allows for systematic evaluation of agent performance:

```python
def test_agent_type(agent_type, num_games=100, num_players=6):
    """Test specific type of agent performance"""
    werewolf_wins = 0
    villager_wins = 0
    
    for i in range(num_games):
        # Create environment
        env = WerewolfEnv(config={'num_players': num_players})
        
        # Create agents
        agents = create_agents(agent_type, env)
        
        # Run game
        result = run_game(env, agents)
        
        # Record result
        if result == 'werewolf':
            werewolf_wins += 1
        elif result == 'villager':
            villager_wins += 1
            
    # Calculate win rates
    werewolf_win_rate = werewolf_wins / num_games
    villager_win_rate = villager_wins / num_games
    
    return {
        'werewolf_win_rate': werewolf_win_rate,
        'villager_win_rate': villager_win_rate
    }
```

## Agent Communication Format

All agents communicate with the game environment using standardized action formats:

### Night Action Format
```json
{
  "action_type": "night_action",
  "player_id": 0,
  "action_name": "werewolf_action",
  "action_params": {
    "view_center_card": true
  }
}
```

### Day Speech Format
```json
{
  "action_type": "day_speech",
  "player_id": 0,
  "speech_type": "claim",
  "content": {
    "text": "I am a villager",
    "role_claim": "villager"
  }
}
```

### Vote Action Format
```json
{
  "action_type": "vote",
  "player_id": 0,
  "target_id": 2
}
```

## Agent Development and Customization

Developers can create custom agents by implementing the base agent interface:

```python
class CustomAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        self.current_role = None
        
    def initialize(self, game_state):
        """Initialize agent with game state information"""
        pass
        
    def act(self, observation):
        """Return an action based on the current observation"""
        pass
        
    def update(self, state, action, next_state, reward):
        """Update internal state based on game progression"""
        pass
```

New agent types can be registered with the agent factory:

```python
def create_agent(agent_type, player_id, **kwargs):
    if agent_type == 'random':
        return RandomAgent(player_id)
    elif agent_type == 'heuristic':
        return HeuristicAgent(player_id)
    elif agent_type == 'rl':
        return RLAgent(player_id, **kwargs)
    elif agent_type == 'custom':
        return CustomAgent(player_id, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
```

## Conclusion

The agent system in the Werewolf Game provides a flexible framework for implementing various AI behaviors, from simple random decisions to complex strategic reasoning. By supporting different agent types, the game can be configured to provide appropriate challenge levels and to evaluate different strategies through systematic testing.

For more information on the game flow and API integration, please refer to the [Game Flow Documentation](game_flow.md) and [API Documentation](api_documentation.md). 