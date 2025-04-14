# Werewolf Game Flow Documentation

## Introduction

This document describes the flow and rules of the Werewolf Game, a social deduction game where players are secretly assigned roles as villagers or werewolves, and must deduce who among them is a werewolf before it's too late. The implementation is a digital adaptation of the popular "One Night Ultimate Werewolf" tabletop game.

## Game Roles

### Werewolf Team
- **Werewolf**: Wakes up at night and views other werewolves. Wins if no werewolves are eliminated at the end of the game.
- **Minion**: Knows who the werewolves are but remains unknown to them. Wins when werewolves win.

### Villager Team
- **Villager**: Has no special ability, but must deduce who the werewolves are through discussion.
- **Seer**: Can look at one player's card or two of the center cards during the night phase.
- **Robber**: Can exchange their card with another player's card and view their new role.
- **Troublemaker**: Can swap the cards of two other players without looking at them.
- **Insomniac**: Wakes up at the end of the night to check if their role has changed.
- **Hunter**: If the Hunter is eliminated, the player they voted for is also eliminated.
- **Tanner**: Wins only if they are eliminated (independent of both teams).
- **Drunk**: Exchanges their card with a center card without seeing it.
- **Mason**: Masons wake up and see each other during the night phase.

## Game Setup

1. **Game Creation**: A game is created with specified configuration:
   - Number of players
   - Role distribution
   - Number of center cards
   - Maximum speech rounds

2. **Role Assignment**: Each player is assigned a secret role, and additional roles are placed in the center.

## Game Phases

### 1. Night Phase

During the night phase, players perform actions according to their roles in a specific order:

1. **Werewolves**: All werewolves wake up and see each other. They may also look at one center card.
2. **Minion**: Wakes up and sees who the werewolves are.
3. **Masons**: Wake up and see each other.
4. **Seer**: May look at one player's card or two center cards.
5. **Robber**: May exchange their card with another player's and view their new role.
6. **Troublemaker**: May exchange the cards of two other players without viewing them.
7. **Drunk**: Exchanges their card with a center card without seeing it.
8. **Insomniac**: Wakes up to check if their role has changed.

The night phase is sequential, with each player taking their turn when prompted by the system. Players can only see information relevant to their role.

### 2. Day Phase

During the day phase, players discuss to figure out who the werewolves are:

1. Each player gets multiple speech opportunities (configured as max_speech_rounds).
2. Speech types include:
   - **Claim**: Declare your role (truthfully or not)
   - **Accuse**: Point suspicion at another player
   - **Defend**: Defend yourself or another player
   - **Question**: Ask another player a question
   - **General**: Make a general statement

### 3. Vote Phase

After discussion, all players vote simultaneously:

1. Each player votes for one other player they believe is a werewolf.
2. The player with the most votes is eliminated.
3. If there's a tie, all tied players are eliminated.

### 4. Game Over

The game ends after the vote phase, and a winner is determined:

- **Werewolf Team Wins** if:
  - No werewolf is eliminated
  - All werewolves are eliminated but a minion is not

- **Villager Team Wins** if:
  - At least one werewolf is eliminated
  - No villager team members are eliminated

- **Tanner Wins** if:
  - The Tanner is eliminated (independent win condition)

## Implementation Details

### AI Agent Types

The game supports different types of AI agents:

1. **Random Agent**: Makes random but valid moves
2. **Heuristic Agent**: Makes decisions based on predefined heuristics appropriate to their role
3. **RL Agent**: Makes decisions using a trained reinforcement learning model (if available)

### Game State Tracking

The game state includes:

- Current phase
- Current player turn
- Role information (original and current)
- Center cards
- Game history
- Voting results
- Winner determination

## API Integration Flow

The game flow is facilitated through API calls that follow this sequence:

1. Game Creation: `/api/game/create`
2. Player Joining: `/api/game/join/{game_id}`
3. Game State Updates: `/api/game/state/{game_id}`
4. AI Decisions: `/api/game/ai-decision`

Each game state update triggers the appropriate next steps based on the current phase and player turn.

## User Interface Flow

The frontend reflects the game state and provides appropriate interfaces for each phase:

1. **Night Phase**: Shows role-specific UI elements for performing night actions
2. **Day Phase**: Provides speech interface and displays other players' speeches
3. **Vote Phase**: Provides voting interface
4. **Game Over**: Displays game result and winner

## Sequence Diagram

```
┌─────────┐      ┌──────────┐      ┌───────────────┐      ┌──────────────┐
│ Frontend│      │Backend API│      │Game Manager   │      │Game Instance │
└────┬────┘      └─────┬────┘      └───────┬───────┘      └──────┬───────┘
     │                 │                    │                     │
     │  Create Game    │                    │                     │
     │────────────────>│                    │                     │
     │                 │  Create Game       │                     │
     │                 │───────────────────>│                     │
     │                 │                    │  Initialize Game    │
     │                 │                    │────────────────────>│
     │                 │                    │                     │
     │                 │  Return Game ID    │                     │
     │<────────────────│<──────────────────│                     │
     │                 │                    │                     │
     │  Join Game      │                    │                     │
     │────────────────>│                    │                     │
     │                 │  Add Player        │                     │
     │                 │───────────────────>│                     │
     │                 │                    │  Add Player         │
     │                 │                    │────────────────────>│
     │                 │                    │                     │
     │                 │  Success Response  │                     │
     │<────────────────│<──────────────────│                     │
     │                 │                    │                     │
     │  Start Game     │                    │                     │
     │────────────────>│                    │                     │
     │                 │  Start Game        │                     │
     │                 │───────────────────>│                     │
     │                 │                    │  Initialize Phases  │
     │                 │                    │────────────────────>│
     │                 │                    │                     │
     │  Get Game State │                    │                     │
     │────────────────>│                    │                     │
     │                 │  Get State         │                     │
     │                 │───────────────────>│                     │
     │                 │                    │  Return State       │
     │                 │                    │<────────────────────│
     │                 │                    │                     │
     │                 │  Return State      │                     │
     │<────────────────│<──────────────────│                     │
     │                 │                    │                     │
     │  Perform Action │                    │                     │
     │────────────────>│                    │                     │
     │                 │  Process Action    │                     │
     │                 │───────────────────>│                     │
     │                 │                    │  Update Game State  │
     │                 │                    │────────────────────>│
     │                 │                    │                     │
     │                 │  Return Result     │                     │
     │<────────────────│<──────────────────│                     │
     │                 │                    │                     │
```

## Testing and Agent Evaluation

The testing framework allows for evaluating different agent types in various configurations:

1. **Agent Type Testing**: Compare performance of different agent types
2. **Specific Scenario Testing**: Test specific role distributions and team compositions
3. **RL Agent Testing**: Evaluate trained reinforcement learning agents

## Conclusion

The Werewolf Game is a complex social deduction game that requires careful implementation of game rules, roles, and player interactions. This document serves as a guide to understanding the game flow and integration with the API.

For detailed API information, please refer to the [API Documentation](api_documentation.md). 