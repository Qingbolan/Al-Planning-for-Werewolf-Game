
# Werewolf Game API Documentation (Revised)

## Overview

This document provides a comprehensive guide to the Werewolf Game API, designed based on REST principles with a stateless communication model for interaction between the frontend client and backend server. The API is built with FastAPI and supports various endpoints for game management and player interactions, closely matching the actual game implementation shown in the execution logs.

## Base URL

```
http://localhost:18000
```

## Game Flow and API Design

The Werewolf game API follows a specific flow with clear separation between game creation and action execution:

### 1. Game Creation Phase

* When creating a game via `/api/game/create` or `/api/game/create-test`, the backend **immediately generates and returns** all random initial game information:
  * Random role assignments to players
  * Random center card assignments
  * Complete initial game state

### 2. Sequential Action Phase

* After game creation, the frontend sequentially sends player actions following the game rules:
  * For AI agents: First request their decision via `/api/game/ai-decision`, then submit it
  * For human players: Directly submit their chosen action
* Each action must be sent for the correct current player in the proper sequence
* The backend enforces the correct action order and validates each action

This stateless design requires the frontend to maintain the game state between requests and ensure actions are submitted in the correct sequence according to the game rules.

## API Endpoints

### Game Management

#### Create New Game

Creates a new game with specified player configuration and **immediately returns the complete initial game state** including all random role assignments and center cards.

* **URL** : `/api/game/create`
* **Method** : `POST`
* **Request Body** :

```json
  {  "num_players": 6,  "players": {    "0": {"is_human": false, "name": "AI-0", "agent_type": "heuristic"},    "1": {"is_human": false, "name": "AI-1", "agent_type": "heuristic"},    "2": {"is_human": false, "name": "AI-2", "agent_type": "heuristic"},    "3": {"is_human": true, "name": "Human Player"},    "4": {"is_human": false, "name": "AI-4", "agent_type": "heuristic"},    "5": {"is_human": false, "name": "AI-5", "agent_type": "heuristic"}  },  "roles": ["werewolf", "werewolf", "minion", "villager", "seer", "troublemaker"],  "center_card_count": 3,  "max_speech_rounds": 3,  "seed": 42}
```

* **Response** :

```json
  {  "game_id": "abcd1234",  "message": "Game created successfully",  "success": true,  "state": {    "phase": "night",    "round": 0,    "speech_round": 0,    "current_player": 1,    "players": [      {        "player_id": 0,        "name": "AI-0",        "is_human": false,        "original_role": "troublemaker",        "current_role": "troublemaker",        "team": "villager",        "agent_type": "heuristic"      },      {        "player_id": 1,        "name": "AI-1",        "is_human": false,        "original_role": "werewolf",        "current_role": "werewolf",        "team": "werewolf",        "agent_type": "heuristic"      },      {        "player_id": 2,        "name": "AI-2",        "is_human": false,        "original_role": "minion",        "current_role": "minion",        "team": "werewolf",        "agent_type": "heuristic"      },      {        "player_id": 3,        "name": "Human Player",        "is_human": true,        "original_role": "villager",        "current_role": "villager",        "team": "villager"      },      {        "player_id": 4,        "name": "AI-4",        "is_human": false,        "original_role": "seer",        "current_role": "seer",        "team": "villager",        "agent_type": "heuristic"      },      {        "player_id": 5,        "name": "AI-5",        "is_human": false,        "original_role": "werewolf",        "current_role": "werewolf",        "team": "werewolf",        "agent_type": "heuristic"      }    ],    "center_cards": ["robber", "villager", "insomniac"],    "werewolf_indices": [1, 5],    "villager_indices": [0, 2, 3, 4],    "action_order": [      "werewolf",      "minion",      "mason",      "seer",      "robber",      "troublemaker",      "drunk",      "insomniac"    ]  }}
```

#### Create Test Game

Creates a game for observation and testing with all AI players. Immediately returns the complete initial game state including all random role assignments and center cards.

* **URL** : `/api/game/create-test`
* **Method** : `GET`
* **Query Parameters** :
* `test_game_type`: AI type to use (default: "heuristic", options: "random", "heuristic_villager_random_werewolf", "random_villager_heuristic_werewolf", "random_mix")
* `num_players`: Number of players in the game (default: 6)
* `seed`: Optional random seed for reproducible testing
* **Response** :

```json
  {  "game_id": "test1234",  "message": "Test game created successfully",  "success": true,  "test_game_type": "heuristic_villager_random_werewolf",  "state": {    "phase": "night",    "round": 0,    "speech_round": 0,    "current_player": 1,    "players": [      {        "player_id": 0,        "name": "AI-0",        "is_human": false,        "original_role": "troublemaker",        "current_role": "troublemaker",         "team": "villager",        "agent_type": "heuristic"      },      {        "player_id": 1,        "name": "AI-1",        "is_human": false,        "original_role": "werewolf",        "current_role": "werewolf",         "team": "werewolf",        "agent_type": "random"      },      {        "player_id": 2,        "name": "AI-2",        "is_human": false,        "original_role": "minion",        "current_role": "minion",         "team": "werewolf",        "agent_type": "heuristic"      },      {        "player_id": 3,        "name": "AI-3",        "is_human": false,        "original_role": "villager",        "current_role": "villager",         "team": "villager",        "agent_type": "heuristic"      },      {        "player_id": 4,        "name": "AI-4",        "is_human": false,        "original_role": "seer",        "current_role": "seer",         "team": "villager",        "agent_type": "heuristic"      },      {        "player_id": 5,        "name": "AI-5",        "is_human": false,        "original_role": "werewolf",        "current_role": "werewolf",         "team": "werewolf",        "agent_type": "random"      }    ],    "center_cards": ["robber", "villager", "insomniac"],    "werewolf_indices": [1, 5],    "villager_indices": [0, 2, 3, 4],    "action_order": [      "werewolf",      "minion",      "mason",      "seer",      "robber",      "troublemaker",      "drunk",      "insomniac"    ]  }}
```

#### Join Game

Allows a human player to join an existing game.

* **URL** : `/api/game/join/{game_id}`
* **Method** : `POST`
* **Request Body** :

```json
  {  "player_name": "Player1"}
```

* **Response** :

```json
  {  "player_id": "human_player",  "game_id": "abcd1234",  "success": true,  "message": "Successfully joined game and started",  "state": {    "phase": "night",    "current_player_id": 0,    "current_role": "werewolf",    "players": [      {        "player_id": 0,        "name": "AI-0",        "is_human": false,        "team": "werewolf"      },      {        "player_id": "human_player",        "name": "Player1",        "is_human": true,        "original_role": "seer",        "current_role": "seer",        "team": "villager"      },      // More players...    ]  }}
```

#### Get Game State

Retrieves the current game state, including visible information for the current player.

* **URL** : `/api/game/state/{game_id}`
* **Method** : `GET`
* **Query Parameters** :
* `player_id`: ID of the player requesting the state
* **Response** :

```json
  {  "game_id": "abcd1234",  "phase": "night",  "current_player_id": 0,  "current_role": "werewolf",  "players": [    {      "player_id": 0,      "name": "AI-0",      "is_human": false,      "original_role": "werewolf", // Only visible to this player or teammates      "current_role": "werewolf",      "team": "werewolf",      "agent_type": "heuristic"    },    // More players (information filtered based on permissions)...  ],  "player_count": 6,  "center_cards": ["?", "?", "?"], // Unknown cards shown as "?"  "known_center_cards": {}, // Center cards that have been viewed  "visible_roles": {}, // Known roles of other players  "turn": 1,  "action_order": [    "werewolf",    "minion",    "mason",    "seer",    "robber",    "troublemaker",    "drunk",    "insomniac"  ],  "valid_actions": [    {      "action_type": "night_action",      "action_name": "werewolf_action",      "action_params": {        "options": ["view_other_werewolves", "view_center_card"]      }    }  ],  "speech_round": null,  "max_speech_rounds": 3,  "votes": null,  "winner": null,  "game_over": false,  "history": [],  "message": null}
```

### Game Actions

#### Perform Player Action

Executes a player's action in the game (night action, daytime speech, or vote) according to the sequential turn order. This is the main endpoint for submitting all types of player actions.

* **URL** : `/api/game/action`
* **Method** : `POST`
* **Request Body** :

```json
  {  "game_id": "abcd1234",  "player_id": 4,  "action": {    "action_type": "NIGHT_ACTION",    "action_name": "check_player",    "action_params": {      "target_id": 5    }  }}
```

* **Response** :

```json
  {  "success": true,  "message": "Action executed successfully",  "action_result": {    "visible_roles": {      "5": "werewolf"    },    "original_role": "seer",    "current_role": "seer"  },  "state_update": {    "phase": "night",    "round": 0,    "speech_round": 0,    "current_player": 0,    "cumulative_rewards": {      "1": 0.0,      "5": 0.0,      "2": 0.0,      "4": 0.0    }  }}
```

#### Execute Game Step (For Automated Testing/Simulation)

Automatically advances the game by executing the next action in sequence using AI decision-making. This is useful for running simulations or automated games.

* **URL** : `/api/game/step`
* **Method** : `POST`
* **Request Body** :

```json
  {  "game_id": "abcd1234"}
```

* **Response** :

```json
  {  "success": true,  "step": 0,  "action": {    "player_id": 1,    "player_role": "werewolf",    "action_type": "NIGHT_ACTION",    "action_name": "check_other_werewolves",    "action_params": {}  },  "state_update": {    "phase": "night",    "round": 0,    "speech_round": 0,    "current_player": 5,    "cumulative_rewards": {      "1": 0.0    }  }}
```

#### Get AI Decision

Requests a decision from an AI player without executing it. This is used to determine what action an AI agent would take, so the frontend can then submit that action.

* **URL** : `/api/game/ai-decision`
* **Method** : `POST`
* **Request Body** :

```json
  {  "game_id": "abcd1234",  "player_id": 1,  "game_state": {    "phase": "night",    "round": 0,    "speech_round": 0,    "current_player": 1,    "players": [...],    "center_cards": [...],    "action_history": [...]  }}
```

* **Response** :

```json
  {  "success": true,  "player_id": 1,  "action": {    "action_type": "NIGHT_ACTION",    "action_name": "check_other_werewolves",    "action_params": {}  },  "reasoning": "Checking for other werewolves to identify team members"}
```

#### Perform Daytime Speech

Executes a player's daytime speech action. Each player speaks once per round, for a total of 3 rounds.

* **URL** : `/api/game/action`
* **Method** : `POST`
* **Request Body** :

```json
  {  "game_id": "abcd1234",  "player_id": 4,  "action": {    "action_type": "DAY_SPEECH",    "speech_type": "CLAIM_ROLE",    "content": "I am a seer"  }}
```

* **Response** :

```json
  {  "success": true,  "message": "Speech completed successfully",  "state_update": {    "phase": "day",    "round": 0,    "speech_round": 0,    "current_player": 5,    "cumulative_rewards": {      "1": 0.0,      "5": 0.0,      "2": 0.0,      "4": 0.0,      "0": 0.0,      "3": 0.0    }  }}
```

#### Perform Vote

Executes a player's voting action. Voting occurs sequentially by player index.

* **URL** : `/api/game/action`
* **Method** : `POST`
* **Request Body** :

```json
  {  "game_id": "abcd1234",  "player_id": 3,  "action": {    "action_type": "VOTE",    "target_id": 1  }}
```

* **Response** :

```json
  {  "success": true,  "message": "Vote submitted successfully",  "state_update": {    "phase": "vote",    "round": 0,     "speech_round": 3,    "current_player": 4,    "cumulative_rewards": {      "1": 0.0,      "5": 0.0,      "2": 0.0,      "4": 0.0,      "0": 0.0,      "3": 0.0    }  }}
```

### Game Conclusion and Results

#### Get Game Result

Retrieves complete results after a game has ended.

* **URL** : `/api/game/result/{game_id}`
* **Method** : `GET`
* **Response** :

```json
  {  "game_id": "abcd1234",  "winner": "werewolf",  "game_over": true,  "voting_results": {    "0": {      "voted_for": 4,      "original_role": "troublemaker",      "current_role": "troublemaker"    },    "1": {      "voted_for": 0,      "original_role": "werewolf",      "current_role": "werewolf"    },    "2": {      "voted_for": 3,      "original_role": "minion",      "current_role": "minion"    },    "3": {      "voted_for": 1,      "original_role": "villager",      "current_role": "villager"    },    "4": {      "voted_for": 0,      "original_role": "seer",      "current_role": "seer"    },    "5": {      "voted_for": 3,      "original_role": "werewolf",      "current_role": "werewolf"    }  },  "role_allocation": [    "troublemaker",     "werewolf",     "minion",     "villager",     "seer",     "werewolf",     "robber",     "villager",     "insomniac"  ],  "player_info": [    {      "player_id": 0,      "original_role": "troublemaker",      "current_role": "troublemaker",      "team": "villager",      "agent_type": "HeuristicAgent"    },    {      "player_id": 1,      "original_role": "werewolf",      "current_role": "werewolf",      "team": "werewolf",      "agent_type": "HeuristicAgent"    },    {      "player_id": 2,      "original_role": "minion",      "current_role": "minion",      "team": "werewolf",      "agent_type": "HeuristicAgent"    },    {      "player_id": 3,      "original_role": "villager",      "current_role": "villager",      "team": "villager",      "agent_type": "HeuristicAgent"    },    {      "player_id": 4,      "original_role": "seer",      "current_role": "seer",      "team": "villager",      "agent_type": "HeuristicAgent"    },    {      "player_id": 5,      "original_role": "werewolf",      "current_role": "werewolf",      "team": "werewolf",      "agent_type": "HeuristicAgent"    }  ],  "center_cards": ["robber", "villager", "insomniac"],  "statistics": {    "total_game_steps": 29,    "steps_per_phase": {      "night": 5,      "day": 18,      "vote": 6    }  },  "game_summary": "Werewolf team wins! Game completed in 29 steps."}
```

## Data Models

### Role Types

Available roles in the game:

* `villager`: Villager
* `werewolf`: Werewolf
* `seer`: Seer
* `robber`: Robber
* `troublemaker`: Troublemaker
* `insomniac`: Insomniac
* `minion`: Minion
* `hunter`: Hunter
* `tanner`: Tanner
* `drunk`: Drunk
* `mason`: Mason

### Game Phases

* `waiting`: Waiting for players to join
* `night`: Night phase, players perform role actions
* `day`: Day phase, players discuss and try to identify werewolves
* `vote`: Voting phase, players vote to decide who is a werewolf
* `game_over`: Game has ended

### Action Types

* `night_action`: Actions during night phase
* `day_speech`: Speech actions during day phase
* `vote`: Voting actions during voting phase

### Speech Types

* `CLAIM_ROLE`: Claiming one's own role (e.g., "I am a seer")
* `ACCUSE`: Accusing another player (e.g., "Player 2 is a werewolf")
* `DEFEND`: Defending oneself or another player
* `REVEAL_INFO`: Revealing information learned during night phase
* `GENERAL`: General speech not falling into the above categories

### Agent Types

* `random`: Random agent, makes completely random but valid decisions
* `heuristic`: Heuristic agent, makes decisions based on role-specific strategies
* `rl`: Reinforcement learning agent, makes decisions using a trained model

## Action Order

Based on the game logs, the actual action order implemented in the system is:

1. Werewolf: View other werewolves (`check_other_werewolves`) or check center card
2. Minion: View all werewolves (`check_werewolves`)
3. Mason: View other masons (if any)
4. Seer: View one player's card (`check_player`) or two center cards
5. Troublemaker: Exchange roles between two other players (`swap_roles`)
6. Robber: Exchange roles with another player and view new role
7. Drunk: Exchange roles with a center card but don't view
8. Insomniac: View own current role after all role swaps

The night phase is followed by the day phase, which consists of three speech rounds, with each player speaking once per round in order of player index (0 to N-1). After all speech rounds are completed, the game enters the voting phase, where players vote in sequence by player index.

## Error Handling

Errors will return appropriate HTTP status codes and JSON bodies with error details:

```json
{
  "detail": "Error message description"
}
```

Common error status codes:

* `400`: Bad Request - The client provided invalid data
* `404`: Not Found - The resource doesn't exist
* `500`: Internal Server Error - An error occurred on the server side

## Example Workflow (Based on Game Logs)

### Starting a New Game

1. Frontend calls `/api/game/create` or `/api/game/create-test` with the desired configuration
2. Backend immediately generates and returns:
   * Random role assignments to all players
   * Random center card assignments
   * Complete initial game state with phase set to "night" and the first player in the action order set as current player
3. Frontend stores this initial state for subsequent requests

### Night Phase Flow (For Mixed Human/AI Game)

1. Frontend displays the current player's turn based on the game state
2. If current player is AI:
   * Call `/api/game/ai-decision` to get the AI's chosen action
   * Then call `/api/game/action` to submit that action
3. If current player is human:
   * Display available night actions for their role
   * Wait for player input
   * Submit human action using `/api/game/action`
4. After each action, update the stored game state based on the response
5. Move to the next player in sequence
6. Night phase follows the role order: werewolves → minion → seer → troublemaker, etc.
7. After all night actions are completed, phase changes to "day" in the response

### Day Phase Flow (3 Speech Rounds)

1. For each player in sequence based on player index (0 to N-1):
   * If current player is AI, first call `/api/game/ai-decision` to get speech, then submit it
   * If current player is human, display speech options and wait for input
   * Submit speech using `/api/game/action` with action_type "DAY_SPEECH"
   * Update stored game state based on response
2. After all players have spoken in one round, speech_round is incremented in the response
3. Repeat for all three speech rounds
4. After all three speech rounds, phase changes to "vote" in the response

### Voting Phase Flow

1. For each player in sequence based on player index:
   * If current player is AI, first call `/api/game/ai-decision` to get vote, then submit it
   * If current player is human, display voting options and wait for input
   * Submit vote using `/api/game/action` with action_type "VOTE"
   * Update stored game state based on response
2. After all players vote, phase changes to "end" in the response

### Game Conclusion

1. Call `/api/game/result/{game_id}` to get final results including:
   * Winner team (werewolf or villager)
   * All voting decisions
   * Original and final role assignments
   * Game statistics

### Automated Simulation (For Testing)

For fully automated games with only AI players:

1. Create a game with all AI players using `/api/game/create-test`
2. Use `/api/game/step` repeatedly to advance the game automatically
3. The game will execute each step in sequence until completion
4. Retrieve final results using `/api/game/result/{game_id}`

## Appendix: Game-Specific API Design Considerations

1. **Asymmetric Information Design** : API responses filter information based on player role and game phase, ensuring each player only sees what they should see
2. **Action Validation** :

* Server validates the legality of each action
* Only the current active player can perform actions
* Actions must conform to the current game phase and role capabilities

1. **Role Ability Implementation** :

* Werewolves can see identities of other werewolves
* Seer can view one player's card or two center cards
* Robber, Troublemaker, and Drunk can exchange role cards
* Insomniac can view their current role

1. **Phase Transitions** :

* Night Phase: Automatically transitions to day phase after all night actions are completed
* Day Phase: Automatically transitions to voting phase after all speech rounds are completed
* Voting Phase: Automatically calculates results and transitions to game over state after all players vote

1. **Testing and Observation Mode** :

* Support for creating test games with different configurations
* Allow observation of games between AI agents
