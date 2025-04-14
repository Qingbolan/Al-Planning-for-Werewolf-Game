# Werewolf Game Documentation

## Overview

This directory contains comprehensive documentation for the Werewolf Game implementation, including API specifications, game flow descriptions, and agent system details. The documentation serves as a guide for developers working on the project and users interacting with the game.

## Documentation Index

1. [API Documentation](api_documentation.md)
   - Detailed specifications of all API endpoints
   - Request and response formats
   - Data models and error handling

2. [Game Flow Documentation](game_flow.md)
   - Game rules and mechanics
   - Phase-by-phase gameplay description
   - Win conditions and game state transitions
   - Sequence diagram of component interactions

3. [Agent System Documentation](agent_system.md)
   - Agent types and characteristics
   - Decision-making processes for different roles
   - Agent communication formats
   - Performance testing methodology

4. [Testing Guide](testing_guide.md)
   - How to use the testing framework
   - Command-line arguments and options
   - Test scenarios and configurations
   - Analyzing test results

## Getting Started

For developers new to the project:

1. Start with the [Game Flow Documentation](game_flow.md) to understand the basic gameplay mechanics.
2. Review the [API Documentation](api_documentation.md) to learn about the communication between frontend and backend.
3. Explore the [Agent System Documentation](agent_system.md) to understand how AI players function.
4. Use the [Testing Guide](testing_guide.md) to evaluate agent performance.

## Implementation Overview

The Werewolf Game is structured as follows:

- **Frontend**: React-based UI for player interaction
- **Backend API**: FastAPI server handling game logic and state management
- **Agent System**: AI implementations for non-human players
- **Testing Framework**: Tools for evaluating agent performance

## Directory Structure

```
game/
├── api/                # API implementation
├── backend/            # Game backend logic
├── frontend/           # React frontend
│   └── src/
│       ├── components/ # UI components
│       └── services/   # API client services
└── documents/          # Documentation (you are here)
```

## Development and Testing

For information on testing agents and evaluating their performance, refer to the [Testing Guide](testing_guide.md) and the testing sections in the [Agent System Documentation](agent_system.md).

## Future Enhancements

Planned enhancements include:
- WebSocket implementation for real-time updates
- Additional role implementations
- Enhanced AI agent strategies
- Mobile-friendly UI improvements 