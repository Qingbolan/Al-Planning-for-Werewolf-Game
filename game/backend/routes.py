"""
API routes for the Werewolf Game (DEPRECATED - USE api.py INSTEAD)

This file is kept for reference but all routes have been moved to api.py.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Path

from game.backend.models import (
    GameConfig, PlayerConfig, CreateGameRequest, JoinGameRequest,
    PlayerAction, AIDecisionRequest, GameState, GameStateResponse,
    ActionResponse, AIDecisionResponse, GameStepResponse, GameResultResponse,
    CreateGameResponse
)
from game.backend.game_manager import GameManager

# Setup logging
logger = logging.getLogger("api_routes")

# Create router - not actually used anymore, see api.py
router = APIRouter()

"""
# The following routes are DEPRECATED and have been moved to api.py
# They are kept here for reference only.

# Game Creation and Management Endpoints
@router.post("/create", response_model=CreateGameResponse)
async def create_game(game_config: CreateGameRequest):
    # Implementation moved to api.py
    pass

@router.get("/create-test", response_model=CreateGameResponse)
async def create_test_game(
    test_game_type: str = Query("heuristic", description="AI type to use"),
    num_players: int = Query(6, description="Number of players in the game"),
    seed: Optional[int] = Query(None, description="Random seed for reproducible testing")
):
    # Implementation moved to api.py
    pass

@router.post("/join/{game_id}")
async def join_game(
    game_id: str = Path(..., description="The ID of the game to join"),
    join_request: JoinGameRequest = None
):
    # Implementation moved to api.py
    pass

@router.get("/state/{game_id}", response_model=GameStateResponse)
async def get_game_state(
    game_id: str = Path(..., description="The ID of the game"),
    player_id: Optional[int] = Query(None, description="ID of the player requesting the state")
):
    # Implementation moved to api.py
    pass

# Game Action Endpoints
@router.post("/action", response_model=ActionResponse)
async def perform_action(action: PlayerAction):
    # Implementation moved to api.py
    pass

@router.post("/step", response_model=GameStepResponse)
async def step_game(game_request: Dict[str, str]):
    # Implementation moved to api.py
    pass

@router.post("/ai-decision", response_model=AIDecisionResponse)
async def get_ai_decision(request: AIDecisionRequest):
    # Implementation moved to api.py
    pass

# Game Conclusion Endpoints
@router.get("/result/{game_id}", response_model=GameResultResponse)
async def get_game_result(
    game_id: str = Path(..., description="The ID of the game")
):
    # Implementation moved to api.py
    pass
""" 