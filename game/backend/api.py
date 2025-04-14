"""
Game API Implementation

This module implements the REST API endpoints for the Werewolf game.
It connects the frontend with the game manager.
"""

import logging
from fastapi import APIRouter, HTTPException, Path, Query, Body
from typing import Optional, Dict, Any

from game.backend.game_manager import GameManager
from game.backend.models import (
    CreateGameRequest, 
    GameConfig, 
    PlayerConfig,
    GameStateResponse, 
    PlayerAction,
    ActionResponse,
    AIDecisionRequest,
    AIDecisionResponse,
    GameStepResponse,
    GameResultResponse,
    CreateGameResponse,
    JoinGameRequest
)

# Setup logging
logger = logging.getLogger("game_api")

# Update router prefix to match API documentation
router = APIRouter(prefix="/api", tags=["game"])


@router.post("/game/create", response_model=CreateGameResponse)
async def create_game(request: CreateGameRequest):
    """
    Create a new game with the specified configuration
    
    Immediately generates and returns all random initial game information:
    - Random role assignments to players
    - Random center card assignments
    - Complete initial game state
    """
    try:
        logger.info(f"Creating game with request: {request}")
        
        # Validate num_players
        if request.num_players <= 0:
            raise HTTPException(status_code=400, detail="Number of players must be positive")
            
        # Validate roles
        if len(request.roles) < request.num_players:
            raise HTTPException(
                status_code=400, 
                detail=f"Not enough roles. Need {request.num_players}, but only {len(request.roles)} provided."
            )
        
        # Convert request to GameConfig
        config = GameConfig(
            num_players=request.num_players,
            players=request.players,
            roles=request.roles,
            center_card_count=request.center_card_count,
            max_speech_rounds=request.max_speech_rounds,
            seed=request.seed
        )
        
        # Create game and return result
        result = GameManager.create_game(config)
        
        if not result.get("success", False):
            # If the GameManager returns an error, forward it as an HTTP exception
            raise HTTPException(
                status_code=400, 
                detail=result.get("message", "Failed to create game")
            )
            
        return result
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error in create_game endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating game: {str(e)}")


@router.get("/game/create-test", response_model=CreateGameResponse)
async def create_test_game(
    test_game_type: str = Query("heuristic", description="AI type to use (options: random, heuristic, heuristic_villager_random_werewolf, random_villager_heuristic_werewolf, random_mix)"),
    num_players: int = Query(6, description="Number of players in the game", gt=0),
    seed: Optional[int] = Query(None, description="Random seed for reproducible testing")
):
    """
    Create a game for observation and testing with all AI players.
    
    Immediately returns the complete initial game state including all
    random role assignments and center cards.
    """
    try:
        # Ensure num_players is valid
        if num_players <= 0:
            raise HTTPException(status_code=400, detail="Number of players must be positive")
            
        # Create the test game
        result = GameManager.create_test_game(test_game_type, num_players, seed)
        
        if not result.get("success", False):
            # If the GameManager returns an error, forward it as an HTTP exception
            raise HTTPException(
                status_code=400, 
                detail=result.get("message", "Failed to create test game")
            )
            
        # 添加test_game_type到响应中
        result["test_game_type"] = test_game_type
            
        return result
    except Exception as e:
        logger.error(f"Error in create_test_game endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating test game: {str(e)}")


@router.post("/game/join/{game_id}")
async def join_game(
    game_id: str = Path(..., description="The ID of the game to join"),
    join_request: JoinGameRequest = Body(..., description="Join request")
):
    """
    Allows a human player to join an existing game.
    """
    # This is a simplified implementation that would need to be expanded
    # in a real system with player authentication and session management
    return {
        "player_id": "human_player",
        "game_id": game_id,
        "success": True,
        "message": "Successfully joined game and started",
        "state": GameManager.get_game_state(game_id).get("state", {})
    }


@router.get("/game/state/{game_id}", response_model=GameStateResponse)
async def get_game_state(
    game_id: str = Path(..., description="The ID of the game"),
    player_id: Optional[int] = Query(None, description="ID of the player requesting the state")
):
    """
    Retrieves the current game state, including visible information for the current player.
    
    If player_id is provided, information will be filtered based on what that player can see.
    """
    result = GameManager.get_game_state(game_id, player_id)
    
    if not result.get("success", False):
        raise HTTPException(status_code=404, detail=result.get("message", "Game not found"))
    
    return result


@router.post("/game/action", response_model=ActionResponse)
async def perform_action(action: PlayerAction = Body(..., description="Player action")):
    """
    Executes a player's action in the game (night action, daytime speech, or vote).
    
    This is the main endpoint for submitting all types of player actions.
    """
    # Execute action
    result = GameManager.perform_action(
        game_id=action.game_id,
        player_id=action.player_id,
        action=action.action
    )
    
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("message", "Invalid action"))
    
    return result


@router.post("/game/ai-decision", response_model=AIDecisionResponse)
async def get_ai_decision(request: AIDecisionRequest = Body(..., description="AI decision request")):
    """
    Requests a decision from an AI player without executing it.
    
    This is used to determine what action an AI agent would take, so the frontend can then
    submit that action.
    """
    # Get AI decision
    result = GameManager.get_ai_decision(
        game_id=request.game_id,
        player_id=request.player_id,
        game_state=request.game_state
    )
    
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("message", "Failed to get AI decision"))
    
    return result


@router.post("/game/step", response_model=GameStepResponse)
async def step_game(game_request: Dict[str, str] = Body(..., description="Game ID")):
    """
    Automatically advances the game by executing the next action in sequence using AI decision-making.
    
    This is useful for running simulations or automated games.
    """
    game_id = game_request.get("game_id")
    if not game_id:
        raise HTTPException(status_code=400, detail="Game ID is required")
    
    result = GameManager.step_game(game_id)
    
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("message", "Failed to step game"))
    
    return result


@router.get("/game/result/{game_id}", response_model=GameResultResponse)
async def get_game_result(
    game_id: str = Path(..., description="The ID of the game")
):
    """
    Retrieves complete results after a game has ended.
    """
    result = GameManager.get_game_result(game_id)
    
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("message", "Game not found or not over"))
    
    return result 