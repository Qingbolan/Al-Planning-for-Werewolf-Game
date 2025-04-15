"""
Data models for the Werewolf Game API
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any, Literal
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Action types in the game"""
    NIGHT_ACTION = "NIGHT_ACTION"
    DAY_SPEECH = "DAY_SPEECH"
    VOTE = "VOTE"


class SpeechType(str, Enum):
    """Types of daytime speech"""
    CLAIM_ROLE = "CLAIM_ROLE"
    ACCUSE = "ACCUSE"
    DEFEND = "DEFEND"
    REVEAL_INFO = "REVEAL_INFO"
    GENERAL = "GENERAL"


class RoleType(str, Enum):
    """Available roles in the game"""
    VILLAGER = "villager"
    WEREWOLF = "werewolf"
    SEER = "seer"
    ROBBER = "robber"
    TROUBLEMAKER = "troublemaker"
    INSOMNIAC = "insomniac"
    MINION = "minion"
    HUNTER = "hunter"
    TANNER = "tanner"
    DRUNK = "drunk"
    MASON = "mason"


class TeamType(str, Enum):
    """Teams in the game"""
    VILLAGER = "villager"
    WEREWOLF = "werewolf"
    TANNER = "tanner"  # Tanner is on their own team


class GamePhase(str, Enum):
    """Game phases"""
    WAITING = "waiting"
    NIGHT = "night"
    DAY = "day"
    VOTE = "vote"
    GAME_OVER = "game_over"


class AgentType(str, Enum):
    """Types of AI agents"""
    RANDOM = "random"
    HEURISTIC = "heuristic"
    RL = "rl"
    

class PlayerConfig(BaseModel):
    """Player configuration for game creation"""
    is_human: bool
    name: str
    agent_type: Optional[str] = Field(None, description="AI agent type (only for non-human players)")
    

class GameConfig(BaseModel):
    """Configuration for creating a new game"""
    num_players: int = Field(..., description="Number of players in the game")
    players: Dict[str, PlayerConfig] = Field(..., description="Player configurations by ID")
    roles: List[str] = Field(..., description="Roles to be used in the game")
    center_card_count: int = Field(3, description="Number of center cards")
    max_speech_rounds: int = Field(3, description="Maximum number of speech rounds in the day phase")
    seed: Optional[int] = Field(None, description="Random seed for reproducible games")


class CreateGameRequest(BaseModel):
    """Request body for creating a new game"""
    num_players: int
    players: Dict[str, PlayerConfig]
    roles: List[str]
    center_card_count: int = 3
    max_speech_rounds: int = 3
    seed: Optional[int] = None


class JoinGameRequest(BaseModel):
    """Request body for joining a game"""
    player_name: str


class PlayerInfo(BaseModel):
    """Player information"""
    player_id: int
    name: str
    is_human: bool
    original_role: Optional[str] = None
    current_role: Optional[str] = None
    team: Optional[str] = None
    agent_type: Optional[str] = None


class NightAction(BaseModel):
    """Night action details"""
    action_type: Literal[ActionType.NIGHT_ACTION] = ActionType.NIGHT_ACTION
    action_name: str
    action_params: Dict[str, Any] = {}


class DaySpeech(BaseModel):
    """Day speech action details"""
    action_type: Literal[ActionType.DAY_SPEECH] = ActionType.DAY_SPEECH
    speech_type: SpeechType
    content: str


class VoteAction(BaseModel):
    """Vote action details"""
    action_type: Literal[ActionType.VOTE] = ActionType.VOTE
    target_id: int


class PlayerAction(BaseModel):
    """Player action request"""
    game_id: str
    player_id: int
    action: Union[NightAction, DaySpeech, VoteAction]


class AIDecisionRequest(BaseModel):
    """Request for AI decision"""
    game_id: str
    player_id: int
    game_state: Optional[Dict[str, Any]] = None


class GameHistory(BaseModel):
    """Game history entry"""
    step: int
    player_id: int
    action: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    timestamp: float


class GameState(BaseModel):
    """Full game state"""
    phase: GamePhase
    round: int = 0
    speech_round: int = 0
    current_player: Optional[int] = None
    players: List[PlayerInfo]
    center_cards: List[str]
    werewolf_indices: List[int] = []
    villager_indices: List[int] = []
    action_order: List[str]
    votes: Optional[Dict[str, int]] = None
    voting_results: Optional[Dict[str, Any]] = None
    winner: Optional[str] = None
    game_over: bool = False
    known_center_cards: Dict[str, Dict[str, str]] = {}
    visible_roles: Dict[str, Dict[str, str]] = {}
    valid_actions: Dict[str, List[Dict[str, Any]]] = {}
    history: List[GameHistory] = []
    cumulative_rewards: Dict[str, float] = {}


class CreateGameResponse(BaseModel):
    """Response for game creation"""
    game_id: str
    message: str
    success: bool
    state: GameState
    test_game_type: Optional[str] = None  # 添加test_game_type字段，用于create-test接口


class GameStateResponse(BaseModel):
    """Response for getting game state"""
    success: bool
    game_id: str
    phase: GamePhase
    current_player_id: Optional[int] = None
    current_role: Optional[str] = None
    players: List[PlayerInfo]
    player_count: int
    center_cards: List[str]
    known_center_cards: Dict[str, Any] = {}
    visible_roles: Dict[str, Any] = {}
    turn: int
    action_order: List[str]
    valid_actions: List[Dict[str, Any]] = []
    speech_round: Optional[int] = None
    max_speech_rounds: int
    votes: Optional[Dict[str, int]] = None
    winner: Optional[str] = None
    game_over: bool = False
    history: List[Dict[str, Any]] = []
    message: Optional[str] = None


class ActionResponse(BaseModel):
    """Response for action execution"""
    success: bool
    message: str
    action_result: Optional[Dict[str, Any]] = None
    state_update: Optional[Dict[str, Any]] = None


class AIDecisionResponse(BaseModel):
    """Response for AI decision"""
    success: bool
    player_id: int
    action: Dict[str, Any]
    reasoning: str


class GameStepResponse(BaseModel):
    """Response for game step"""
    success: bool
    step: int
    action: Dict[str, Any]
    state_update: Dict[str, Any]


class GameResultResponse(BaseModel):
    """Response for game result"""
    game_id: str
    winner: str
    game_over: bool
    voting_results: Dict[str, Dict[str, Any]]
    role_allocation: List[str]
    player_info: List[Dict[str, Any]]
    center_cards: List[str]
    statistics: Dict[str, Any]
    game_summary: str 


class NightActionRequest(BaseModel):
    """Request for night action"""
    player_id: int
    role: str
    game_state: GameState

class NightActionResponse(BaseModel):
    """Response for night action"""
    success: bool
    action: NightAction
    game_state: GameState


