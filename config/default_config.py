"""
Default configuration for Werewolf Game
"""

# Basic game configuration
DEFAULT_GAME_CONFIG = {
    # Number of players
    'num_players': 6,
    
    # Number of center cards
    'num_center_cards': 3,
    
    # Maximum number of rounds
    'max_rounds': 1,
    
    # Maximum steps per phase
    'max_steps_night': 50,
    'max_steps_day': 100,
    'max_steps_vote': 50,
    
    # Maximum speech length per player
    'max_speech_length': 5,
    
    # Observation space settings
    'obs_include_history': True,     # Whether to include history information
    'obs_history_length': 6,         # History information length
    
    # Role assignment - 确保特定角色分配
    'roles': [
        # 这三个角色一定会分配给玩家
        'werewolf', 'werewolf',      # 2 werewolves (必定分配给玩家)
        'minion',                    # 1 minion (必定分配给玩家)
        
        # 以下6个角色会随机分配：3个分配给剩余的玩家，3个进入中央牌堆
        'villager',                  # 1 villager
        'seer',                      # 1 seer
        'robber',                    # 1 robber
        'troublemaker',              # 1 troublemaker
        'villager',                  # 1 more villager
        'insomniac'                  # 1 insomniac
    ]
}

# Role team assignments (simplified team names)
ROLE_TEAMS = {
    'villager': 'villager',
    'seer': 'villager',
    'robber': 'villager',
    'troublemaker': 'villager',
    'insomniac': 'villager',
    'werewolf': 'werewolf',
    'minion': 'werewolf'
}

# Role action order
ROLE_ACTION_ORDER = [
    'werewolf',
    'minion',
    'seer',
    'robber',
    'troublemaker',
    'insomniac'
]

# Night action space definition
NIGHT_ACTIONS = {
    'werewolf': ['check_other_werewolves', 'check_center_card'],
    'minion': ['check_werewolves'],
    'seer': ['check_player', 'check_center_cards'],
    'robber': ['swap_role'],
    'troublemaker': ['swap_roles'],
    'insomniac': ['check_final_role'],
    'villager': []  # Villagers have no night actions
}

# Day speech templates
SPEECH_TEMPLATES = [
    # Role claim
    "I am a {role}",
    # Action result claim
    "As a {role}, I {action} {target}, and the result is {result}",
    # Accusation
    "I think player {player_id} is a {role}",
    # Defense
    "I am not a {role} because {reason}",
    # Vote intention
    "I plan to vote for player {player_id}"
] 