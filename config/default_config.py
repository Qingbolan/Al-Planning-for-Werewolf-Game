"""
狼人杀游戏默认配置
"""

# 游戏基本配置
DEFAULT_GAME_CONFIG = {
    # 游戏参与者数量
    'num_players': 6,
    
    # 中央牌堆数量
    'num_center_cards': 3,
    
    # 游戏最大轮数
    'max_rounds': 1,
    
    # 每个阶段的最大步数
    'max_steps_night': 50,
    'max_steps_day': 100,
    'max_steps_vote': 50,
    
    # 每个玩家发言的最大长度
    'max_speech_length': 5,
    
    # 奖励设置
    'reward_team_win': 1.0,      # 团队胜利奖励
    'reward_team_loss': -1.0,    # 团队失败惩罚
    'reward_correct_identify': 0.2,  # 正确识别角色奖励
    'reward_successful_hide': 0.2,   # 成功隐藏身份奖励
    'reward_persuasion': 0.1,        # 说服他人奖励
    
    # 观察空间设置
    'obs_include_history': True,     # 是否包含历史信息
    'obs_history_length': 5,         # 历史信息长度
    
    # 角色分配
    'roles': [
        'villager', 'villager',     # 2个村民
        'werewolf', 'werewolf',     # 2个狼人
        'seer',                     # 1个预言家
        'robber',                   # 1个强盗
        'troublemaker',             # 1个捣蛋鬼
        'minion',                   # 1个爪牙 
        'insomniac'                 # 1个失眠者
    ]
}

# 角色归属阵营
ROLE_TEAMS = {
    'villager': 'villager_team',
    'seer': 'villager_team',
    'robber': 'villager_team',
    'troublemaker': 'villager_team',
    'insomniac': 'villager_team',
    'werewolf': 'werewolf_team',
    'minion': 'werewolf_team'
}

# 角色行动顺序
ROLE_ACTION_ORDER = [
    'werewolf',
    'minion',
    'seer',
    'robber',
    'troublemaker',
    'insomniac'
]

# 夜晚行动空间定义
NIGHT_ACTIONS = {
    'werewolf': ['check_other_werewolves', 'check_center_card'],
    'minion': ['check_werewolves'],
    'seer': ['check_player', 'check_center_cards'],
    'robber': ['swap_role'],
    'troublemaker': ['swap_roles'],
    'insomniac': ['check_final_role'],
    'villager': []  # 村民夜间没有行动
}

# 白天发言模板
SPEECH_TEMPLATES = [
    # 角色声明
    "我是{role}",
    # 行动结果声明
    "作为{role}，我{action}了{target}，结果是{result}",
    # 指控
    "我认为玩家{player_id}是{role}",
    # 辩解
    "我不是{role}，因为{reason}",
    # 投票意向
    "我计划投票给玩家{player_id}"
] 