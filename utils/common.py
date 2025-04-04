def validate_type(value, expected_type, name="value"):
    if not isinstance(value, expected_type):
        raise ValueError(f"{name} 必须为 {expected_type.__name__} 类型")

def validate_state(state):
    validate_type(state, dict, name="状态")
    # 其他状态验证逻辑

def validate_action(action):
    validate_type(action, str, name="动作")
    # 其他动作验证逻辑 