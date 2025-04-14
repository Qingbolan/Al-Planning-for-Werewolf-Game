#!/usr/bin/env python
"""
狼人游戏API测试脚本
"""

import requests
import json
import time
from typing import Dict, Any, List, Optional

# API基础URL
BASE_URL = "http://localhost:18000"

# 颜色常量，用于美化输出
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_colored(message: str, color: str) -> None:
    """打印彩色文本"""
    print(f"{color}{message}{RESET}")

def print_request(method: str, url: str, data: Optional[Dict] = None) -> None:
    """打印请求信息"""
    print_colored(f"\n{method} {url}", BLUE)
    if data:
        print_colored(f"请求数据: {json.dumps(data, ensure_ascii=False, indent=2)}", BLUE)

def print_response(response, show_full=False) -> None:
    """打印响应信息"""
    if response.status_code == 200:
        print_colored(f"状态码: {response.status_code} OK", GREEN)
        if show_full:
            print_colored(f"响应数据: {json.dumps(response.json(), ensure_ascii=False, indent=2)}", GREEN)
        else:
            # 只显示部分关键信息
            data = response.json()
            if "success" in data:
                print_colored(f"成功: {data['success']}", GREEN)
            if "message" in data:
                print_colored(f"消息: {data['message']}", GREEN)
            if "game_id" in data:
                print_colored(f"游戏ID: {data['game_id']}", GREEN)
    else:
        print_colored(f"状态码: {response.status_code}", RED)
        print_colored(f"响应数据: {response.text}", RED)

def test_create_game() -> Dict:
    """测试创建新游戏"""
    print_colored("\n===== 测试创建游戏 =====", YELLOW)
    
    url = f"{BASE_URL}/api/game/create"
    data = {
        "num_players": 6,
        "players": {
            "0": {"is_human": False, "name": "AI-0", "agent_type": "heuristic"},
            "1": {"is_human": False, "name": "AI-1", "agent_type": "heuristic"},
            "2": {"is_human": False, "name": "AI-2", "agent_type": "heuristic"},
            "3": {"is_human": True, "name": "Human Player"},
            "4": {"is_human": False, "name": "AI-4", "agent_type": "heuristic"},
            "5": {"is_human": False, "name": "AI-5", "agent_type": "heuristic"}
        },
        "roles": ["werewolf", "werewolf", "minion", "villager", "seer", "troublemaker", "robber", "villager", "insomniac"],
        "center_card_count": 3,
        "max_speech_rounds": 3,
        "seed": 42
    }
    
    print_request("POST", url, data)
    response = requests.post(url, json=data)
    print_response(response)
    
    if response.status_code == 200:
        return response.json()
    return {"success": False}

def test_create_test_game() -> Dict:
    """测试创建测试游戏"""
    print_colored("\n===== 测试创建测试游戏 =====", YELLOW)
    
    url = f"{BASE_URL}/api/game/create-test"
    params = {
        "test_game_type": "heuristic",
        "num_players": 6,
        "seed": 42
    }
    
    print_request("GET", url, params)
    response = requests.get(url, params=params)
    print_response(response)
    
    if response.status_code == 200:
        return response.json()
    return {"success": False}

def test_join_game(game_id: str) -> Dict:
    """测试加入游戏"""
    print_colored("\n===== 测试加入游戏 =====", YELLOW)
    
    url = f"{BASE_URL}/api/game/join/{game_id}"
    data = {
        "player_name": "测试玩家"
    }
    
    print_request("POST", url, data)
    response = requests.post(url, json=data)
    print_response(response)
    
    if response.status_code == 200:
        return response.json()
    return {"success": False}

def test_get_game_state(game_id: str, player_id: Optional[int] = None) -> Dict:
    """测试获取游戏状态"""
    print_colored("\n===== 测试获取游戏状态 =====", YELLOW)
    
    url = f"{BASE_URL}/api/game/state/{game_id}"
    params = {}
    if player_id is not None:
        params["player_id"] = player_id
    
    print_request("GET", url, params if params else None)
    response = requests.get(url, params=params)
    print_response(response, show_full=True)
    
    if response.status_code == 200:
        return response.json()
    return {"success": False}

def test_get_ai_decision(game_id: str, player_id: int, game_state: Dict) -> Dict:
    """测试获取AI决策"""
    print_colored("\n===== 测试获取AI决策 =====", YELLOW)
    
    url = f"{BASE_URL}/api/game/ai-decision"
    data = {
        "game_id": game_id,
        "player_id": player_id,
        "game_state": game_state
    }
    
    print_request("POST", url, data)
    response = requests.post(url, json=data)
    print_response(response)
    
    if response.status_code == 200:
        return response.json()
    return {"success": False}

# 新增函数：测试狼人夜间行动
def test_werewolf_night_action(game_id: str, player_id: int) -> Dict:
    """测试狼人夜间行动"""
    print_colored("\n===== 测试狼人夜间行动 =====", YELLOW)
    
    action = {
        "action_type": "NIGHT_ACTION",
        "action_name": "check_other_werewolves",
        "action_params": {}
    }
    
    return test_perform_action(game_id, player_id, action)

# 新增函数：测试爪牙夜间行动
def test_minion_night_action(game_id: str, player_id: int) -> Dict:
    """测试爪牙夜间行动"""
    print_colored("\n===== 测试爪牙夜间行动 =====", YELLOW)
    
    action = {
        "action_type": "NIGHT_ACTION",
        "action_name": "check_werewolves",
        "action_params": {}
    }
    
    return test_perform_action(game_id, player_id, action)

# 新增函数：测试预言家夜间行动
def test_seer_night_action(game_id: str, player_id: int, target_id: int) -> Dict:
    """测试预言家夜间行动 - 查看玩家角色"""
    print_colored("\n===== 测试预言家夜间行动 =====", YELLOW)
    
    action = {
        "action_type": "NIGHT_ACTION",
        "action_name": "check_player",
        "action_params": {
            "target_id": target_id
        }
    }
    
    return test_perform_action(game_id, player_id, action)

# 新增函数：测试预言家查看中央牌
def test_seer_check_center(game_id: str, player_id: int) -> Dict:
    """测试预言家夜间行动 - 查看中央牌"""
    print_colored("\n===== 测试预言家查看中央牌 =====", YELLOW)
    
    action = {
        "action_type": "NIGHT_ACTION",
        "action_name": "check_center_cards",
        "action_params": {
            "indices": [0, 1]  # 查看前两张中央牌
        }
    }
    
    return test_perform_action(game_id, player_id, action)

# 新增函数：测试捣蛋鬼夜间行动
def test_troublemaker_night_action(game_id: str, player_id: int, target1_id: int, target2_id: int) -> Dict:
    """测试捣蛋鬼夜间行动"""
    print_colored("\n===== 测试捣蛋鬼夜间行动 =====", YELLOW)
    
    action = {
        "action_type": "NIGHT_ACTION",
        "action_name": "swap_roles",
        "action_params": {
            "target1_id": target1_id,
            "target2_id": target2_id
        }
    }
    
    return test_perform_action(game_id, player_id, action)

# 新增函数：测试强盗夜间行动
def test_robber_night_action(game_id: str, player_id: int, target_id: int) -> Dict:
    """测试强盗夜间行动"""
    print_colored("\n===== 测试强盗夜间行动 =====", YELLOW)
    
    action = {
        "action_type": "NIGHT_ACTION",
        "action_name": "swap_with_player",
        "action_params": {
            "target_id": target_id
        }
    }
    
    return test_perform_action(game_id, player_id, action)

# 新增函数：测试白天发言
def test_day_speech(game_id: str, player_id: int, speech_type: str, content: str) -> Dict:
    """测试白天发言"""
    print_colored("\n===== 测试白天发言 =====", YELLOW)
    
    action = {
        "action_type": "DAY_SPEECH",
        "speech_type": speech_type,
        "content": content
    }
    
    return test_perform_action(game_id, player_id, action)

# 新增函数：测试玩家投票
def test_vote(game_id: str, player_id: int, target_id: int) -> Dict:
    """测试玩家投票"""
    print_colored("\n===== 测试玩家投票 =====", YELLOW)
    
    action = {
        "action_type": "VOTE",
        "target_id": target_id
    }
    
    return test_perform_action(game_id, player_id, action)

def test_perform_action(game_id: str, player_id: int, action: Dict) -> Dict:
    """测试执行玩家行动"""
    print_colored("\n===== 测试执行玩家行动 =====", YELLOW)
    
    url = f"{BASE_URL}/api/game/action"
    data = {
        "game_id": game_id,
        "player_id": player_id,
        "action": action
    }
    
    print_request("POST", url, data)
    response = requests.post(url, json=data)
    print_response(response)
    
    if response.status_code == 200:
        return response.json()
    return {"success": False}

def test_step_game(game_id: str) -> Dict:
    """测试执行游戏步骤"""
    print_colored("\n===== 测试执行游戏步骤 =====", YELLOW)
    
    url = f"{BASE_URL}/api/game/step"
    data = {
        "game_id": game_id
    }
    
    print_request("POST", url, data)
    response = requests.post(url, json=data)
    print_response(response)
    
    if response.status_code == 200:
        return response.json()
    return {"success": False}

def test_get_game_result(game_id: str) -> Dict:
    """测试获取游戏结果"""
    print_colored("\n===== 测试获取游戏结果 =====", YELLOW)
    
    url = f"{BASE_URL}/api/game/result/{game_id}"
    
    print_request("GET", url)
    response = requests.get(url)
    print_response(response, show_full=True)
    
    if response.status_code == 200:
        return response.json()
    return {"success": False}

def run_full_game_simulation():
    """运行完整的游戏模拟测试"""
    print_colored("\n***** 开始完整游戏模拟测试 *****", YELLOW)
    
    # 1. 创建测试游戏
    game_data = test_create_test_game()
    if not game_data.get("success", False):
        print_colored("创建游戏失败，中止测试", RED)
        return
    
    game_id = game_data["game_id"]
    print_colored(f"成功创建游戏，ID: {game_id}", GREEN)
    
    # 2. 获取游戏状态
    game_state = test_get_game_state(game_id)
    
    # 3. 模拟完整游戏流程
    print_colored("\n***** 使用step API自动运行游戏 *****", YELLOW)
    
    step_count = 0
    max_steps = 100  # 防止无限循环
    
    while step_count < max_steps:
        step_result = test_step_game(game_id)
        if not step_result.get("success", False):
            print_colored("游戏步骤执行失败", RED)
            break
        
        step_count += 1
        time.sleep(0.5)  # 短暂暂停以便查看输出
        
        # 检查游戏是否结束
        state_update = step_result.get("state_update", {})
        if state_update.get("phase") == "game_over":
            print_colored("游戏已结束!", GREEN)
            break
    
    # 4. 获取最终游戏结果
    test_get_game_result(game_id)

def run_manual_api_tests():
    """手动逐步测试每个API"""
    print_colored("\n***** 开始手动API测试 *****", YELLOW)
    
    # 1. 创建常规游戏
    game_data = test_create_game()
    if not game_data.get("success", False):
        print_colored("创建游戏失败，中止测试", RED)
        return
    
    game_id = game_data["game_id"]
    game_state = game_data.get("state", {})
    print_colored(f"成功创建游戏，ID: {game_id}", GREEN)
    
    # 2. 测试加入游戏
    join_result = test_join_game(game_id)
    
    # 3. 测试获取游戏状态
    state_result = test_get_game_state(game_id)
    
    # 4. 为AI玩家获取决策并执行
    # 假设角色分配如下（根据实际返回数据调整）
    # 找出狼人玩家
    werewolf_player_id = None
    for player in game_state.get("players", []):
        if player.get("original_role") == "werewolf":
            werewolf_player_id = player.get("player_id")
            break
            
    if werewolf_player_id is not None:
        # 测试狼人查看其他狼人
        werewolf_action = test_werewolf_night_action(game_id, werewolf_player_id)
    
    # 找出预言家玩家
    seer_player_id = None
    for player in game_state.get("players", []):
        if player.get("original_role") == "seer":
            seer_player_id = player.get("player_id")
            break
            
    if seer_player_id is not None:
        # 找一个非预言家的玩家作为目标
        target_id = (seer_player_id + 1) % len(game_state.get("players", []))
        # 测试预言家查看玩家
        seer_action = test_seer_night_action(game_id, seer_player_id, target_id)
        
    # 找出捣蛋鬼玩家
    troublemaker_player_id = None
    for player in game_state.get("players", []):
        if player.get("original_role") == "troublemaker":
            troublemaker_player_id = player.get("player_id")
            break
            
    if troublemaker_player_id is not None:
        # 选择两个非捣蛋鬼的玩家进行交换
        target1_id = (troublemaker_player_id + 1) % len(game_state.get("players", []))
        target2_id = (troublemaker_player_id + 2) % len(game_state.get("players", []))
        # 测试捣蛋鬼交换玩家角色
        troublemaker_action = test_troublemaker_night_action(game_id, troublemaker_player_id, target1_id, target2_id)
    
    # 5. 测试白天发言（对每个玩家）
    player_count = len(game_state.get("players", []))
    for player_id in range(player_count):
        speech_content = f"玩家{player_id}的发言：我是善良村民"
        speech_type = "CLAIM_ROLE"
        day_speech = test_day_speech(game_id, player_id, speech_type, speech_content)
    
    # 6. 测试玩家投票
    for player_id in range(player_count):
        # 每个玩家投给随机的其他玩家
        target_id = (player_id + 1) % player_count
        vote_action = test_vote(game_id, player_id, target_id)
    
    # 7. 测试获取游戏结果
    result = test_get_game_result(game_id)

def run_specific_action_test():
    """测试特定的API和行动"""
    print_colored("\n***** 开始特定行动测试 *****", YELLOW)
    
    # 1. 创建测试游戏
    game_data = test_create_test_game()
    if not game_data.get("success", False):
        print_colored("创建游戏失败，中止测试", RED)
        return
    
    game_id = game_data["game_id"]
    game_state = game_data.get("state", {})
    print_colored(f"成功创建游戏，ID: {game_id}", GREEN)
    
    # 2. 获取游戏状态
    state_result = test_get_game_state(game_id)
    
    # 3. 提供测试菜单
    while True:
        print("\n请选择要测试的特定行动：")
        print("1. 狼人夜间行动")
        print("2. 爪牙夜间行动")
        print("3. 预言家查看玩家")
        print("4. 预言家查看中央牌")
        print("5. 捣蛋鬼夜间行动")
        print("6. 强盗夜间行动")
        print("7. 白天发言")
        print("8. 投票")
        print("9. 获取游戏结果")
        print("0. 退出特定行动测试")
        
        choice = input("请输入选择 (0-9): ")
        
        if choice == "0":
            break
        elif choice == "1":
            player_id = int(input("请输入狼人玩家ID: "))
            test_werewolf_night_action(game_id, player_id)
        elif choice == "2":
            player_id = int(input("请输入爪牙玩家ID: "))
            test_minion_night_action(game_id, player_id)
        elif choice == "3":
            player_id = int(input("请输入预言家玩家ID: "))
            target_id = int(input("请输入目标玩家ID: "))
            test_seer_night_action(game_id, player_id, target_id)
        elif choice == "4":
            player_id = int(input("请输入预言家玩家ID: "))
            test_seer_check_center(game_id, player_id)
        elif choice == "5":
            player_id = int(input("请输入捣蛋鬼玩家ID: "))
            target1_id = int(input("请输入第一个目标玩家ID: "))
            target2_id = int(input("请输入第二个目标玩家ID: "))
            test_troublemaker_night_action(game_id, player_id, target1_id, target2_id)
        elif choice == "6":
            player_id = int(input("请输入强盗玩家ID: "))
            target_id = int(input("请输入目标玩家ID: "))
            test_robber_night_action(game_id, player_id, target_id)
        elif choice == "7":
            player_id = int(input("请输入发言玩家ID: "))
            print("\n请选择发言类型:")
            print("1. CLAIM_ROLE - 声称角色")
            print("2. ACCUSE - 指控他人")
            print("3. DEFEND - 为自己或他人辩护")
            print("4. REVEAL_INFO - 揭示夜间获得的信息")
            print("5. GENERAL - 一般发言")
            speech_type_choice = input("请选择发言类型 (1-5): ")
            
            speech_types = {
                "1": "CLAIM_ROLE",
                "2": "ACCUSE",
                "3": "DEFEND",
                "4": "REVEAL_INFO",
                "5": "GENERAL"
            }
            
            speech_type = speech_types.get(speech_type_choice, "GENERAL")
            content = input("请输入发言内容: ")
            test_day_speech(game_id, player_id, speech_type, content)
        elif choice == "8":
            player_id = int(input("请输入投票玩家ID: "))
            target_id = int(input("请输入投票目标玩家ID: "))
            test_vote(game_id, player_id, target_id)
        elif choice == "9":
            test_get_game_result(game_id)
        else:
            print_colored("无效选择，请重新输入", RED)

if __name__ == "__main__":
    print_colored("==== 狼人游戏API测试 ====", YELLOW)
    
    try:
        # 检查API服务器是否运行
        health_check = requests.get(f"{BASE_URL}/")
        print_colored("API服务器已连接", GREEN)
    except requests.exceptions.ConnectionError:
        print_colored(f"无法连接到API服务器: {BASE_URL}", RED)
        print_colored("请确保服务器正在运行，并且端口18000可访问", RED)
        exit(1)
    
    # 选择测试模式
    print("\n请选择测试模式:")
    print("1. 完整游戏模拟 (使用step API自动运行整个游戏)")
    print("2. 单独API测试 (分别测试每个API端点)")
    print("3. 特定行动测试 (测试特定的游戏行动)")
    
    choice = input("请输入选择 (1-3): ")
    
    if choice == "1":
        run_full_game_simulation()
    elif choice == "2":
        run_manual_api_tests()
    elif choice == "3":
        run_specific_action_test()
    else:
        print_colored("无效选择，退出测试", RED) 