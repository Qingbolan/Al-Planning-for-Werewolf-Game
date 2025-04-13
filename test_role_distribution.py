#!/usr/bin/env python
"""
测试角色分配逻辑，确保狼人和爪牙总是分配给玩家
"""
import random
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

from werewolf_env.state import GameState
from config.default_config import DEFAULT_GAME_CONFIG

def test_role_distribution(num_tests=1000):
    """
    测试多次运行游戏时的角色分配情况
    
    Args:
        num_tests: 测试次数
    """
    # 记录统计数据
    player_roles_counter = Counter()
    center_roles_counter = Counter()
    
    for i in range(num_tests):
        # 创建游戏状态
        game_state = GameState(DEFAULT_GAME_CONFIG)
        
        # 收集玩家角色
        player_roles = [player['original_role'] for player in game_state.players]
        for role in player_roles:
            player_roles_counter[role] += 1
        
        # 收集中央牌堆角色
        for role in game_state.center_cards:
            center_roles_counter[role] += 1
    
    # 计算百分比
    total_player_cards = sum(player_roles_counter.values())
    total_center_cards = sum(center_roles_counter.values())
    
    player_percentages = {role: count/num_tests*100 for role, count in player_roles_counter.items()}
    center_percentages = {role: count/num_tests*100 for role, count in center_roles_counter.items()}
    
    # 打印统计结果
    print(f"\n=== 测试结果 ({num_tests}次游戏) ===\n")
    
    print("玩家角色分配情况:")
    for role, count in sorted(player_roles_counter.items()):
        percentage = count / total_player_cards * 100
        per_game = count / num_tests
        print(f"  {role}: {count} 次 ({percentage:.1f}%), 平均每局 {per_game:.2f} 个")
    
    print("\n中央牌堆角色分配情况:")
    for role, count in sorted(center_roles_counter.items()):
        if count > 0:
            percentage = count / total_center_cards * 100
            per_game = count / num_tests
            print(f"  {role}: {count} 次 ({percentage:.1f}%), 平均每局 {per_game:.2f} 个")
    
    # 验证关键角色是否总是分配给玩家
    werewolf_count = player_roles_counter['werewolf']
    minion_count = player_roles_counter['minion']
    werewolf_center = center_roles_counter.get('werewolf', 0)
    minion_center = center_roles_counter.get('minion', 0)
    
    print("\n关键角色验证:")
    print(f"  狼人应该有 {num_tests*2} 个, 实际玩家手中有 {werewolf_count} 个, 中央牌堆有 {werewolf_center} 个")
    print(f"  爪牙应该有 {num_tests} 个, 实际玩家手中有 {minion_count} 个, 中央牌堆有 {minion_center} 个")
    
    if werewolf_count == num_tests*2 and minion_count == num_tests and werewolf_center == 0 and minion_center == 0:
        print("\n✓ 测试通过! 狼人和爪牙总是分配给玩家，从不在中央牌堆中")
    else:
        print("\n✗ 测试失败! 角色分配不符合预期")
    
    # 可视化结果
    visualize_distribution(player_percentages, center_percentages)

def visualize_distribution(player_percentages, center_percentages):
    """
    可视化角色分配情况
    
    Args:
        player_percentages: 玩家角色百分比
        center_percentages: 中央牌堆角色百分比
    """
    try:
        plt.figure(figsize=(14, 6))
        
        # 玩家角色分布
        plt.subplot(1, 2, 1)
        roles = list(player_percentages.keys())
        percentages = [player_percentages[role] for role in roles]
        
        plt.bar(roles, percentages)
        plt.title('玩家角色分布')
        plt.ylabel('出现在玩家手中的百分比')
        plt.xticks(rotation=45)
        
        # 中央牌堆角色分布
        plt.subplot(1, 2, 2)
        center_roles = list(center_percentages.keys())
        center_values = [center_percentages[role] for role in center_roles]
        
        plt.bar(center_roles, center_values)
        plt.title('中央牌堆角色分布')
        plt.ylabel('出现在中央牌堆的百分比')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('role_distribution.png')
        print("\n分布图已保存为 'role_distribution.png'")
    except Exception as e:
        print(f"可视化失败: {e}")

if __name__ == "__main__":
    # 设置随机种子，确保结果可重现
    random.seed(42)
    np.random.seed(42)
    
    # 运行测试
    test_role_distribution(num_tests=1000)
