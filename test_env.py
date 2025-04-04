"""
狼人杀游戏环境测试脚本
"""
import random
from werewolf_env.env import WerewolfEnv
from config.default_config import DEFAULT_GAME_CONFIG

def test_basic_environment():
    """测试环境的基本功能"""
    print("开始基本环境测试...")
    
    # 创建环境
    env = WerewolfEnv(render_mode="human")
    
    # 重置环境
    obs, info = env.reset()
    print("环境已重置")
    print(f"初始观察: {obs}")
    print(f"初始信息: {info}")
    
    # 运行一个完整的游戏
    done = False
    step_count = 0
    max_steps = 100  # 防止无限循环
    
    while not done and step_count < max_steps:
        # 随机选择动作
        action = env.action_space.sample()
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\n步骤 {step_count}:")
        print(f"动作: {action}")
        print(f"奖励: {reward}")
        print(f"阶段: {info['phase']}")
        print(f"当前玩家: {info['current_player']}")
        
        # 检查是否结束
        done = terminated or truncated
        step_count += 1
    
    print("\n游戏结束!")
    if 'game_result' in obs:
        print(f"胜利阵营: {obs['game_result']}")
    
    # 关闭环境
    env.close()
    
    print("基本环境测试完成")

def test_specific_scenario():
    """测试特定场景"""
    print("\n开始特定场景测试...")
    
    # 设置自定义配置
    config = DEFAULT_GAME_CONFIG.copy()
    config['num_players'] = 4
    config['roles'] = ['villager', 'villager', 'werewolf', 'seer', 'robber', 'troublemaker', 'minion']
    
    # 创建环境
    env = WerewolfEnv(config=config, render_mode="human")
    
    # 重置环境
    obs, info = env.reset(seed=42)  # 使用固定种子
    print("环境已重置(固定种子)")
    
    # 模拟一些特定动作序列
    actions = []
    # 这里可以设计一系列特定的动作来测试游戏逻辑
    # 例如:
    # - 狼人查看中央牌堆
    # - 预言家查看另一个玩家
    # - 白天发言指认狼人
    # - 投票
    
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\n步骤 {i}:")
        print(f"动作: {action}")
        print(f"阶段: {info['phase']}")
    
    # 如果没有特定动作，可以继续用随机动作
    if not actions:
        # 运行一个完整的游戏
        done = False
        step_count = 0
        max_steps = 100  # 防止无限循环
        
        while not done and step_count < max_steps:
            # 随机选择动作
            action = env.action_space.sample()
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"\n步骤 {step_count}:")
            print(f"动作: {action}")
            print(f"阶段: {info['phase']}")
            
            # 检查是否结束
            done = terminated or truncated
            step_count += 1
    
    # 关闭环境
    env.close()
    
    print("特定场景测试完成")

def test_role_actions():
    """测试各个角色的特定行动"""
    print("\n开始角色行动测试...")
    
    # 创建环境
    env = WerewolfEnv(render_mode="human")
    
    # 重置环境
    obs, info = env.reset()
    
    # 获取玩家分配的角色
    if env.game_state:
        print("角色分配:")
        for i, player in enumerate(env.game_state.players):
            print(f"玩家 {i}: {player['original_role']}")
    
    # 关闭环境
    env.close()
    
    print("角色行动测试完成")

if __name__ == "__main__":
    # 运行测试
    test_basic_environment()
    test_specific_scenario()
    test_role_actions() 