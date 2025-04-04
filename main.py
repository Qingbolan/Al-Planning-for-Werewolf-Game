"""
狼人杀游戏主程序
"""
import argparse
import os
import torch
import numpy as np
import random
import time
from typing import List, Dict, Any

from werewolf_env import WerewolfEnv
from agents import (
    BaseAgent, RandomAgent, HeuristicAgent, RLAgent,
    create_agent, create_rl_agent
)
from train import Trainer
from utils.visualizer import BeliefVisualizer


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='狼人杀游戏')
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='play',
                       choices=['play', 'train', 'evaluate'],
                       help='运行模式: play (游玩), train (训练), evaluate (评估)')
    
    # 游戏配置
    parser.add_argument('--num_players', type=int, default=6,
                       help='玩家数量')
    parser.add_argument('--render', action='store_true',
                       help='是否渲染游戏')
    
    # 智能体配置
    parser.add_argument('--agent_types', type=str, nargs='+', 
                       default=['random'],
                       choices=['random', 'heuristic', 'rl'],
                       help='使用的智能体类型')
    
    # 训练配置
    parser.add_argument('--train_episodes', type=int, default=1000,
                       help='训练局数')
    parser.add_argument('--num_generations', type=int, default=10,
                       help='训练世代数')
    parser.add_argument('--load_model', type=str, default=None,
                       help='加载模型路径')
    parser.add_argument('--save_model', type=str, default='./models/rl_agent.pt',
                       help='保存模型路径')
    
    # 评估配置
    parser.add_argument('--eval_episodes', type=int, default=100,
                       help='评估局数')
    parser.add_argument('--visualize', action='store_true',
                       help='是否可视化信念状态')
    
    # 其他配置
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='运行设备 (cpu 或 cuda)')
    
    return parser.parse_args()


def play_game(args):
    """运行一局游戏"""
    # 创建环境
    env = WerewolfEnv({})
    
    # 创建智能体
    agents = []
    for i in range(args.num_players):
        agent_type = args.agent_types[i % len(args.agent_types)]
        if agent_type == 'rl':
            agent = create_rl_agent(i, args.device)
            if args.load_model:
                agent.load_model(args.load_model)
        else:
            agent = create_agent(agent_type, i)
        agents.append(agent)
    
    # 重置环境
    observations = env.reset()
    
    # 初始化智能体
    for i, agent in enumerate(agents):
        agent.initialize(env.game_state)
    
    # 游戏主循环
    done = False
    while not done:
        # 渲染
        if args.render:
            env.render()
            time.sleep(0.5)  # 慢速显示
        
        # 可视化信念状态
        if args.visualize and isinstance(agents[0], (HeuristicAgent, RLAgent)):
            believer_id = 0
            belief_visualizer = BeliefVisualizer()
            timestamp = int(time.time())
            belief_visualizer.generate_belief_report(
                agents[believer_id].belief_updater.belief_state,
                env.game_state,
                believer_id,
                f"./visualizations/belief_{timestamp}.png"
            )
        
        current_player = env.game_state.current_player
        
        # 如果当前阶段没有特定玩家行动，则跳过
        if current_player is None:
            # 游戏阶段转换
            observations, rewards, done, infos = env.step(None)
            continue
        
        # 获取当前玩家的智能体
        agent = agents[current_player]
        
        # 智能体决策
        action = agent.act(observations[current_player])
        
        # 执行行动
        observations, rewards, done, infos = env.step(action)
        
        # 对RL智能体进行更新
        if isinstance(agent, RLAgent):
            agent.update(rewards[current_player], done)
    
    # 游戏结束
    if args.render:
        env.render()
        print(f"游戏结束! 胜利方: {env.game_state.winner}")
    
    return {
        'winner': env.game_state.winner,
        'num_turns': env.game_state.turn
    }


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 确保目录存在
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./visualizations', exist_ok=True)
    
    if args.mode == 'play':
        # 运行单局游戏
        result = play_game(args)
        print(f"游戏结果: {result}")
        
    elif args.mode == 'train':
        # 训练模式
        trainer = Trainer(num_players=args.num_players)
        
        if 'rl' in args.agent_types:
            # 深度强化学习训练
            print("训练RL智能体...")
            # 这里应该实现RL智能体的训练逻辑
            # 目前使用简单的自我对弈作为示例
            trainer.train_self_play(
                initial_agent_types=args.agent_types,
                num_generations=args.num_generations,
                episodes_per_generation=args.train_episodes // args.num_generations,
                render=args.render,
                visualize=args.visualize
            )
        else:
            # 启发式智能体训练/评估
            print("评估启发式智能体...")
            trainer.evaluate(
                agent_types=args.agent_types,
                num_episodes=args.train_episodes,
                render=args.render,
                visualize=args.visualize
            )
        
    elif args.mode == 'evaluate':
        # 评估模式
        trainer = Trainer(num_players=args.num_players)
        
        print(f"评估智能体: {args.agent_types}...")
        eval_results = trainer.evaluate(
            agent_types=args.agent_types,
            num_episodes=args.eval_episodes,
            render=args.render,
            visualize=args.visualize
        )
        
        print("评估结果:")
        print(f"狼人胜率: {eval_results['werewolf_win_rate']:.2f}")
        print(f"村民胜率: {eval_results['villager_win_rate']:.2f}")
        print(f"平均游戏长度: {eval_results['avg_game_length']:.2f} 轮")
        
        # 保存结果
        trainer.save_results(eval_results, f"eval_{'-'.join(args.agent_types)}")


if __name__ == "__main__":
    main() 