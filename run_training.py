"""
狼人杀强化学习训练启动脚本
"""
import argparse
import torch
import random
import numpy as np
import os

from train.rl_trainer import RLTrainer
from config.default_config import DEFAULT_GAME_CONFIG


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='狼人杀强化学习训练')
    
    # 基本训练参数
    parser.add_argument('--num_episodes', type=int, default=5000, help='训练局数')
    parser.add_argument('--num_players', type=int, default=6, help='玩家数量')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--evaluate_every', type=int, default=100, help='每多少局评估一次')
    parser.add_argument('--save_every', type=int, default=500, help='每多少局保存一次模型')
    parser.add_argument('--render_every', type=int, default=200, help='每多少局渲染一次')
    
    # 模型参数
    parser.add_argument('--obs_dim', type=int, default=128, help='观察空间维度')
    parser.add_argument('--action_dim', type=int, default=100, help='动作空间维度')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    
    # 优化器参数
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='熵系数')
    parser.add_argument('--value_coef', type=float, default=0.5, help='价值函数系数')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='梯度裁剪范数')
    
    # 环境参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 路径参数
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志目录')
    parser.add_argument('--save_dir', type=str, default='./models/saved', help='模型保存目录')
    parser.add_argument('--visualize_dir', type=str, default='./visualizations', help='可视化保存目录')
    
    # 其他参数
    parser.add_argument('--use_cuda', action='store_true', help='是否使用CUDA')
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb记录实验')
    parser.add_argument('--continue_training', action='store_true', help='是否继续上次的训练')
    parser.add_argument('--model_path', type=str, default=None, help='继续训练时加载的模型路径')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 设置设备
    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.visualize_dir, exist_ok=True)
    
    # 创建训练器
    trainer = RLTrainer(
        env_config=DEFAULT_GAME_CONFIG,
        num_players=args.num_players,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        device=device,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        visualize_dir=args.visualize_dir,
        use_wandb=args.use_wandb
    )
    
    # 继续训练
    if args.continue_training and args.model_path:
        print(f"从 {args.model_path} 继续训练")
        trainer.load_model(args.model_path)
    
    # 开始训练
    print("开始训练...")
    stats = trainer.train(
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        opponent_types=['random', 'heuristic'],
        evaluate_every=args.evaluate_every,
        save_every=args.save_every,
        render_every=args.render_every
    )
    
    # 最终评估
    print("进行最终评估...")
    eval_result = trainer.evaluate(num_episodes=100, render=False)
    print("\n最终评估结果:")
    print(f"狼人胜率: {eval_result['werewolf_win_rate']:.2f}")
    print(f"村民胜率: {eval_result['villager_win_rate']:.2f}")
    print(f"平均奖励: {eval_result['avg_reward']:.2f}")
    
    print(f"\n训练结束! 模型已保存到 {args.save_dir}")


if __name__ == "__main__":
    main() 