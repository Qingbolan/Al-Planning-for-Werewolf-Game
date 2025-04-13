"""
多阶段训练脚本 - 狼人杀AI
"""
import argparse
import os
import torch
import random
import numpy as np
from typing import Dict, Any

from train.multi_stage.multi_stage_trainer import MultiStageTrainer
from config.default_config import DEFAULT_GAME_CONFIG


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='狼人杀AI多阶段训练')
    
    # 基本训练参数
    parser.add_argument('--stage1_episodes', type=int, default=1000, help='第一阶段训练局数')
    parser.add_argument('--stage2_episodes', type=int, default=2000, help='第二阶段训练局数')
    parser.add_argument('--stage3_episodes', type=int, default=3000, help='第三阶段训练局数')
    parser.add_argument('--evaluate_every', type=int, default=100, help='每多少局评估一次')
    parser.add_argument('--save_every', type=int, default=500, help='每多少局保存一次模型')
    parser.add_argument('--render_every', type=int, default=200, help='每多少局渲染一次')
    
    # 环境参数
    parser.add_argument('--num_players', type=int, default=6, help='玩家数量')
    parser.add_argument('--max_speech_rounds', type=int, default=3, help='发言轮数')
    parser.add_argument('--reverse_vote_rules', action='store_true', default=True, help='是否使用反转投票规则')
    
    # 目录参数
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志目录')
    parser.add_argument('--save_dir', type=str, default='./models', help='模型保存目录')
    parser.add_argument('--visualize_dir', type=str, default='./visualizations', help='可视化保存目录')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='计算设备')
    parser.add_argument('--debug', action='store_true', help='调试模式（减少训练局数）')
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_args()
        
        # 设置随机种子
        set_seed(args.seed)
        
        # 创建目录
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.visualize_dir, exist_ok=True)
        
        # 如果是调试模式，减少训练局数
        if args.debug:
            args.stage1_episodes = 20
            args.stage2_episodes = 30
            args.stage3_episodes = 50
            args.evaluate_every = 10
            args.save_every = 20
            args.render_every = 10
            print("调试模式：减少训练局数")
        
        # 修改游戏配置
        game_config: Dict[str, Any] = DEFAULT_GAME_CONFIG.copy()
        game_config.update({
            'num_players': args.num_players,
            'max_speech_rounds': args.max_speech_rounds,
            'reverse_vote_rules': args.reverse_vote_rules
        })
        
        print(f"使用设备: {args.device}")
        print(f"游戏配置: {game_config}")
        
        # 创建多阶段训练器
        trainer = MultiStageTrainer(
            env_config=game_config,
            num_players=args.num_players,
            log_dir=args.log_dir,
            save_dir=args.save_dir,
            visualize_dir=args.visualize_dir,
            device=args.device
        )
        
        # 开始训练
        print("\n======= 开始多阶段训练 =======")
        final_model = trainer.train(
            stage1_episodes=args.stage1_episodes,
            stage2_episodes=args.stage2_episodes,
            stage3_episodes=args.stage3_episodes,
            evaluate_every=args.evaluate_every,
            save_every=args.save_every,
            render_every=args.render_every
        )
        
        # 保存最终模型
        final_model_path = os.path.join(args.save_dir, 'final_model.pt')
        torch.save(final_model.state_dict(), final_model_path)
        print(f"\n最终模型已保存至: {final_model_path}")
        
        # 评估最终模型
        print("\n======= 最终模型评估 =======")
        eval_results = trainer.evaluate(num_episodes=100, render=False)
        
        print(f"狼人胜率: {eval_results['werewolf_win_rate']:.2f}")
        print(f"村民胜率: {eval_results['villager_win_rate']:.2f}")
        print(f"平均奖励: {eval_results['avg_reward']:.2f}")
        
        print("\n多阶段训练完成!")
    except Exception as e:
        import traceback
        print(f"\n发生错误: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()
        
        # 打印特征尺寸等调试信息
        # if 'state_features' in locals():
        #     print(f"\n特征向量尺寸: {state_features.shape}")
        
        # if 'model' in locals():
        #     print("\n模型结构:")
        #     print(model)


if __name__ == "__main__":
    main() 