"""
训练第一阶段（启发式引导训练）的GPU加速脚本
支持多线程和GPU训练
"""
import argparse
import os
import torch
import numpy as np
import time
from tqdm import tqdm

from train.multi_stage.stage1_trainer import Stage1Trainer
from config.default_config import DEFAULT_GAME_CONFIG
from agents.heuristic_agent import HeuristicAgent


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='狼人杀第一阶段GPU加速训练')
    
    # 训练参数
    parser.add_argument('--num_episodes', type=int, default=1000, help='训练局数')
    parser.add_argument('--evaluate_every', type=int, default=100, help='每多少局评估一次')
    parser.add_argument('--save_every', type=int, default=200, help='每多少局保存一次模型')
    parser.add_argument('--render_every', type=int, default=500, help='每多少局渲染一次')
    
    # 环境参数
    parser.add_argument('--num_players', type=int, default=6, help='玩家数量')
    
    # 目录参数
    parser.add_argument('--log_dir', type=str, default='./logs/stage1_gpu', help='日志目录')
    parser.add_argument('--save_dir', type=str, default='./models/stage1_gpu', help='模型保存目录')
    parser.add_argument('--visualize_dir', type=str, default='./visualizations/stage1_gpu', help='可视化保存目录')
    
    # GPU和多线程参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='计算设备')
    parser.add_argument('--num_workers', type=int, default=4, help='并行工作进程数量')
    parser.add_argument('--batch_size', type=int, default=16, help='训练批量大小')
    parser.add_argument('--use_amp', action='store_true', help='是否使用混合精度训练')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--debug', action='store_true', help='调试模式（减少训练局数）')
    
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.visualize_dir, exist_ok=True)
    
    # 调试模式
    if args.debug:
        args.num_episodes = 20
        args.evaluate_every = 5
        args.save_every = 10
        args.render_every = 5
        print("调试模式: 减少训练参数")
    
    # 环境配置
    env_config = DEFAULT_GAME_CONFIG.copy()
    env_config.update({
        'num_players': args.num_players
    })
    
    # 输出GPU信息
    if args.device == 'cuda' and torch.cuda.is_available():
        print(f"使用GPU训练: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB")
        print(f"最大GPU内存使用: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f}MB")
        
        # 清理GPU内存
        torch.cuda.empty_cache()
    else:
        print("使用CPU训练")
    
    print(f"并行进程数: {args.num_workers}")
    print(f"游戏配置: {env_config}")
    
    # 创建训练器
    trainer = Stage1Trainer(
        env_config=env_config,
        num_players=args.num_players,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        visualize_dir=args.visualize_dir,
        device=args.device,
        num_workers=args.num_workers
    )
    
    # 配置混合精度训练
    scaler = None
    if args.use_amp and args.device == 'cuda':
        print("启用混合精度训练")
        scaler = torch.cuda.amp.GradScaler()
    
    # 开始训练
    print(f"\n======= 开始第一阶段训练 (GPU加速) =======")
    print(f"训练局数: {args.num_episodes}")
    
    start_time = time.time()
    
    # 重写train方法，优化为GPU训练
    def create_agents_factory(model, device):
        """创建一个返回agent列表的工厂函数"""
        from models.rl_agent import RLAgent
        
        def create_agents():
            agents = []
            # 随机选择一个位置放置RL智能体
            rl_index = np.random.randint(0, args.num_players)
            
            for i in range(args.num_players):
                if i == rl_index:
                    # RL智能体
                    agents.append(RLAgent(i, model=model, device=device))
                else:
                    # 启发式智能体
                    agents.append(HeuristicAgent(i))
            return agents
        
        return create_agents
    
    # 训练循环
    pbar = tqdm(range(args.num_episodes), desc="GPU训练进度")
    for episode in pbar:
        # 创建智能体工厂
        create_agents_fn = create_agents_factory(trainer.model, args.device)
        
        # 使用并行运行多局游戏
        results = trainer.run_parallel_episodes(create_agents_fn, 1, training=True)
        result = results[0]
        
        # 更新统计数据
        trainer.stats['total_games'] += 1
        trainer.stats['game_lengths'].append(result['game_length'])
        
        # 获取RL智能体的索引
        for player_id, player_data in result.get('training_data', {}).items():
            # 找到RL智能体的训练数据
            if player_data is not None:
                # 更新模型
                if args.use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss = update_model_with_data(trainer.model, trainer.optimizer, player_data, args.batch_size, scaler)
                else:
                    loss = update_model_with_data(trainer.model, trainer.optimizer, player_data, args.batch_size)
                
                trainer.stats['losses'].append(loss)
                
                # 获取奖励
                reward = result['total_rewards'].get(int(player_id), 0.0)
                trainer.stats['rewards'].append(reward)
                
                pbar.set_description(f"Loss: {loss:.4f}, Reward: {reward:.4f}")
                break
        
        # 记录胜利情况
        if result['winner'] == 'werewolf':
            trainer.stats['werewolf_wins'] += 1
        elif result['winner'] == 'villager':
            trainer.stats['villager_wins'] += 1
        
        # 定期评估
        if episode % args.evaluate_every == args.evaluate_every - 1:
            eval_create_agents_fn = create_agents_factory(trainer.model, args.device)
            eval_result = trainer.parallel_evaluate(eval_create_agents_fn, 10)
            
            print(f"\n评估结果 (第{episode+1}局):")
            print(f"狼人胜率: {eval_result['werewolf_win_rate']:.2f}")
            print(f"村民胜率: {eval_result['villager_win_rate']:.2f}")
            print(f"平均奖励: {eval_result['avg_reward']:.2f}")
            
            # 输出GPU内存使用情况
            if args.device == 'cuda':
                print(f"当前GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB")
                print(f"最大GPU内存使用: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f}MB")
                
                # 清理GPU内存
                torch.cuda.empty_cache()
        
        # 定期保存模型
        if episode % args.save_every == args.save_every - 1:
            trainer.save_model(f"{args.save_dir}/model_episode_{episode+1}.pt")
    
    end_time = time.time()
    print(f"\n训练完成，总用时: {end_time - start_time:.2f}秒")
    
    # 保存最终模型
    trainer.save_model(f"{args.save_dir}/model_final.pt")
    
    # 最终评估
    print("\n======= 最终模型评估 =======")
    eval_create_agents_fn = create_agents_factory(trainer.model, args.device)
    eval_result = trainer.parallel_evaluate(eval_create_agents_fn, 100)
    
    print(f"狼人胜率: {eval_result['werewolf_win_rate']:.2f}")
    print(f"村民胜率: {eval_result['villager_win_rate']:.2f}")
    print(f"平均奖励: {eval_result['avg_reward']:.2f}")
    
    # 保存训练器状态
    trainer.save_stats(os.path.join(args.log_dir, "training_stats.json"))
    print("训练完成!")


def update_model_with_data(model, optimizer, training_data, batch_size, scaler=None):
    """使用收集的训练数据更新模型"""
    states, log_probs, values, rewards = training_data
    
    if not states or not log_probs:
        return 0.0
    
    # 使用批量处理进行更新
    num_samples = len(states)
    total_loss = 0.0
    
    for i in range(0, num_samples, batch_size):
        batch_states = states[i:i + batch_size]
        batch_log_probs = log_probs[i:i + batch_size]
        batch_values = values[i:i + batch_size]
        batch_rewards = rewards[i:i + batch_size]
        
        # 计算优势
        advantages = []
        for j in range(len(batch_rewards)):
            advantage = batch_rewards[j] - batch_values[j].detach()
            advantages.append(advantage)
        
        # 将优势转换为张量
        advantages_tensor = torch.cat(advantages)
        
        # 计算策略损失
        policy_loss = -torch.cat(batch_log_probs) * advantages_tensor
        
        # 计算价值损失
        value_loss = torch.nn.functional.mse_loss(
            torch.cat(batch_values), 
            torch.cat(batch_rewards)
        )
        
        # 计算熵奖励（鼓励探索）
        entropy = torch.cat([e.mean() for e in batch_log_probs])
        
        # 总损失
        loss = policy_loss.mean() + 0.5 * value_loss - 0.01 * entropy.mean()
        
        # 更新模型
        optimizer.zero_grad()
        
        if scaler is not None:
            # 使用混合精度训练
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 常规训练
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    # 平均损失
    return total_loss / (num_samples // batch_size + 1)


if __name__ == "__main__":
    main() 