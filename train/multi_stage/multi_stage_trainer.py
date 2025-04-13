"""
多阶段训练器
"""
from typing import Dict, List, Any, Optional
import os
import json
import torch
import numpy as np

from train.base.base_trainer import BaseTrainer
from train.multi_stage.stage1_trainer import Stage1Trainer
from train.multi_stage.stage2_trainer import Stage2Trainer
from train.multi_stage.stage3_trainer import Stage3Trainer
from models.rl_agent import WerewolfNetwork


class MultiStageTrainer(BaseTrainer):
    """多阶段训练器"""
    
    def __init__(self,
                 env_config: Dict[str, Any],
                 num_players: int = 6,
                 log_dir: str = './logs',
                 save_dir: str = './models',
                 visualize_dir: str = './visualizations',
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化多阶段训练器
        
        Args:
            env_config: 环境配置
            num_players: 玩家数量
            log_dir: 日志目录
            save_dir: 模型保存目录
            visualize_dir: 可视化保存目录
            device: 计算设备
        """
        super().__init__(env_config, num_players, log_dir, save_dir, visualize_dir, device)
        
        # 初始化各阶段训练器
        self.stage1_trainer = Stage1Trainer(
            env_config=env_config,
            num_players=num_players,
            log_dir=os.path.join(log_dir, 'stage1'),
            save_dir=os.path.join(save_dir, 'stage1'),
            visualize_dir=os.path.join(visualize_dir, 'stage1'),
            device=device
        )
        
        self.stage2_trainer = Stage2Trainer(
            env_config=env_config,
            num_players=num_players,
            log_dir=os.path.join(log_dir, 'stage2'),
            save_dir=os.path.join(save_dir, 'stage2'),
            visualize_dir=os.path.join(visualize_dir, 'stage2'),
            device=device
        )
        
        self.stage3_trainer = Stage3Trainer(
            env_config=env_config,
            num_players=num_players,
            log_dir=os.path.join(log_dir, 'stage3'),
            save_dir=os.path.join(save_dir, 'stage3'),
            visualize_dir=os.path.join(visualize_dir, 'stage3'),
            device=device
        )
        
        # 训练状态
        self.current_stage = 1
        self.training_history = []
    
    def train(self,
              stage1_episodes: int = 1000,
              stage2_episodes: int = 2000,
              stage3_episodes: int = 3000,
              evaluate_every: int = 100,
              save_every: int = 500,
              render_every: int = 200) -> WerewolfNetwork:
        """
        执行多阶段训练
        
        Args:
            stage1_episodes: 第一阶段训练局数
            stage2_episodes: 第二阶段训练局数
            stage3_episodes: 第三阶段训练局数
            evaluate_every: 每多少局评估一次
            save_every: 每多少局保存一次模型
            render_every: 每多少局渲染一次
        """
        # 第一阶段：启发式引导训练
        print("开始第一阶段训练：启发式引导训练")
        self.current_stage = 1
        stage1_model = self.stage1_trainer.train(
            num_episodes=stage1_episodes,
            evaluate_every=evaluate_every,
            save_every=save_every,
            render_every=render_every
        )
        self.training_history.append({
            'stage': 1,
            'episodes': stage1_episodes,
            'model_path': os.path.join(self.save_dir, 'stage1', 'model_final.pt')
        })
        
        # 第二阶段：混合训练
        print("开始第二阶段训练：混合训练")
        self.current_stage = 2
        stage2_model = self.stage2_trainer.train(
            num_episodes=stage2_episodes,
            pretrained_model=stage1_model,
            evaluate_every=evaluate_every,
            save_every=save_every,
            render_every=render_every
        )
        self.training_history.append({
            'stage': 2,
            'episodes': stage2_episodes,
            'model_path': os.path.join(self.save_dir, 'stage2', 'model_final.pt')
        })
        
        # 第三阶段：自对抗训练
        print("开始第三阶段训练：自对抗训练")
        self.current_stage = 3
        stage3_model = self.stage3_trainer.train(
            num_episodes=stage3_episodes,
            pretrained_model=stage2_model,
            evaluate_every=evaluate_every,
            save_every=save_every,
            render_every=render_every
        )
        self.training_history.append({
            'stage': 3,
            'episodes': stage3_episodes,
            'model_path': os.path.join(self.save_dir, 'stage3', 'model_final.pt')
        })
        
        # 保存训练历史
        self.save_training_history()
        
        return stage3_model
    
    def evaluate(self, num_episodes: int = 100, render: bool = False) -> Dict[str, Any]:
        """
        评估当前阶段模型
        
        Args:
            num_episodes: 评估局数
            render: 是否渲染
            
        Returns:
            评估结果
        """
        if self.current_stage == 1:
            return self.stage1_trainer.evaluate(num_episodes, render)
        elif self.current_stage == 2:
            return self.stage2_trainer.evaluate(num_episodes, render)
        else:
            return self.stage3_trainer.evaluate(num_episodes, render)
    
    def save_training_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load_training_history(self) -> List[Dict[str, Any]]:
        """加载训练历史"""
        history_path = os.path.join(self.log_dir, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                return json.load(f)
        return [] 