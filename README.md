# 狼人杀强化学习项目

这个项目实现了一个用于狼人杀游戏的强化学习环境和智能体。项目使用Gymnasium框架实现游戏环境，并基于PyTorch实现了强化学习算法来训练智能体。

## 项目结构

```
├── agents/               # 智能体实现
│   ├── base_agent.py     # 基础智能体、随机智能体和启发式智能体
├── config/               # 配置文件
│   ├── default_config.py # 默认游戏配置
├── models/               # 模型实现
│   ├── rl_agent.py       # 强化学习智能体和神经网络模型
├── train/                # 训练相关代码
│   ├── rl_trainer.py     # 强化学习训练器
├── utils/                # 工具函数
│   ├── belief_updater.py # 信念更新器
├── werewolf_env/         # 游戏环境
│   ├── env.py            # 主环境类
│   ├── state.py          # 游戏状态
│   ├── actions.py        # 动作定义
│   ├── roles.py          # 角色定义
├── main.py               # 游戏主程序
├── run_training.py       # 训练入口
├── test_env.py           # 环境测试
├── requirements.txt      # 依赖包
```

## 依赖包

项目依赖以下Python包：

```
gymnasium>=0.28.1
numpy>=1.24.0
torch>=2.0.0
stable-baselines3>=2.0.0
pettingzoo>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
wandb>=0.15.0
```

## 安装

1. 克隆此仓库
2. 安装依赖包：

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练智能体

使用以下命令开始训练：

```bash
python run_training.py --num_episodes 5000 --batch_size 4
```

主要参数说明：

- `--num_episodes`: 训练局数
- `--batch_size`: 批次大小
- `--learning_rate`: 学习率
- `--gamma`: 折扣因子
- `--use_cuda`: 是否使用GPU训练
- `--use_wandb`: 是否使用wandb记录实验

更多参数可以通过`python run_training.py --help`查看。

### 继续训练

如果要从之前保存的模型继续训练：

```bash
python run_training.py --continue_training --model_path ./models/saved/model_episode_1000.pt
```

### 自定义训练

您也可以通过导入训练器来自定义训练流程：

```python
from train import RLTrainer

trainer = RLTrainer(
    num_players=6,
    obs_dim=128,
    action_dim=100,
    learning_rate=0.0003
)

# 开始训练
trainer.train(num_episodes=5000)
```

## 游戏规则

这个项目实现了狼人杀游戏的一个简化版本，包括以下角色：

1. 狼人 (Werewolf)
2. 村民 (Villager)
3. 预言家 (Seer)
4. 强盗 (Robber)
5. 捣蛋鬼 (Troublemaker)
6. 失眠者 (Insomniac)
7. 爪牙 (Minion)

游戏流程包括以下阶段：

1. 夜晚阶段：各角色按顺序执行各自的特殊能力
2. 白天阶段：玩家讨论和发言
3. 投票阶段：所有玩家投票决定驱逐一名玩家
4. 结算阶段：判断游戏胜负

## 智能体

项目实现了三种类型的智能体：

1. 随机智能体 (RandomAgent)：随机选择合法动作
2. 启发式智能体 (HeuristicAgent)：基于规则策略选择动作
3. 强化学习智能体 (RLAgent)：基于神经网络模型学习策略

## 模型

强化学习智能体使用了一个包含以下组件的神经网络模型：

1. 特征提取器
2. 角色嵌入层
3. 玩家嵌入层
4. LSTM层处理历史信息
5. 策略头输出动作概率
6. 价值头输出状态价值

## 贡献

欢迎提交issue和pull request来改进这个项目。

## 许可证

MIT 许可证
