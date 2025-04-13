# 狼人杀多阶段训练系统设计

## 概述

狼人杀是一款复杂的多智能体博弈游戏，涉及不完全信息、社交推理和多角色互动。为了训练出高效的AI代理，我们提出了一种多阶段训练系统，将训练过程分为三个渐进式阶段，逐步提高AI代理的策略复杂度和性能。

## 系统架构

多阶段训练系统实现在 `train/multi_stage_trainer.py` 文件中的 `MultiStageTrainer` 类，以及各个阶段的具体训练器：
- `Stage1Trainer`: 启发式引导训练阶段
- `Stage2Trainer`: 混合训练阶段
- `Stage3Trainer`: 自对弈训练阶段

### 核心组件

1. **环境设置**：使用狼人杀游戏环境 `WerewolfEnv`
2. **模型结构**：使用 `WerewolfNetwork` 作为策略网络
3. **阶段控制**：通过各阶段训练器管理不同训练策略
4. **数据收集与分析**：记录训练过程中的胜率、奖励等指标

## 多阶段训练流程

### 第一阶段：启发式引导训练

在第一阶段，只训练一个或少数几个强化学习代理，其他角色由规则型代理扮演。

| 特性 | 描述 |
|------|------|
| 目标 | 使RL代理学习基本游戏规则和策略 |
| 代理组成 | 1个RL代理 + 多个规则型代理 |
| 训练重点 | 基础技能、角色理解、动作选择 |
| 终止条件 | 达到预设胜率或训练回合数 |

关键实现：
```python
# 第一阶段训练示例
def train_stage1(self, num_episodes):
    rl_agent = self.create_rl_agent(role="werewolf")
    heuristic_agents = self.create_heuristic_agents()
    
    for episode in range(num_episodes):
        # 在规则代理环境中训练RL代理
        experiences = self.run_episode([rl_agent] + heuristic_agents)
        self.update_model(experiences)
```

### 第二阶段：混合训练

第二阶段引入多个RL代理与规则型代理混合训练。

| 特性 | 描述 |
|------|------|
| 目标 | 提高代理间的交互能力和策略多样性 |
| 代理组成 | 多个RL代理（使用第一阶段模型初始化）+ 少量规则型代理 |
| 训练重点 | 代理间博弈、信息推理、欺骗与合作 |
| 终止条件 | 达到稳定的策略分布或训练回合数 |

关键实现：
```python
# 第二阶段训练示例
def train_stage2(self, num_episodes):
    # 加载第一阶段训练好的模型
    stage1_model = self.load_model(self.stage1_model_path)
    
    # 创建多个使用相同基础模型的RL代理
    rl_agents = [self.create_rl_agent(role=role, model=stage1_model.copy()) 
                for role in ["werewolf", "villager", "seer"]]
    
    # 添加少量规则型代理补充
    heuristic_agents = self.create_heuristic_agents(num_agents=3)
    
    for episode in range(num_episodes):
        # 所有代理混合训练
        experiences = self.run_episode(rl_agents + heuristic_agents)
        # 每个RL代理分别更新
        for agent_idx, agent in enumerate(rl_agents):
            agent_exp = experiences[agent_idx]
            self.update_model(agent_exp, agent.model)
```

### 第三阶段：自对弈训练

第三阶段完全使用RL代理进行自对弈训练。

| 特性 | 描述 |
|------|------|
| 目标 | 生成高水平、多样化的对抗策略 |
| 代理组成 | 全部为RL代理，使用第二阶段的模型 |
| 训练重点 | 高级策略、自适应对抗、元策略学习 |
| 终止条件 | 策略收敛或达到预设性能水平 |

关键实现：
```python
# 第三阶段训练示例
def train_stage3(self, num_episodes):
    # 加载第二阶段训练好的模型
    stage2_model = self.load_model(self.stage2_model_path)
    
    # 所有角色都使用RL代理
    rl_agents = [self.create_rl_agent(role=role, model=stage2_model.copy()) 
                for role in self.all_roles]
    
    for episode in range(num_episodes):
        # 完全自对弈
        experiences = self.run_episode(rl_agents)
        # 应用多种技术优化模型
        self.apply_advanced_techniques(experiences)
        # 模型更新
        self.update_all_models(experiences)
```

## 训练技术与优化

### 经验回放

使用经验回放缓冲区存储和重用过去的经验：

```python
# 经验回放实现示例
class ExperienceBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        
    def add(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

### 优先经验回放

基于TD误差的优先级采样，关注更有价值的经验：

```python
def prioritized_sample(self):
    # 基于TD误差计算优先级
    priorities = [abs(exp.td_error) + self.epsilon for exp in self.buffer]
    probabilities = [p / sum(priorities) for p in priorities]
    # 按优先级采样
    indices = np.random.choice(len(self.buffer), self.batch_size, p=probabilities)
    return [self.buffer[i] for i in indices]
```

### 课程学习

从简单场景逐渐过渡到复杂场景：

```python
# 课程学习示例
def apply_curriculum(self, episode):
    # 根据训练进度调整环境复杂度
    if episode < self.curriculum_thresholds[0]:
        # 简单场景：少量玩家，基本角色
        return {"num_players": 4, "roles": ["werewolf", "villager", "villager", "seer"]}
    elif episode < self.curriculum_thresholds[1]:
        # 中等场景：增加玩家数量和角色种类
        return {"num_players": 6, "roles": ["werewolf", "werewolf", "villager", "villager", "seer", "robber"]}
    else:
        # 完整游戏配置
        return {"num_players": 8, "roles": self.full_role_set}
```

## 评估与分析

### 性能指标

系统会跟踪以下关键指标：

1. **胜率**：各阵营的胜率统计
2. **平均奖励**：代理获得的平均奖励值
3. **策略熵**：衡量策略的随机性/确定性
4. **行动分布**：各种行动的选择频率

### 可视化工具

训练过程包含多种可视化工具：

```python
def plot_training_metrics(self):
    # 绘制训练曲线
    plt.figure(figsize=(12, 8))
    
    # 胜率曲线
    plt.subplot(2, 2, 1)
    plt.plot(self.win_rates["werewolf"], label="狼人胜率")
    plt.plot(self.win_rates["villager"], label="村民胜率")
    plt.legend()
    
    # 奖励曲线
    plt.subplot(2, 2, 2)
    plt.plot(self.rewards_history)
    plt.title("平均奖励")
    
    # 损失曲线
    plt.subplot(2, 2, 3)
    plt.plot(self.loss_history)
    plt.title("损失函数")
    
    # 保存图表
    plt.savefig(os.path.join(self.vis_dir, "training_metrics.png"))
```

## 目前实现状态

多阶段训练系统已经完成基础框架设计，包括：

- `MultiStageTrainer` 类的核心结构
- `Stage1Trainer` 的完整实现
- 基础的评估和数据记录功能

### 待实现功能

1. **完整的 Stage2Trainer 和 Stage3Trainer 实现**
2. **高级训练技术集成**：策略蒸馏、对抗训练
3. **分布式训练支持**：支持多GPU/多机训练
4. **超参数自动优化**：使用贝叶斯优化等方法优化超参数

## 使用指南

### 配置与启动

通过以下方式使用多阶段训练系统：

```python
# 创建多阶段训练器
trainer = MultiStageTrainer(
    env_config={
        "num_players": 6,
        "roles": ["werewolf", "werewolf", "villager", "villager", "seer", "robber"]
    },
    log_dir="logs",
    save_dir="models",
    vis_dir="visualizations"
)

# 启动训练
trainer.train(
    stage1_episodes=10000,
    stage2_episodes=20000,
    stage3_episodes=30000,
    eval_interval=1000
)
```

### 模型保存与加载

训练过程中会定期保存模型，也支持手动保存和加载：

```python
# 保存模型
trainer.save_model("custom_model_name")

# 加载模型继续训练
trainer.load_model("path/to/model")
trainer.continue_training(additional_episodes=5000)
```

## 参考资料

- [OpenAI五对五Dota2训练方法](https://openai.com/blog/openai-five/)
- [DeepMind AlphaGo训练系统](https://www.nature.com/articles/nature16961)
- [多智能体强化学习最新进展](https://arxiv.org/abs/1911.10635)
- [POMDP环境中的策略学习](https://arxiv.org/abs/2005.13341) 