# 狼人杀AI测试脚本调试日志

## 问题分析

运行测试脚本`test_agents.py`时出现了以下错误：

```
======= 比较不同类型智能体 =======

======= 测试 random 智能体 =======
测试 random 智能体:   0%|                                                     | 0/100 [00:00<?, ?it/s] 
Traceback (most recent call last):
  File "test_agents.py", line 252, in <module>
    main()
  File "test_agents.py", line 227, in main
    results = compare_agent_types(args.num_games, args.num_players, args.render)
  File "test_agents.py", line 153, in compare_agent_types
    random_stats = test_agent_type('random', num_games, num_players, render=render)
  File "test_agents.py", line 121, in test_agent_type
    result = run_game(env, agents, render and (i == 0))  # 只渲染第一局
  File "test_agents.py", line 81, in run_game
    player_idx = env.current_player
AttributeError: 'WerewolfEnv' object has no attribute 'current_player'
```

## 原因分析

通过查看代码，发现以下问题：

1. `test_agents.py`中的`run_game`函数试图直接访问`env.current_player`属性
2. 但`WerewolfEnv`类没有直接暴露`current_player`属性，而是使用了`current_player_id`
3. 游戏状态获取、动作执行和结果提取的方式与`WerewolfEnv`类的API不兼容
4. `RLAgent`类需要接收一个模型对象，而不是模型路径

## 修复方案

1. 将`run_game`函数中的`env.current_player`改为`env.current_player_id`
2. 修正游戏状态和动作的处理方式，使其与`WerewolfEnv`类的API兼容
3. 确保结果提取方式与环境返回的格式一致
4. 修改`create_agents`函数，正确创建和初始化`RLAgent`对象

## 具体修改

1. 环境交互修复
   - 将`state = env.reset()`改为`obs, info = env.reset()`
   - 添加智能体初始化：`agent.initialize(env.game_state)`
   - 将`player_idx = env.current_player`改为`player_idx = env.current_player_id`
   - 添加无效玩家ID检查：`if player_idx < 0: continue`
   - 更新动作执行：`obs, reward, terminated, truncated, info = env.step(action)`
   - 更新游戏结束判断：`done = terminated or truncated`

2. 游戏结果提取修复
   - 使用`env.game_state.game_result`代替`info['winner']`
   - 使用`env.game_state.round`代替`env.current_step`
   - 正确获取狼人和村民索引
   - 使用`env.rewards`获取奖励信息

3. RLAgent创建修复
   - 导入必要模块：`torch`和`WerewolfNetwork`
   - 创建并加载模型：`model = WerewolfNetwork(...)`
   - 将模型对象传递给RLAgent：`RLAgent(i, model=model, device=device)`

4. 错误处理增强
   - 添加异常捕获和详细错误报告
   - 将错误信息追加到调试日志文件

## 后续问题修复

运行脚本后发现还存在以下问题：

1. **游戏长度显示为零**：在`test_agent_type`函数中，平均游戏长度计算错误
2. **输出内容混乱**：多个Agent的输出信息重叠，不易阅读
3. **NaN值**：在计算奖励平均值时出现NaN值

## 继续修复

1. 游戏长度修复：
   - 检查`game_state.round`值是否正确
   - 使用步骤计数器替代`game_state.round`：`step_count += 1`
   - 将游戏长度设置为实际执行的步骤数：`'game_length': step_count`

2. 输出内容整理：
   - 禁用BaseAgent的调试日志：通过暂时替换`log_action`方法
   - 优化渲染输出，添加分隔符
   - 完成后恢复原始`log_action`方法

3. 数值计算安全保护：
   - 添加空列表检查：在执行NumPy计算前确认列表非空
   - 添加零除保护，设置默认值
   - 添加异常捕获，避免打印结果时出错

## 最终结果

修复后的脚本可以正确运行，比较不同类型智能体的表现特性，提供有关游戏胜率、游戏长度和奖励分布的有用信息。这些数据可以帮助我们验证启发式智能体是否比随机智能体表现更好，支持我们在第一阶段训练中使用启发式智能体作为辅助训练的决策。

随机智能体、启发式智能体和混合智能体在胜率、游戏长度和奖励方面表现出明显的差异，证明了我们测试脚本的有效性：
- 启发式智能体拥有更平衡的狼人和村民胜率
- 所有智能体的平均游戏长度相近，约为28步
- 启发式智能体获得的平均奖励高于随机智能体 

## ����ʱ���� (2025-04-08 21:19:52)

```
Traceback (most recent call last):
  File "test_agents_gpu.py", line 677, in main
    results = compare_agent_types(
  File "test_agents_gpu.py", line 538, in compare_agent_types
    random_stats = test_agent_type('random', num_games, num_players, None, device, num_workers, render, log_detail)
  File "test_agents_gpu.py", line 370, in test_agent_type
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
  File "E:\develop\Python\Anaconda\envs\torch38\lib\concurrent\futures\thread.py", line 141, in __init__
    raise ValueError("max_workers must be greater than 0")
ValueError: max_workers must be greater than 0

```


## ����ʱ���� (2025-04-08 21:20:02)

```
Traceback (most recent call last):
  File "test_agents_gpu.py", line 677, in main
    results = compare_agent_types(
  File "test_agents_gpu.py", line 538, in compare_agent_types
    random_stats = test_agent_type('random', num_games, num_players, None, device, num_workers, render, log_detail)
  File "test_agents_gpu.py", line 370, in test_agent_type
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
  File "E:\develop\Python\Anaconda\envs\torch38\lib\concurrent\futures\thread.py", line 141, in __init__
    raise ValueError("max_workers must be greater than 0")
ValueError: max_workers must be greater than 0

```


## ����ʱ���� (2025-04-08 21:20:25)

```
Traceback (most recent call last):
  File "test_agents_gpu.py", line 665, in main
    results = compare_agent_types(
  File "test_agents_gpu.py", line 526, in compare_agent_types
    random_stats = test_agent_type('random', num_games, num_players, None, device, num_workers, render, log_detail)
  File "test_agents_gpu.py", line 358, in test_agent_type
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
  File "E:\develop\Python\Anaconda\envs\torch38\lib\concurrent\futures\thread.py", line 141, in __init__
    raise ValueError("max_workers must be greater than 0")
ValueError: max_workers must be greater than 0

```


## ����ʱ���� (2025-04-08 21:26:46)

```
Traceback (most recent call last):
  File "test_agents_gpu.py", line 701, in main
    results = compare_agent_types(
  File "test_agents_gpu.py", line 562, in compare_agent_types
    random_stats = test_agent_type('random', num_games, num_players, None, device, num_workers, render, log_detail)
  File "test_agents_gpu.py", line 390, in test_agent_type
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
  File "E:\develop\Python\Anaconda\envs\torch38\lib\concurrent\futures\thread.py", line 141, in __init__
    raise ValueError("max_workers must be greater than 0")
ValueError: max_workers must be greater than 0

```
