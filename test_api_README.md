# 狼人游戏API测试脚本

这个测试脚本用于测试狼人游戏的API端点，确保所有功能正常工作。脚本提供了三种测试模式：完整游戏模拟、单独API测试和特定行动测试。

## 前提条件

1. 确保狼人游戏API服务器正在运行（端口18000）
2. 安装所需Python依赖：
   ```
   pip install requests
   ```

## 使用方法

### 方法一：直接运行测试脚本

```
python test_api.py
```

### 方法二：使用带虚拟环境的启动脚本（推荐）

**macOS/Linux**：
```
chmod +x run_test_with_venv.sh
./run_test_with_venv.sh
```

**Windows**：
```
run_test_with_venv.bat
```

这些脚本会自动：
1. 创建并激活虚拟环境
2. 安装必要的依赖
3. 检查API服务器是否运行
4. 如果需要则启动API服务器
5. 运行测试脚本

### 解决"externally-managed-environment"错误

如果您在macOS或某些Linux系统上遇到"externally-managed-environment"错误，请使用虚拟环境：

```bash
# 创建虚拟环境
python3 -m venv werewolf_venv

# 激活虚拟环境
source werewolf_venv/bin/activate  # macOS/Linux
werewolf_venv\Scripts\activate.bat  # Windows

# 安装依赖
pip install requests

# 运行测试
python test_api.py
```

完成后，可以使用`deactivate`命令退出虚拟环境。

## 测试模式详解

### 模式1: 完整游戏模拟

此模式会创建一个测试游戏并使用step API自动运行整个游戏流程，直到游戏结束。这是最简单的全流程测试方式，适合验证整体游戏流程是否正常。

### 模式2: 单独API测试

此模式会依次测试各个API端点：
1. 创建游戏
2. 加入游戏
3. 获取游戏状态
4. 执行多种夜间行动（狼人、预言家、捣蛋鬼等）
5. 执行白天发言
6. 执行投票
7. 获取游戏结果

这种模式适合全面测试所有API功能。

### 模式3: 特定行动测试

此模式提供交互式菜单，可以选择测试特定的游戏行动：
1. 狼人夜间行动
2. 爪牙夜间行动
3. 预言家查看玩家
4. 预言家查看中央牌
5. 捣蛋鬼夜间行动
6. 强盗夜间行动
7. 白天发言
8. 投票
9. 获取游戏结果

这种模式适合针对性地测试特定功能或排查特定问题。

## API端点清单

测试脚本覆盖了以下API端点：

1. `/api/game/create` - 创建新游戏
2. `/api/game/create-test` - 创建测试游戏
3. `/api/game/join/{game_id}` - 加入游戏
4. `/api/game/state/{game_id}` - 获取游戏状态
5. `/api/game/ai-decision` - 获取AI决策
6. `/api/game/action` - 执行玩家行动（夜间行动、白天发言、投票）
7. `/api/game/step` - 执行游戏步骤（自动模拟）
8. `/api/game/result/{game_id}` - 获取游戏结果

## 故障排除

1. 如果连接服务器失败，请确保API服务器正在运行：
   ```
   python app.py
   ```

2. 如果端口被占用，可能需要修改BASE_URL中的端口号，或终止占用该端口的进程后重新启动服务器。

3. 如果特定行动测试失败，请检查：
   - 玩家ID是否有效
   - 该玩家是否拥有执行该行动的角色
   - 游戏当前是否处于允许该行动的阶段

4. 如果遇到PIP安装问题，请使用虚拟环境启动脚本或手动创建虚拟环境。

## 输出说明

测试脚本使用彩色输出来区分不同类型的信息：
- 蓝色：请求数据
- 绿色：成功响应
- 红色：错误信息
- 黄色：测试阶段提示 