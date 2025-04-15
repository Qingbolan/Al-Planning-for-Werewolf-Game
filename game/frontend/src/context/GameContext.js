import React, { createContext, useContext, useState, useEffect } from 'react';
import { gameApi } from '../services/api';

// 创建游戏上下文
const GameContext = createContext();

// 自定义钩子以使用游戏上下文
export const useGame = () => useContext(GameContext);

// 游戏提供者组件
export const GameProvider = ({ children }) => {
  // 游戏状态
  const [gameState, setGameState] = useState(null);
  const [player, setPlayer] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([]);
  const [gameResult, setGameResult] = useState(null);
  const [processing, setProcessing] = useState(false);
  
  // 默认玩家名称
  const defaultPlayerName = '玩家';

  // 从localStorage获取保存的游戏会话
  useEffect(() => {
    const savedGameId = localStorage.getItem('gameId');
    const savedPlayerId = localStorage.getItem('playerId');
    
    if (savedGameId && savedPlayerId) {
      console.log('正在恢复游戏会话:', { savedGameId, savedPlayerId });
      
      setPlayer({
        playerId: savedPlayerId,
        gameId: savedGameId,
        name: defaultPlayerName,
      });
      
      // 游戏会话恢复后，需要在用户执行下一步动作时更新状态
      // 不再主动获取游戏状态
    }
  }, []);

  // 创建游戏
  const createGame = async (config) => {
    try {
      setLoading(true);
      setError(null);
      
      // 清除旧的游戏会话
      localStorage.removeItem('gameId');
      localStorage.removeItem('playerId');
      
      console.log('发送游戏创建请求:', config);
      const data = await gameApi.createGame(config);
      console.log('游戏创建成功:', data);
      
      if (data && data.success) {
        // 设置游戏状态
        setGameState(data.state);
        
        // 从配置中找出人类玩家
        let humanPlayerId = null;
        let humanPlayerName = null;
        
        if (config.players) {
          for (const [id, playerInfo] of Object.entries(config.players)) {
            if (playerInfo.is_human) {
              humanPlayerId = parseInt(id, 10);
              humanPlayerName = playerInfo.name;
              break;
            }
          }
        }
        
        // 如果找到人类玩家，设置玩家状态
        if (humanPlayerId !== null) {
          setPlayer({
            playerId: humanPlayerId,
            name: humanPlayerName || defaultPlayerName
          });
          
          // 保存会话数据
          localStorage.setItem('playerId', humanPlayerId);
        } else {
          // 如果没有人类玩家，设置为观察者模式
          setPlayer({
            playerId: 'observer',
            name: 'Observer'
          });
          
          // 保存会话数据
          localStorage.setItem('playerId', 'observer');
        }
      }
      
      return data;
    } catch (error) {
      console.error('创建游戏失败:', error);
      setError('创建游戏失败: ' + (error.response?.data?.detail || error.message || '未知错误'));
      throw error;
    } finally {
      setLoading(false);
    }
  };

  // 创建测试游戏
  const createTestGame = async (testGameType = "heuristic", numPlayers = 6, seed = null) => {
    try {
      setLoading(true);
      setError(null);
    
      // 创建测试游戏
      const result = await gameApi.createTestGame(testGameType, numPlayers, seed);
      
      if (result.success) {
        console.log('测试游戏创建成功:', result);
        
        // 设置游戏状态
        setGameState(result.state);
        
        // 设置玩家信息 (观察者模式)
        setPlayer({
          playerId: 'observer',
          gameId: result.gameId,
          name: 'Observer'
        });
      } else {
        setError(result.error || '创建测试游戏时发生未知错误');
      }
      
      setLoading(false);
      return result;
    } catch (error) {
      console.error('创建测试游戏失败:', error);
      setError('创建测试游戏失败: ' + error.message);
      setLoading(false);
      throw error;
    }
  };

  // 加入游戏
  const joinGame = async (gameId, playerName) => {
    try {
      setLoading(true);
      setError(null);
      
      const result = await gameApi.joinGame(gameId, playerName);
      console.log('加入游戏成功:', result);
      
      if (result.success) {
        // 设置玩家信息
        setPlayer({
          playerId: result.player_id,
          name: playerName || defaultPlayerName
        });
        
        // 设置游戏状态
        setGameState(result.state);
      }
      
      setLoading(false);
      return result;
    } catch (error) {
      console.error('加入游戏失败:', error);
      setError('加入游戏失败: ' + (error.response?.data?.detail || error.message || '未知错误'));
      setLoading(false);
      throw error;
    }
  };

  // 执行游戏行动
  const performAction = async (action) => {
    try {
      setError(null);
      setLoading(true);
      
      if (!player || !player.gameId || !player.playerId) {
        throw new Error('玩家未加入游戏');
      }
      
      // 通过REST API发送行动
      const result = await gameApi.performAction(player.gameId, player.playerId, action);
      
      // 更新本地游戏状态
      if (result.state_update) {
        setGameState(prevState => ({
          ...prevState,
          ...result.state_update
        }));
        
        // 如果游戏已结束，获取游戏结果
        if (result.state_update.game_over) {
          fetchGameResult(player.gameId);
        }
      }
      
      setLoading(false);
      return result;
    } catch (error) {
      console.error('执行行动失败:', error);
      setError('执行行动失败: ' + (error.message || '未知错误'));
      setLoading(false);
      throw error;
    }
  };

  // 获取AI决策
  const getAIDecision = async (playerId) => {
    try {
      setLoading(true);
      setError(null);
      
      if (!player || !player.gameId) {
        throw new Error('游戏未创建');
      }
      
      // 在发送前规范化游戏状态
      const normalizedGameState = normalizeGameState(gameState);
      
      const result = await gameApi.getAIDecision(player.gameId, playerId, normalizedGameState);
      
      setLoading(false);
      return result;
    } catch (error) {
      console.error('获取AI决策失败:', error);
      setError('获取AI决策失败: ' + (error.message || '未知错误'));
      setLoading(false);
      throw error;
    }
  };

  // 规范化游戏状态，确保结构一致性
  const normalizeGameState = (gameState) => {
    if (!gameState) return null;
    
    const normalized = {...gameState};
    
    // 确保current_player字段一致性
    if ('current_player_id' in normalized && normalized.current_player_id !== undefined) {
      normalized.current_player = normalized.current_player_id;
      delete normalized.current_player_id;
    }
    
    return normalized;
  };

  // 重新实现 executeGameStep 函数，明确各个阶段的处理逻辑
  const executeGameStep = async () => {
    try {
      // 检查游戏状态是否存在
      if (!gameState) {
        console.error("无法执行游戏步骤：游戏状态不存在");
        return { success: false, error: "游戏状态不存在" };
      }

      // 标记处理中状态
      setProcessing(true);

      // 根据当前游戏阶段执行对应逻辑
      if (gameState.phase === 'night') {
        // 执行夜晚阶段
        return await executeNightStep();
      } else if (gameState.phase === 'day') {
        // 执行白天阶段
        return await executeDayStep();
      } else if (gameState.phase === 'vote') {
        // 执行投票阶段
        return await executeVoteStep();
      } else {
        console.error(`未知游戏阶段: ${gameState.phase}`);
        return { success: false, error: `未知游戏阶段: ${gameState.phase}` };
      }
    } catch (error) {
      console.error("执行游戏步骤时出错:", error);
      setError(`执行游戏步骤时出错: ${error.message}`);
      return { success: false, error: error.message };
    } finally {
      setProcessing(false);
    }
  };

  // 夜晚阶段执行逻辑
  const executeNightStep = async () => {
    // 复制当前游戏状态
    const updatedGameState = { ...gameState };
    
    // 如果当前玩家为空，则寻找第一个拥有夜间行动的玩家
    if (updatedGameState.current_player === null || updatedGameState.current_player === undefined) {
      console.log("当前玩家ID未定义，寻找第一个有夜间行动的玩家");
      moveToNextNightPlayer(updatedGameState);
    }
    
    // 若经过上面处理，阶段已转换为白天，则直接更新状态返回
    if (updatedGameState.phase === 'day') {
      setGameState(updatedGameState);
      return { 
        success: true, 
        state_update: updatedGameState 
      };
    }

    // 获取当前玩家信息与角色
    const currentPlayerId = updatedGameState.current_player;
    const currentPlayer = updatedGameState.players.find(p => p.player_id === currentPlayerId);
    
    if (!currentPlayer) {
      console.error(`无法找到当前玩家(ID: ${currentPlayerId})`);
      return { success: false, error: `无法找到当前玩家(ID: ${currentPlayerId})` };
    }

    const currentRole = currentPlayer.current_role;
    
    // 检查该角色是否具有夜间行动能力
    const nightActionRoles = ['werewolf', 'seer', 'robber', 'troublemaker', 'insomniac'];
    const hasNightAction = nightActionRoles.includes(currentRole);
    
    if (hasNightAction) {
      try {
        // 在发送前规范化游戏状态
        const normalizedGameState = normalizeGameState(gameState);
        
        // 调用API执行夜间行动
        const response = await gameApi.autoNightStep(
          currentPlayerId,
          currentRole,
          normalizedGameState
        );
        
        if (response.success) {
          // 将行动结果记录到历史记录，添加step字段，使用数字时间戳
          updatedGameState.history = [...(gameState.history || []), {
            phase: 'night',
            player_id: currentPlayerId,
            step: gameState.history ? gameState.history.length : 0,
            action: response.action,
            timestamp: Date.now()
          }];
          
          // 根据角色处理行动结果
          processNightActionResult(updatedGameState, currentPlayerId, currentRole, response.action);
        } else {
          console.error("夜间行动执行失败:", response.message);
          return { success: false, error: `夜间行动执行失败: ${response.message}` };
        }
      } catch (error) {
        console.error("执行夜间行动时出错:", error);
        return { success: false, error: `执行夜间行动时出错: ${error.message}` };
      }
    }
    
    // 完成当前玩家的夜间行动后，按照行动顺序寻找下一个可行动的玩家
    moveToNextNightPlayer(updatedGameState);
    
    // 更新游戏状态
    setGameState(updatedGameState);
    
    return { 
      success: true, 
      state_update: updatedGameState 
    };
  };

  // 辅助函数：根据夜间行动顺序确定下一个应行动的玩家
  const moveToNextNightPlayer = (gameState) => {
    // 使用 gameState.action_order 或默认顺序
    const actionOrder = gameState.action_order || ['werewolf', 'seer', 'robber', 'troublemaker', 'insomniac'];
    
    // 如果当前玩家为空，则从第一个行动角色开始查找
    if (gameState.current_player === null || gameState.current_player === undefined) {
      for (let i = 0; i < actionOrder.length; i++) {
        const role = actionOrder[i];
        const playersWithRole = gameState.players.filter(p => p.current_role === role);
        if (playersWithRole.length > 0) {
          gameState.current_player = playersWithRole[0].player_id;
          console.log(`初始设置夜间行动玩家：角色 ${role}, 玩家ID: ${playersWithRole[0].player_id}`);
          return;
        }
      }
      // 如果没有玩家具有夜间行动，结束夜晚阶段进入白天阶段
      console.log("没有找到具有夜间行动的玩家，直接进入白天阶段");
      gameState.phase = 'day';
      gameState.current_player = 0;
      gameState.speech_round = 1;
      return;
    }
    
    // 当当前玩家已设置时，首先检查是否有相同角色的其他玩家需要行动
    const currentPlayer = gameState.players.find(p => p.player_id === gameState.current_player);
    const currentRole = currentPlayer ? currentPlayer.current_role : null;
    
    // 查找相同角色的其他尚未行动的玩家
    if (currentRole) {
      const sameRolePlayers = gameState.players.filter(p => 
        p.current_role === currentRole && 
        p.player_id !== gameState.current_player
      );
      
      // 如果找到同角色的其他玩家，让其行动
      if (sameRolePlayers.length > 0) {
        // 按照玩家ID排序，选择下一个ID
        sameRolePlayers.sort((a, b) => a.player_id - b.player_id);
        
        // 找到当前ID之后的第一个同角色玩家
        const nextPlayer = sameRolePlayers.find(p => p.player_id > gameState.current_player);
        if (nextPlayer) {
          gameState.current_player = nextPlayer.player_id;
          console.log(`同角色下一个行动玩家：角色 ${currentRole}, 玩家ID: ${gameState.current_player}`);
          return;
        }
      }
    }
    
    // 如果没有同角色的其他玩家，则查找下一个角色
    const currentRoleIndex = currentRole ? actionOrder.indexOf(currentRole) : -1;
    
    // 从当前位置之后查找下一个拥有夜间行动能力的角色
    for (let i = currentRoleIndex + 1; i < actionOrder.length; i++) {
      const nextRole = actionOrder[i];
      const playersWithRole = gameState.players.filter(p => p.current_role === nextRole);
      if (playersWithRole.length > 0) {
        // 按玩家ID排序，选择最小ID的玩家
        playersWithRole.sort((a, b) => a.player_id - b.player_id);
        gameState.current_player = playersWithRole[0].player_id;
        console.log(`夜间行动顺序更新：当前角色 ${currentRole} 转为下一个角色 ${nextRole}, 玩家ID: ${playersWithRole[0].player_id}`);
        return;
      } else {
        console.log(`跳过角色 ${nextRole}，无玩家拥有该角色`);
      }
    }
    
    // 如果没有找到下一个具备行动能力的玩家，则结束夜晚阶段，进入白天
    console.log('夜晚阶段结束，进入白天阶段');
    gameState.phase = 'day';
    gameState.current_player = 0; // 重置为白天阶段第一个玩家
    gameState.speech_round = 1;
  };

  // 白天阶段执行逻辑
  const executeDayStep = async () => {
    // 白天阶段处理代码
    // ...
    
    // 返回处理结果
    return { 
      success: true, 
      state_update: { /* 更新后的状态 */ } 
    };
  };

  // 投票阶段执行逻辑
  const executeVoteStep = async () => {
    // 投票阶段处理代码
    // ...
    
    // 返回处理结果
    return { 
      success: true, 
      state_update: { /* 更新后的状态 */ } 
    };
  };

  // 辅助函数：处理夜间行动结果并更新游戏状态
  const processNightActionResult = (gameState, playerId, role, action) => {
    if (!action) return gameState;
    
    const actionName = action.action_name;
    const actionParams = action.action_params || {};
    
    // 复制游戏状态以进行修改
    const updatedGameState = { ...gameState };
    console.log("processNightActionResult", actionName, actionParams);
    
    // 根据不同角色和行动类型处理结果
    switch (role) {
      case 'werewolf':
        if (actionName === 'werewolf_check') {
          updatedGameState.werewolf_info = updatedGameState.werewolf_info || {};
          if ('target' in actionParams) {
            console.log("werewolf_check", actionParams.target);
            let otherWerewolves = updatedGameState.werewolf_info.other_werewolves || [];
            const target = actionParams.target;
            const targetIndex = otherWerewolves.indexOf(target);
            if (targetIndex !== -1) {
              console.log("Debug: updating existing werewolf at index", targetIndex, "with target", target);
              otherWerewolves[targetIndex] = target;
            } else {
              console.log("Debug: target", target, "not found; will add at index", otherWerewolves.length);
            }
            if (!otherWerewolves.includes(actionParams.target)) {
              otherWerewolves.push(actionParams.target);
            }
            updatedGameState.werewolf_info.other_werewolves = otherWerewolves;
          } 
        }
        break;
        
      case 'seer':
        if (actionName === 'seer_check') {
          updatedGameState.seer_info = updatedGameState.seer_info || {};
          if ('target' in actionParams) {
            const targetId = actionParams.target;
            const targetPlayer = gameState.players.find(p => p.player_id === targetId);
            if (targetPlayer) {
              updatedGameState.seer_info.checked_player = {
                player_id: targetId,
                role: targetPlayer.current_role
              };
            }
          } else if ('center_cards' in actionParams) {
            const cardIndices = actionParams.center_cards;
            const checkedCards = cardIndices.map(index => ({
              index,
              role: gameState.center_cards[index]
            }));
            updatedGameState.seer_info.checked_cards = checkedCards;
          }
        }
        break;
        
      case 'robber':
        if (actionName === 'robber_swap') {
          updatedGameState.robber_info = updatedGameState.robber_info || {};
          if ('target' in actionParams) {
            const targetId = actionParams.target;
            const targetPlayer = gameState.players.find(p => p.player_id === targetId);
            const robberPlayer = gameState.players.find(p => p.player_id === playerId);
            if (targetPlayer && robberPlayer) {
              updatedGameState.robber_info.swapped_with = {
                player_id: targetId,
                original_role: targetPlayer.current_role
              };
              const robberRole = robberPlayer.current_role;
              const targetRole = targetPlayer.current_role;
              const robberIndex = gameState.players.findIndex(p => p.player_id === playerId);
              const targetIndex = gameState.players.findIndex(p => p.player_id === targetId);
              if (robberIndex >= 0 && targetIndex >= 0) {
                updatedGameState.players[robberIndex].current_role = targetRole;
                updatedGameState.players[targetIndex].current_role = robberRole;
              }
            }
          }
        }
        break;
        
      case 'troublemaker':
        if (actionName === 'troublemaker_swap') {
          updatedGameState.troublemaker_info = updatedGameState.troublemaker_info || {};
          if ('target1' in actionParams && 'target2' in actionParams) {
            const target1Id = actionParams.target1;
            const target2Id = actionParams.target2;
            const target1Player = gameState.players.find(p => p.player_id === target1Id);
            const target2Player = gameState.players.find(p => p.player_id === target2Id);
            if (target1Player && target2Player) {
              updatedGameState.troublemaker_info.swapped_players = {
                player1: { id: target1Id, original_role: target1Player.current_role },
                player2: { id: target2Id, original_role: target2Player.current_role }
              };
              const role1 = target1Player.current_role;
              const role2 = target2Player.current_role;
              const target1Index = gameState.players.findIndex(p => p.player_id === target1Id);
              const target2Index = gameState.players.findIndex(p => p.player_id === target2Id);
              if (target1Index >= 0 && target2Index >= 0) {
                updatedGameState.players[target1Index].current_role = role2;
                updatedGameState.players[target2Index].current_role = role1;
              }
            }
          }
        }
        break;
        
      case 'insomniac':
        if (actionName === 'insomniac_check') {
          updatedGameState.insomniac_info = updatedGameState.insomniac_info || {};
          const insomniacPlayer = gameState.players.find(p => p.player_id === playerId);
          if (insomniacPlayer) {
            updatedGameState.insomniac_info.final_role = insomniacPlayer.current_role;
          }
        }
        break;
        
      case 'minion':
        if (actionName === 'minion_check') {
          updatedGameState.minion_info = updatedGameState.minion_info || {};
          if ('werewolves' in actionParams) {
            updatedGameState.minion_info.werewolves = actionParams.werewolves;
          }
        }
        break;
      default:
        console.error(`未知角色: ${role}`);
        break;
    }
    return updatedGameState;
  };

  // 获取游戏结果
  const fetchGameResult = async (gameId) => {
    try {
      setLoading(true);
      
      const result = await gameApi.getGameResult(gameId);
      setGameResult(result);
      
      setLoading(false);
      return result;
    } catch (error) {
      console.error('获取游戏结果失败:', error);
      setError('获取游戏结果失败: ' + (error.message || '未知错误'));
      setLoading(false);
      throw error;
    }
  };

  // 发送聊天消息
  const sendChatMessage = (message) => {
    console.log('发送聊天消息:', message);
    
    // 添加消息到本地状态，使用数字时间戳
    setMessages(prev => [...prev, {
      name: player?.name || defaultPlayerName,
      message: message,
      timestamp: Date.now()
    }]);
  };

  // 断开连接
  const disconnect = () => {
    console.log('断开连接');
    
    // 清理本地存储
    localStorage.removeItem('gameId');
    localStorage.removeItem('playerId');
    
    // 重置状态
    setGameState(null);
    setPlayer(null);
    setMessages([]);
    setGameResult(null);
  };

  // 提供上下文值
  const contextValue = {
    gameState,
    setGameState,
    player,
    error,
    loading,
    messages,
    gameResult,
    createGame,
    createTestGame,
    joinGame,
    performAction,
    getAIDecision,
    executeGameStep,
    fetchGameResult,
    sendChatMessage,
    disconnect,
  };

  return (
    <GameContext.Provider value={contextValue}>
      {children}
    </GameContext.Provider>
  );
};

export default GameProvider;