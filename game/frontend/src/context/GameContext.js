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
            gameId: data.game_id,
            name: humanPlayerName || defaultPlayerName
          });
          
          // 保存会话数据
          localStorage.setItem('gameId', data.game_id);
          localStorage.setItem('playerId', humanPlayerId);
        } else {
          // 如果没有人类玩家，设置为观察者模式
          setPlayer({
            playerId: 'observer',
            gameId: data.game_id,
            name: 'Observer'
          });
          
          // 保存会话数据
          localStorage.setItem('gameId', data.game_id);
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
          gameId: result.game_id,
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
      
      const result = await gameApi.getAIDecision(player.gameId, playerId, gameState);
      
      setLoading(false);
      return result;
    } catch (error) {
      console.error('获取AI决策失败:', error);
      setError('获取AI决策失败: ' + (error.message || '未知错误'));
      setLoading(false);
      throw error;
    }
  };

  // 执行游戏步骤 (自动化测试)
  const executeGameStep = async () => {
    try {
      setLoading(true);
      setError(null);
      
      if (!player || !player.gameId) {
        throw new Error('游戏未创建');
      }
      
      const result = await gameApi.executeGameStep(player.gameId);
      
      // 更新游戏状态
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
      console.error('执行游戏步骤失败:', error);
      setError('执行游戏步骤失败: ' + (error.message || '未知错误'));
      setLoading(false);
      throw error;
    }
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
    
    // 添加消息到本地状态
    setMessages(prev => [...prev, {
      name: player?.name || defaultPlayerName,
      message: message,
      timestamp: new Date().toISOString()
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