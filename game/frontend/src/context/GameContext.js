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
      
      // 在这里可以添加从服务器获取游戏状态的请求
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
      
      const data = await gameApi.createGame(config);
      console.log('Game created successfully:', data);
      
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
  const createTestGame = async (testGameType = "heuristic") => {
    try {
      setLoading(true);
      setError(null);
    
      // 创建测试游戏
      const result = await gameApi.createTestGame(testGameType);
      
      if (result.success) {
        console.log('测试游戏创建成功:', result);
        
        // 设置玩家信息
        setPlayer({
          playerId: result.playerId,
          gameId: result.gameId,
          name: defaultPlayerName
        });
        
        // 在这里你可以从服务器获取初始游戏状态
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
      console.log('Game joined:', result);
      
      if (result.success) {
        // 设置玩家信息
        setPlayer({
          playerId: result.player_id,
          gameId: result.game_id,
          name: playerName || defaultPlayerName
        });
        
        // 在这里你可以从服务器获取初始游戏状态
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
      
      // 通过REST API发送行动
      const result = await gameApi.performAction(player.gameId, player.playerId, action);
      
      // 更新本地游戏状态
      if (result.gameState) {
        setGameState(result.gameState);
      }
      
      return result;
    } catch (error) {
      console.error('执行行动失败:', error);
      setError('执行行动失败: ' + (error.message || '未知错误'));
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
  };

  // 提供上下文值
  const contextValue = {
    gameState,
    setGameState,
    player,
    error,
    loading,
    messages,
    createGame,
    createTestGame,
    joinGame,
    performAction,
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