import axios from 'axios';

// API基础URL - 添加这个配置
const BASE_API_URL = 'http://localhost:8000'; // 指向后端API服务器

// API客户端
const apiClient = axios.create({
  baseURL: BASE_API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 游戏API
const gameApi = {
  // 创建游戏
  createGame: async (config) => {
    try {
      // 确保配置字段类型正确
      const requestConfig = {
        ...config,
        num_players: parseInt(config.num_players, 10),
        human_player_count: 1, // 固定为1个人类玩家
        max_speech_rounds: parseInt(config.max_speech_rounds || 3, 10),
      };
      console.log('Create game request:', requestConfig);
      
      const response = await apiClient.post('/api/game/create', requestConfig);
      return response.data;
    } catch (error) {
      console.error('Failed to create game:', error);
      throw error;
    }
  },

  // 加入游戏
  joinGame: async (gameId, playerName) => {
    try {
      console.log(`Joining game ${gameId}, player: ${playerName}`);
      
      // 先清除已有的游戏会话数据
      localStorage.removeItem('gameId');
      localStorage.removeItem('playerId');
      
      const response = await apiClient.post(`/api/game/join/${gameId}`, {
        player_name: playerName,
      });
      
      if (response.data && response.data.success) {
        // 保存新的会话数据
        localStorage.setItem('gameId', response.data.game_id);
        localStorage.setItem('playerId', response.data.player_id);
        console.log('Game session saved');
      }
      
      return response.data;
    } catch (error) {
      console.error('Failed to join game:', error);
      throw error;
    }
  },

  // 创建测试游戏
  createTestGame: async (testGameType = "heuristic") => {
    try {
      const response = await apiClient.get(`/api/game/create-test?test_game_type=${testGameType}`);
      
      // 存储游戏信息（仅用于测试）
      localStorage.setItem('gameId', response.data.game_id);
      localStorage.setItem('playerId', response.data.player_id);
      
      return {
        gameId: response.data.game_id,
        playerId: response.data.player_id,
        name: response.data.name,
        success: true
      };
    } catch (error) {
      console.error('Failed to create test game:', error);
      return {
        error: error.response?.data?.detail || '创建测试游戏时发生未知错误',
        success: false
      };
    }
  },

  // 执行游戏操作
  performAction: async (gameId, playerId, action) => {
    try {
      const response = await apiClient.post(`/api/game/action`, {
        game_id: gameId,
        player_id: playerId,
        action: action
      });
      
      return response.data;
    } catch (error) {
      console.error('执行游戏操作失败:', error);
      throw error;
    }
  },
};

export { gameApi }; 