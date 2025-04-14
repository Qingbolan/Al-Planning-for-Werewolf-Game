import axios from 'axios';

// API基础URL
const BASE_API_URL = 'http://localhost:18000'; // 指向后端API服务器

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
      // 确保配置中包含所有必要的字段
      if (!config.num_players) {
        throw new Error('num_players参数缺失');
      }
      
      // 确保配置字段类型正确
      const requestConfig = {
        ...config,
        num_players: parseInt(config.num_players, 10),
        center_card_count: parseInt(config.center_card_count || 3, 10),
        max_speech_rounds: parseInt(config.max_speech_rounds || 3, 10),
      };
      
      // 确保players对象存在且与num_players一致
      if (!requestConfig.players || Object.keys(requestConfig.players).length !== requestConfig.num_players) {
        console.warn(`玩家数量不一致: num_players=${requestConfig.num_players}, 实际players数量=${Object.keys(requestConfig.players || {}).length}`);
      }
      
      // 确保roles列表存在
      if (!requestConfig.roles || !Array.isArray(requestConfig.roles)) {
        console.warn('roles参数缺失或不是数组');
      }
      
      console.log('创建游戏请求:', JSON.stringify(requestConfig, null, 2));
      
      const response = await apiClient.post('/api/game/create', requestConfig);
      console.log('创建游戏响应:', response.data);
      
      // 将游戏ID存储到本地
      if (response.data && response.data.game_id) {
        localStorage.setItem('gameId', response.data.game_id);
      }
      
      return response.data;
    } catch (error) {
      console.error('创建游戏失败:', error);
      console.log('错误详情:', error.response?.data);
      throw error;
    }
  },

  // 加入游戏
  joinGame: async (gameId, playerName) => {
    try {
      console.log(`加入游戏 ${gameId}, 玩家: ${playerName}`);
      
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
        console.log('游戏会话已保存');
      }
      
      return response.data;
    } catch (error) {
      console.error('加入游戏失败:', error);
      throw error;
    }
  },

  // 创建测试游戏
  createTestGame: async (testGameType = "heuristic", numPlayers = 6, seed = null) => {
    try {
      // 构建查询参数
      let url = `/api/game/create-test?test_game_type=${testGameType}&num_players=${numPlayers}`;
      if (seed !== null) {
        url += `&seed=${seed}`;
      }
      
      console.log('创建测试游戏URL:', url);
      const response = await apiClient.get(url);
      
      // 如果成功创建，将游戏ID存储到本地
      if (response.data && response.data.game_id) {
        // 存储游戏信息（用于观察模式）
        localStorage.setItem('gameId', response.data.game_id);
        localStorage.setItem('playerId', 'observer');
      }
      
      // 返回创建的游戏信息
      return {
        gameId: response.data.game_id,
        success: response.data.success,
        message: response.data.message,
        state: response.data.state,
        test_game_type: response.data.test_game_type
      };
    } catch (error) {
      console.error('创建测试游戏失败:', error);
      console.log('错误详情:', error.response?.data?.detail || error.message);
      return {
        error: error.response?.data?.detail || '创建测试游戏时发生未知错误',
        success: false
      };
    }
  },

  // 执行游戏操作 (夜晚行动、白天发言、投票)
  performAction: async (gameId, playerId, action) => {
    try {
      // 确保playerId是整数
      let playerIdToSend = playerId;
      
      // 如果不是observer，则转换为整数
      if (playerId !== 'observer') {
        const playerIdInt = parseInt(playerId, 10);
        if (isNaN(playerIdInt)) {
          throw new Error('玩家ID必须是整数');
        }
        playerIdToSend = playerIdInt;
      }
      
      const response = await apiClient.post(`/api/game/action`, {
        game_id: gameId,
        player_id: playerIdToSend,
        action: action
      });
      
      return response.data;
    } catch (error) {
      console.error('执行游戏操作失败:', error);
      throw error;
    }
  },

  // 获取AI决策
  getAIDecision: async (gameId, playerId, gameState) => {
    try {
      // 确保playerId是整数
      const playerIdInt = parseInt(playerId, 10);
      if (isNaN(playerIdInt)) {
        throw new Error('玩家ID必须是整数');
      }
      
      const response = await apiClient.post(`/api/game/ai-decision`, {
        game_id: gameId,
        player_id: playerIdInt,
        game_state: gameState
      });
      
      return response.data;
    } catch (error) {
      console.error('获取AI决策失败:', error);
      throw error;
    }
  },

  // 执行游戏步骤 (自动化测试/模拟)
  executeGameStep: async (gameId) => {
    try {
      const response = await apiClient.post(`/api/game/step`, {
        game_id: gameId
      });
      
      return response.data;
    } catch (error) {
      console.error('执行游戏步骤失败:', error);
      throw error;
    }
  },

  // 获取游戏结果
  getGameResult: async (gameId) => {
    try {
      const response = await apiClient.get(`/api/game/result/${gameId}`);
      return response.data;
    } catch (error) {
      console.error('获取游戏结果失败:', error);
      throw error;
    }
  }
};

export { gameApi }; 