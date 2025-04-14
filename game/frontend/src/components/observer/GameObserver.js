import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Button,
  CircularProgress,
  Alert
} from '@mui/material';
import NightPhase from './NightPhase';
import DayPhase from './DayPhase';
import VotePhase from './VotePhase';
import GameOverPhase from './GameOverPhase';
import { useGame } from '../../context/GameContext';
import {
  PlayArrow as PlayArrowIcon,
  SkipNext as SkipNextIcon,
  Home as HomeIcon
} from '@mui/icons-material';

const GameObserver = () => {
  const navigate = useNavigate();
  const { gameState, setGameState } = useGame();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [gameId, setGameId] = useState(null);

  // 初始化：尝试从 localStorage 中获取测试游戏 ID
  useEffect(() => {
    const initializeGame = async () => {
      try {
        setLoading(true);
        const observerGameId = localStorage.getItem('observerGameId');
        if (observerGameId && observerGameId !== "undefined") {
          console.log('使用已存在的游戏ID:', observerGameId);
          setGameId(observerGameId);
          // 获取初始游戏状态
          await fetchInitialGameState(observerGameId);
        } else {
          // 无测试游戏 ID 或ID无效时给出提示
          localStorage.removeItem('observerGameId');
          setError('请先创建一个测试游戏后再访问此页面');
        }
      } catch (err) {
        console.error('初始化游戏失败:', err);
        setError(`初始化游戏失败: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };
    initializeGame();
  }, []);

  // 获取初始游戏状态 - 使用正确的API或本地数据
  const fetchInitialGameState = async (id) => {
    if (!id || id === "undefined") return;
    
    // 这里使用模拟数据，完全避免调用 /api/game/state/ 端点
    // 实际项目中应该使用正确的API端点或状态管理方案
    const mockGameState = {
      game_id: id,
      phase: 'night',
      turn: 1,
      current_player_id: 0,
      players: [
        { player_id: 0, original_role: 'werewolf', current_role: 'werewolf', team: 'werewolf' },
        { player_id: 1, original_role: 'seer', current_role: 'seer', team: 'villager' },
        { player_id: 2, original_role: 'villager', current_role: 'villager', team: 'villager' },
      ],
      center_cards: ['robber', 'troublemaker', 'villager'],
      history: [],
      game_over: false
    };
    
    setGameState(mockGameState);
  };

  // 模拟游戏状态更新 - 完全避免调用 /api/game/state/ 端点
  useEffect(() => {
    if (!gameId || gameId === "undefined") return;
    
    // 使用本地定时器模拟游戏状态变化，而不是通过API轮询
    const interval = setInterval(() => {
      // 此处可以执行本地状态更新逻辑，或者如果有其他正确的API端点可以使用
      // 目前只是保持当前状态
    }, 3000);
    
    return () => clearInterval(interval);
  }, [gameId]);

  // 返回首页：清除 localStorage 中的 observerGameId 并导航回首页
  const handleBackToHome = () => {
    localStorage.removeItem('observerGameId');
    navigate('/');
  };

  // 获取下一个夜晚角色行动 - 使用模拟数据或正确的API
  const getNextNightAction = async () => {
    if (!gameId || gameId === "undefined") return null;
    
    // 返回模拟数据
    return {
      action: {
        action_type: 'night_action',
        action_name: 'werewolf_action',
        player_id: 0
      }
    };
  };

  // 推进到下一游戏阶段 - 使用本地状态更新
  const advanceGamePhase = async () => {
    if (!gameId || gameId === "undefined" || !gameState) return null;
    
    // 本地更新游戏阶段
    const newState = { ...gameState };
    
    // 简单的阶段循环：night -> day -> vote -> game_over
    if (newState.phase === 'night') {
      newState.phase = 'day';
    } else if (newState.phase === 'day') {
      newState.phase = 'vote';
    } else if (newState.phase === 'vote') {
      newState.phase = 'game_over';
      newState.game_over = true;
      newState.winner = 'villager'; // 模拟游戏结果
    }
    
    setGameState(newState);
    return newState;
  };

  // 处理"下一个行动"逻辑
  const handleNextAction = async () => {
    try {
      // 判断当前阶段为夜晚则逐步执行角色行动，否则直接推进阶段
      if (gameState && gameState.phase === 'night') {
        const decision = await getNextNightAction();
        // 如果返回的决策为空或没有 action 字段，则说明本阶段所有角色都已执行完毕，推进阶段
        if (!decision || !decision.action) {
          await advanceGamePhase();
        } else {
          // 模拟执行决策后的状态更新
          const newState = { ...gameState };
          if (!newState.history) newState.history = [];
          
          newState.history.push({
            phase: 'night',
            player_id: decision.action.player_id,
            action: decision.action,
            turn: newState.history.length + 1
          });
          
          // 更新当前玩家为下一个玩家
          newState.current_player_id = (newState.current_player_id + 1) % newState.players.length;
          
          setGameState(newState);
        }
      } else {
        await advanceGamePhase();
      }
    } catch (err) {
      console.error('执行下一个行动失败:', err);
      setError('执行行动失败: ' + err.message);
    }
  };

  // 根据当前游戏状态渲染对应阶段组件
  const renderPhaseComponent = () => {
    if (!gameState) {
      return (
        <Container
          sx={{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            height: '100vh'
          }}
        >
          <Typography variant="h5" sx={{ mb: 2 }}>
            正在等待游戏状态...
          </Typography>
        </Container>
      );
    }

    if (gameState.game_over) {
      return (
        <GameOverPhase gameState={gameState} onBackToHome={handleBackToHome} />
      );
    }

    switch (gameState.phase) {
      case 'night':
        return <NightPhase gameState={gameState} />;
      case 'day':
        return <DayPhase gameState={gameState} />;
      case 'vote':
        return <VotePhase gameState={gameState} />;
      default:
        return (
          <Box
            sx={{
              p: 3,
              textAlign: 'center',
              color: 'white',
              backgroundImage: `url("/cover.png")`,
              backgroundSize: 'cover',
              backgroundPosition: 'center',
              minHeight: '100vh',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center'
            }}
          >
            <Typography variant="h4" sx={{ mb: 2 }}>
              未知游戏阶段: {gameState.phase}
            </Typography>
            <Button variant="contained" onClick={handleBackToHome}>
              返回首页
            </Button>
          </Box>
        );
    }
  };

  // 渲染控制面板（固定在页面右下角）
  const renderGameControls = () => (
    <Box
      sx={{
        position: 'fixed',
        bottom: 20,
        right: 20,
        zIndex: 1000,
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
        backgroundColor: 'rgba(0,0,0,0.7)',
        padding: 2,
        borderRadius: 2
      }}
    >
      <Typography variant="h6" sx={{ color: '#fff', mb: 1 }}>
        演示控制
      </Typography>
      <Button
        variant="contained"
        color="primary"
        onClick={handleNextAction}
        startIcon={<PlayArrowIcon />}
        sx={{ mb: 1 }}
      >
        下一个行动
      </Button>
      <Button
        variant="contained"
        color="secondary"
        onClick={advanceGamePhase}
        startIcon={<SkipNextIcon />}
      >
        跳至下一阶段
      </Button>
      <Button
        variant="outlined"
        color="error"
        onClick={handleBackToHome}
        startIcon={<HomeIcon />}
        sx={{ mt: 1 }}
      >
        返回首页
      </Button>
    </Box>
  );

  // 根据加载和错误状态渲染页面
  if (loading) {
    return (
      <Container
        sx={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100vh'
        }}
      >
        <CircularProgress size={60} sx={{ mb: 3 }} />
        <Typography variant="h5" sx={{ mb: 2 }}>
          正在初始化游戏...
        </Typography>
        <Typography variant="body1" color="text.secondary">
          正在加载 AI 测试游戏，请稍候...
        </Typography>
      </Container>
    );
  }

  if (error) {
    return (
      <Container
        sx={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100vh'
        }}
      >
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
        <Button
          variant="contained"
          startIcon={<HomeIcon />}
          onClick={handleBackToHome}
        >
          返回首页
        </Button>
      </Container>
    );
  }

  return (
    <>
      {renderPhaseComponent()}
      {renderGameControls()}
    </>
  );
};

export default GameObserver;
