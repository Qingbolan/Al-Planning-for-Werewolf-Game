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
  const { gameState, executeGameStep, disconnect, loading: contextLoading, error: contextError } = useGame();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Initialize: Check if game state exists in context
  useEffect(() => {
    const checkGameState = async () => {
      try {
        setLoading(true);
        
        // Check if we already have a game state in context
        if (gameState) {
          setLoading(false);
          return;
        }
        
        // Check if we have a saved game ID in localStorage
        const observerGameId = localStorage.getItem('observerGameId');
        if (!observerGameId || observerGameId === "undefined") {
          // No valid game ID found
          localStorage.removeItem('observerGameId');
          setError('Please create a test game first before accessing this page');
        }
      } catch (err) {
        console.error('Failed to initialize game:', err);
        setError(`Game initialization failed: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };
    
    checkGameState();
  }, [gameState]);

  // Return to home: Clear observerGameId from localStorage and navigate to home
  const handleBackToHome = () => {
    localStorage.removeItem('observerGameId');
    disconnect(); // Use disconnect method from GameContext
    navigate('/');
  };

  // Advance the game by one step
  const handleNextAction = async () => {
    try {
      setLoading(true);
      
      console.log('执行下一步操作，当前游戏状态:', {
        phase: gameState?.phase,
        currentPlayerId: gameState?.current_player_id,
        currentPlayerRole: gameState?.players?.find(p => p.player_id === gameState?.current_player_id)?.current_role
      });
      
      const result = await executeGameStep();
      
      if (result && result.success) {
        console.log('动作执行成功，更新后的状态:', {
          newPhase: result.state_update?.phase,
          newCurrentPlayer: result.state_update?.current_player_id
        });
      } else {
        console.error('执行下一步失败:', result?.error);
        setError(result?.error || '未知错误');
      }
    } catch (err) {
      console.error('Failed to execute next action:', err);
      setError('Action execution failed: ' + (err.message || '未知错误'));
    } finally {
      setLoading(false);
    }
  };

  // Advance game phase (multiple steps at once)
  const advanceGamePhase = async () => {
    try {
      // 记录当前游戏阶段
      const currentPhase = gameState.phase;
      
      // 执行多个步骤直到阶段发生变化
      let phaseSwitched = false;
      let maxAttempts = 50; // 防止无限循环
      let attempts = 0;
      
      setLoading(true);
      
      while (!phaseSwitched && attempts < maxAttempts) {
        // 执行一个游戏步骤
        const result = await executeGameStep();
        attempts++;
        
        if (!result || !result.success) {
          console.error('执行游戏步骤失败:', result?.error);
          setError(result?.error || '未知错误');
          break;
        }
        
        // 检查游戏阶段是否已改变
        if (
          result.state_update && 
          result.state_update.phase && 
          result.state_update.phase !== currentPhase
        ) {
          phaseSwitched = true;
          console.log(`游戏阶段已变更: ${currentPhase} -> ${result.state_update.phase}`);
        }
        
        // 如果游戏结束，也停止循环
        if (result.state_update && result.state_update.game_over) {
          phaseSwitched = true;
          console.log('游戏已结束');
        }
      }
      
      if (!phaseSwitched) {
        console.warn(`无法切换游戏阶段，尝试了${attempts}次`);
      }
    } catch (err) {
      console.error('Failed to advance game phase:', err);
      setError('Phase advancement failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Render corresponding phase component based on current game state
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
            Waiting for game state...
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
              Unknown game phase: {gameState.phase}
            </Typography>
            <Button variant="contained" onClick={handleBackToHome}>
              Return to Home
            </Button>
          </Box>
        );
    }
  };

  // Render control panel with better labels and conditions
  const renderGameControls = () => {
    if (!gameState || gameState.game_over) {
      // 游戏结束或未开始时不显示控制面板
      return null;
    }
    
    // 确定当前阶段的按钮文本
    const getNextActionText = () => {
      switch (gameState.phase) {
        case 'night':
          return 'Next Night Action';
        case 'day':
          return 'Next Speech';
        case 'vote':
          return 'Next Vote';
        default:
          return 'Next Step';
      }
    };
    
    // 确定"跳过"按钮的文本
    const getSkipButtonText = () => {
      switch (gameState.phase) {
        case 'night':
          return 'Skip to Day Phase';
        case 'day':
          return 'Skip to Vote Phase';
        case 'vote':
          return 'Skip to Game End';
        default:
          return 'Skip Phase';
      }
    };
    
    // 确定是否显示跳过按钮
    // 夜晚阶段：检查是否所有特殊角色都已执行完毕
    const isNightPhase = gameState.phase === 'night';
    const actionOrder = gameState.action_order || ['werewolf', 'minion', 'seer', 'robber', 'troublemaker', 'insomniac'];
    
    // 获取当前玩家和角色
    const currentPlayerId = gameState.current_player_id;
    const currentPlayer = gameState.players.find(p => p.player_id === currentPlayerId);
    const currentRole = currentPlayer?.current_role;
    
    // 夜晚阶段的跳过按钮条件:
    // 1. 角色不是特殊角色
    // 2. 或者是最后一个特殊角色
    const canSkipNight = !isNightPhase || 
      !currentRole || 
      !actionOrder.includes(currentRole) || 
      actionOrder.indexOf(currentRole) === actionOrder.length - 1;
    
    return (
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
        {/* 下一步操作按钮 */}
        <Button
          variant="contained"
          color="primary"
          onClick={handleNextAction}
          startIcon={<PlayArrowIcon />}
          sx={{ mb: 1 }}
          disabled={loading}
        >
          {getNextActionText()}
        </Button>
        
        {/* 跳过阶段按钮 */}
        {canSkipNight && (
          <Button
            variant="contained"
            color="secondary"
            onClick={advanceGamePhase}
            startIcon={<SkipNextIcon />}
            disabled={loading}
          >
            {getSkipButtonText()}
          </Button>
        )}
        
        {/* 返回主页按钮 */}
        <Button
          variant="outlined"
          color="error"
          onClick={handleBackToHome}
          startIcon={<HomeIcon />}
          sx={{ mt: 1 }}
        >
          Return to Home
        </Button>
      </Box>
    );
  };

  // Show loading state from either local state or context
  if (loading || contextLoading) {
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
          Initializing Game...
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Loading AI test game, please wait...
        </Typography>
      </Container>
    );
  }

  // Show errors from either local state or context
  const displayError = error || contextError;
  if (displayError) {
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
          {displayError}
        </Alert>
        <Button
          variant="contained"
          startIcon={<HomeIcon />}
          onClick={handleBackToHome}
        >
          Return to Home
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
