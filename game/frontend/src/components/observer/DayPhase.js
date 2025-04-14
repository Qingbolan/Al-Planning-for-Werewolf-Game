import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Avatar,
  Chip,
  Paper
} from '@mui/material';
import PlayerList from './PlayerList';
import CenterCards from './CenterCards';
import GameHistory from './GameHistory';

// 角色图片映射
const roleImages = {
  werewolf: '/images/werewolf.png',
  villager: '/images/villager.png',
  seer: '/images/seer.png',
  robber: '/images/robber.png',
  troublemaker: '/images/troublemaker.png',
  insomniac: '/images/insomniac.png',
  minion: '/images/minion.png',
  default: '/images/anyose.png',
};

// 获取角色图片
const getRoleImage = (role) => {
  return roleImages[role] || roleImages.default;
};

// 白天发言组件
const DaySpeech = ({ gameState }) => {
  if (!gameState || !gameState.history || gameState.history.length === 0) {
    return null;
  }

  // 获取最近的白天发言
  const latestDaySpeech = [...gameState.history]
    .filter(action => action.phase === 'day' && action.action && action.action.action_type === 'DAY_SPEECH')
    .pop();

  if (!latestDaySpeech) {
    return null;
  }

  // 获取发言的玩家
  const speechPlayer = gameState.players.find(p => p.player_id === latestDaySpeech.player_id);
  if (!speechPlayer) {
    return null;
  }

  // 判断发言类型
  let speechType = '';
  let speechContent = '';
  
  if (latestDaySpeech.action.speech_type === 'CLAIM_ROLE') {
    speechType = '角色声明';
    speechContent = `声称自己是 ${latestDaySpeech.action.content?.role_claim || '某个角色'}`;
  } else if (latestDaySpeech.action.speech_type === 'ACCUSE') {
    speechType = '指控';
    const targetPlayer = gameState.players.find(p => p.player_id === latestDaySpeech.action.content?.target_id);
    speechContent = `指控 AI玩家 ${targetPlayer?.player_id || '某人'} 是狼人`;
  } else if (latestDaySpeech.action.speech_type === 'DEFEND') {
    speechType = '辩护';
    const targetPlayer = gameState.players.find(p => p.player_id === latestDaySpeech.action.content?.target_id);
    speechContent = `为 AI玩家 ${targetPlayer?.player_id || '某人'} 辩护`;
  } else {
    speechType = '发言';
    speechContent = '分享了信息';
  }

  return (
    <Paper sx={{ 
      p: 2, 
      mb: 3, 
      backgroundColor: 'rgba(0,0,0,0.75)', 
      color: '#fff',
      border: '1px solid rgba(255,193,7,0.5)'
    }}>
      <Typography variant="h6" gutterBottom sx={{ color: '#ffcc80' }}>
        当前发言 - {speechType}
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'flex-start', mt: 1 }}>
        <Avatar src={getRoleImage(speechPlayer.current_role)} sx={{ mr: 2, mt: 1, width: 56, height: 56 }} />
        <Box sx={{ flexGrow: 1 }}>
          <Typography variant="h6">
            AI玩家 {speechPlayer.player_id} ({speechPlayer.current_role})
          </Typography>
          <Typography variant="body1" sx={{ color: '#e0e0e0' }}>
            {speechContent}
          </Typography>
          
          {latestDaySpeech.action.content?.text && (
            <Card sx={{ mt: 2, p: 1, bgcolor: 'rgba(255,255,255,0.1)', borderRadius: '4px' }}>
              <CardContent sx={{ py: 1 }}>
                <Typography variant="body2" sx={{ fontStyle: 'italic', color: '#ffe0b2' }}>
                  "{latestDaySpeech.action.content.text}"
                </Typography>
              </CardContent>
            </Card>
          )}
        </Box>
      </Box>
    </Paper>
  );
};

const DayPhase = ({ gameState }) => {
  const currentPlayer = gameState.current_player_id !== null 
    ? gameState.players.find(p => p.player_id === gameState.current_player_id)
    : null;

  return (
    <div style={{
      backgroundImage: `url("/day.png")`,
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      minHeight: '100vh',
      padding: '20px 0'
    }}>
      <Box sx={{ maxWidth: 'lg', mx: 'auto', px: 2 }}>
        {/* 白天阶段标题 */}
        <Card sx={{ mb: 3, bgcolor: 'rgba(0,0,0,0.75)', color: '#fff', p: 2 }}>
          <CardContent>
            <Typography variant="h5" gutterBottom sx={{ color: '#fff', fontWeight: 'bold', textShadow: '2px 2px 4px rgba(0,0,0,0.7)' }}>
              ☀️ 白天阶段
            </Typography>
            <Typography variant="body1" sx={{ color: '#fff', textShadow: '1px 1px 2px rgba(0,0,0,0.7)' }}>
              天亮了，玩家们开始交流信息，发言轮次: {gameState.speech_round}/{gameState.max_speech_rounds}
            </Typography>
            
            <Box sx={{ mt: 2 }}>
              
              {currentPlayer && (
                <Typography variant="body1" sx={{ mt: 1, display: 'flex', alignItems: 'center' }}>
                  当前发言玩家: 
                  <Chip 
                    avatar={<Avatar src={getRoleImage(currentPlayer.current_role)} />}
                    label={`AI玩家 ${currentPlayer.player_id}`}
                    color="warning"
                    sx={{ ml: 1 }}
                  />
                </Typography>
              )}
            </Box>
          </CardContent>
        </Card>

        {/* 白天发言描述 */}
        <DaySpeech gameState={gameState} />
        
        {/* 玩家列表 */}
        <PlayerList players={gameState.players} currentPlayerId={gameState.current_player_id} />
        
        {/* 中央牌 */}
        <CenterCards centerCards={gameState.center_cards} />
        
        {/* 游戏历史 */}
        <GameHistory history={gameState.history} />
      </Box>
    </div>
  );
};

export default DayPhase; 