import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Avatar,
  Chip,
  Paper,
  Grid,
  Button
} from '@mui/material';
import CelebrationIcon from '@mui/icons-material/Celebration';
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

// 游戏结果组件
const GameResults = ({ gameState }) => {
  // 计算每位玩家获得的票数
  const voteCount = {};
  gameState.players.forEach(player => {
    voteCount[player.player_id] = 0;
  });

  // 统计票数
  gameState.history
    .filter(action => action.action && action.action.action_type === 'VOTE')
    .forEach(action => {
      const targetId = action.action.target_id;
      if (targetId !== undefined && voteCount[targetId] !== undefined) {
        voteCount[targetId]++;
      }
    });

  // 找出被票出的玩家
  const mostVotes = Math.max(...Object.values(voteCount));
  const eliminated = Object.keys(voteCount).filter(id => voteCount[id] === mostVotes);

  // 获取狼人和村民玩家
  const werewolves = gameState.players.filter(p => p.team === 'werewolf');
  const villagers = gameState.players.filter(p => p.team === 'villager');

  return (
    <Paper sx={{ 
      p: 2, 
      mb: 3, 
      backgroundColor: 'rgba(0,0,0,0.75)', 
      color: '#fff',
      border: gameState.winner === 'werewolf' ? '2px solid #f44336' : '2px solid #4caf50'
    }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5" sx={{ 
          color: gameState.winner === 'werewolf' ? '#f44336' : '#4caf50',
          fontWeight: 'bold',
          display: 'flex',
          alignItems: 'center'
        }}>
          <CelebrationIcon sx={{ mr: 1 }} />
          {gameState.winner === 'werewolf' ? '狼人阵营获胜!' : '村民阵营获胜!'}
        </Typography>
        <Chip 
          label="游戏结束"
          color="warning"
          variant="outlined"
        />
      </Box>
      
      {/* 投票结果 */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ color: '#ffcc80' }}>
          最终投票结果
        </Typography>
        
        {eliminated.length > 0 && mostVotes > 0 && (
          <Box sx={{ mb: 2, p: 1, backgroundColor: 'rgba(244,67,54,0.1)', borderRadius: '4px' }}>
            <Typography variant="body1" sx={{ color: '#ff8a80' }}>
              {eliminated.length === 1 
                ? `AI玩家 ${eliminated[0]} 被投票出局（${mostVotes}票）` 
                : `多名玩家平票：${eliminated.map(id => `AI玩家 ${id}`).join('，')}（各${mostVotes}票）`}
            </Typography>
          </Box>
        )}
      </Box>
      
      {/* 阵营信息 */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Card sx={{ 
            bgcolor: 'rgba(244,67,54,0.2)', 
            p: 2, 
            borderRadius: '8px', 
            border: gameState.winner === 'werewolf' ? '1px solid #f44336' : 'none'
          }}>
            <Typography variant="h6" gutterBottom sx={{ color: '#f44336' }}>
              🐺 狼人阵营
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {werewolves.map((wolf) => (
                <Chip 
                  key={wolf.player_id}
                  avatar={<Avatar src={getRoleImage(wolf.current_role)} />}
                  label={`AI玩家 ${wolf.player_id} (${wolf.current_role})`}
                  sx={{ mb: 1 }}
                  color="error"
                />
              ))}
            </Box>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card sx={{ 
            bgcolor: 'rgba(76,175,80,0.2)', 
            p: 2, 
            borderRadius: '8px',
            border: gameState.winner === 'villager' ? '1px solid #4caf50' : 'none'
          }}>
            <Typography variant="h6" gutterBottom sx={{ color: '#4caf50' }}>
              🏡 村民阵营
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {villagers.map((villager) => (
                <Chip 
                  key={villager.player_id}
                  avatar={<Avatar src={getRoleImage(villager.current_role)} />}
                  label={`AI玩家 ${villager.player_id} (${villager.current_role})`}
                  sx={{ mb: 1 }}
                  color="success"
                />
              ))}
            </Box>
          </Card>
        </Grid>
      </Grid>
      
      {/* 游戏统计 */}
      <Box>
        <Typography variant="h6" gutterBottom>
          游戏统计
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={6} sm={3}>
            <Card sx={{ bgcolor: 'rgba(30,30,40,0.8)', p: 1, textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">回合数</Typography>
              <Typography variant="h6">{Math.max(...gameState.history.map(h => h.turn), 0) + 1}</Typography>
            </Card>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Card sx={{ bgcolor: 'rgba(30,30,40,0.8)', p: 1, textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">发言轮次</Typography>
              <Typography variant="h6">{gameState.max_speech_rounds}</Typography>
            </Card>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Card sx={{ bgcolor: 'rgba(30,30,40,0.8)', p: 1, textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">玩家数</Typography>
              <Typography variant="h6">{gameState.players.length}</Typography>
            </Card>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Card sx={{ bgcolor: 'rgba(30,30,40,0.8)', p: 1, textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">投票总数</Typography>
              <Typography variant="h6">
                {gameState.history.filter(action => action.action && action.action.action_type === 'VOTE').length}
              </Typography>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </Paper>
  );
};

const GameOverPhase = ({ gameState, onBackToHome }) => {
  return (
    <div style={{
      backgroundImage: `url("/cover.png")`,
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      minHeight: '100vh',
      padding: '20px 0'
    }}>
      <Box sx={{ maxWidth: 'lg', mx: 'auto', px: 2 }}>
        {/* 游戏结束标题 */}
        <Card sx={{ mb: 3, bgcolor: 'rgba(0,0,0,0.75)', color: '#fff', p: 2 }}>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <Typography variant="h5" gutterBottom sx={{ color: '#fff', fontWeight: 'bold', textShadow: '2px 2px 4px rgba(0,0,0,0.7)' }}>
                  🏁 游戏结束
                </Typography>
                <Typography variant="body1" sx={{ color: '#ffeb3b', textShadow: '1px 1px 2px rgba(0,0,0,0.7)' }}>
                  {gameState.winner === 'werewolf' ? '🐺 狼人阵营获胜!' : 
                  gameState.winner === 'villager' ? '🏡 村民阵营获胜!' : 
                  '游戏结束，没有获胜者'}
                </Typography>
              </div>
              <Button 
                variant="contained" 
                color="primary" 
                onClick={onBackToHome}
                sx={{ height: 'fit-content' }}
              >
                返回首页
              </Button>
            </Box>
            
            <Box sx={{ mt: 2 }}>
            </Box>
          </CardContent>
        </Card>

        {/* 游戏结果 */}
        <GameResults gameState={gameState} />
        
        {/* 玩家列表 */}
        <PlayerList players={gameState.players} currentPlayerId={null} />
        
        {/* 中央牌 */}
        <CenterCards centerCards={gameState.center_cards} />
        
        {/* 游戏历史 */}
        <GameHistory history={gameState.history} />
      </Box>
    </div>
  );
};

export default GameOverPhase; 