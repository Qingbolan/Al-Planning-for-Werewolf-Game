import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Avatar,
  Chip,
  Box,
  Card,
  CardContent
} from '@mui/material';

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

const PlayerList = ({ players, currentPlayerId }) => {
  return (
    <Paper sx={{ p: 2, mb: 3, backgroundColor: 'rgba(0,0,0,0.75)', color: '#fff' }}>
      <Typography variant="h6" gutterBottom>
        AI玩家 ({players.length})
      </Typography>
      <Grid container spacing={1}>
        {players.map((player, index) => (
          <Grid item xs={4} sm={3} md={2} key={index}>
            <Card
              sx={{
                border: currentPlayerId === player.player_id
                  ? '2px solid #f50057'
                  : 'none',
                backgroundColor: 'rgba(30,30,40,0.9)',
                color: '#fff',
                minHeight: '140px',
                display: 'flex',
                flexDirection: 'column',
                p: 1
              }}
            >
              <CardContent sx={{ p: 1, "&:last-child": { pb: 1 } }}>
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <Avatar
                    src={getRoleImage(player.current_role)}
                    sx={{ width: 40, height: 40, mb: 0.5 }}
                  />
                  <Typography variant="body2" noWrap>
                    AI玩家 {player.player_id}
                  </Typography>
                  <Chip
                    size="small"
                    label={player.team || '未知阵营'}
                    color={player.team === 'werewolf' ? 'error' : 'success'}
                    sx={{ mt: 0.5, fontSize: '0.7rem' }}
                  />
                  {player.current_role && (
                    <Typography variant="caption" sx={{ mt: 0.5 }}>
                      角色: {player.current_role}
                    </Typography>
                  )}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Paper>
  );
};

export default PlayerList; 