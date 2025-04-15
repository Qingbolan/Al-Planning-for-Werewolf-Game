import React from 'react';
import {
  Grid,
  Typography,
  Avatar,
  Chip,
  Box,
  Card,
  Tooltip
} from '@mui/material';
import { useGame } from '../../context/GameContext';

// Role images mapping
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

// Get role image
const getRoleImage = (role) => {
  return roleImages[role] || roleImages.default;
};

// Role name formatting
const formatRoleName = (role) => {
  return role.charAt(0).toUpperCase() + role.slice(1);
};

const PlayerList = ({ players: propsPlayers, currentPlayerId }) => {
  // 从GameContext获取完整游戏状态
  const { gameState } = useGame();
  
  // 优先使用GameContext中的玩家数据，如果没有则使用props传入的玩家数据
  const players = gameState && gameState.players && gameState.players.length > 0 
    ? gameState.players 
    : propsPlayers;
    
  // 如果GameContext中有当前玩家ID，优先使用，否则使用props传入的
  const activePlayerId = gameState && gameState.current_player_id !== undefined
    ? gameState.current_player_id
    : currentPlayerId;
  
  if (!players || players.length === 0) return null;
  
  return (
    <Grid container spacing={1.5}>
      {players.map((player, index) => {
        const isCurrentPlayer = activePlayerId === player.player_id;
        return (
          <Grid item xs={12} key={index}>
            <Card
              elevation={isCurrentPlayer ? 3 : 1}
              sx={{
                border: isCurrentPlayer
                  ? '1px solid #f50057'
                  : '1px solid rgba(255,255,255,0.12)',
                backgroundColor: isCurrentPlayer 
                  ? 'rgba(245,0,87,0.08)'
                  : 'rgba(30,30,40,0.7)',
                color: '#fff',
                display: 'flex',
                borderRadius: 2,
                transition: 'all 0.2s ease',
                '&:hover': {
                  backgroundColor: isCurrentPlayer 
                    ? 'rgba(245,0,87,0.12)'
                    : 'rgba(30,30,40,0.9)',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.2)'
                }
              }}
            >
              <Box sx={{ 
                width: '60px', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                p: 1,
                bgcolor: isCurrentPlayer ? 'rgba(245,0,87,0.15)' : 'rgba(0,0,0,0.2)',
                borderRight: '1px solid rgba(255,255,255,0.05)'
              }}>
                <Avatar
                  src={getRoleImage(player.current_role)}
                  sx={{ 
                    width: 45, 
                    height: 45, 
                    border: isCurrentPlayer 
                      ? '2px solid rgba(245,0,87,0.5)' 
                      : '2px solid rgba(255,255,255,0.1)'
                  }}
                />
              </Box>
              
              <Box sx={{ 
                display: 'flex', 
                flexDirection: 'column', 
                justifyContent: 'center',
                flexGrow: 1,
                px: 2,
                py: 1
              }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Typography variant="subtitle2" fontWeight="bold" noWrap>
                    {player.name || `AI Player ${player.player_id}`}
                    {isCurrentPlayer && (
                      <Typography 
                        component="span" 
                        sx={{ 
                          ml: 1, 
                          color: '#f50057', 
                          fontSize: '0.75rem', 
                          fontWeight: 'bold' 
                        }}
                      >
                        (ACTIVE)
                      </Typography>
                    )}
                  </Typography>
                  <Tooltip title={`Team: ${player.team === 'werewolf' ? 'Werewolves' : 'Villagers'}`}>
                    <Chip
                      size="small"
                      label={player.team || 'Unknown'}
                      color={player.team === 'werewolf' ? 'error' : 'success'}
                      sx={{ 
                        height: '20px',
                        '& .MuiChip-label': { 
                          px: 1, 
                          fontSize: '0.7rem',
                          fontWeight: 'bold'
                        }
                      }}
                    />
                  </Tooltip>
                </Box>
                
                {player.current_role && (
                  <Tooltip title="Current role (may have changed during the night)">
                    <Typography variant="caption" sx={{ 
                      mt: 0.5, 
                      color: '#bdbdbd',
                      display: 'flex',
                      alignItems: 'center'
                    }}>
                      <Box component="span" sx={{ 
                        width: '8px', 
                        height: '8px', 
                        borderRadius: '50%',
                        bgcolor: player.team === 'werewolf' ? '#f44336' : '#4caf50',
                        mr: 1
                      }} />
                      Role: {formatRoleName(player.current_role)}
                    </Typography>
                  </Tooltip>
                )}
              </Box>
            </Card>
          </Grid>
        );
      })}
    </Grid>
  );
};

export default PlayerList; 