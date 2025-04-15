import React from 'react';
import {
  Paper,
  Typography,
  List,
  ListItem,
  Divider,
  Box,
  Avatar,
  Chip,
  IconButton,
  Tooltip
} from '@mui/material';
import { 
  NightsStay as NightIcon,
  WbSunny as DayIcon,
  HowToVote as VoteIcon,
  Info as InfoIcon
} from '@mui/icons-material';
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

// Helper function to get action description
const getActionTypeLabel = (actionType) => {
  switch(actionType) {
    case 'NIGHT_ACTION':
      return 'Night Ability';
    case 'DAY_SPEECH':
      return 'Speech';
    case 'VOTE':
      return 'Vote';
    default:
      return 'Action';
  }
};

// Helper function to format role name with first letter capitalized
const formatRoleName = (role) => {
  if (!role) return 'Unknown';
  return role.charAt(0).toUpperCase() + role.slice(1);
};

// Helper function to get detailed action description based on role and action
const getDetailedActionDescription = (role, actionName, targetId) => {
  switch(role) {
    case 'werewolf':
      return 'checked for other werewolves and viewed a center card';
    case 'seer':
      return targetId !== undefined 
        ? `checked AI Player ${targetId}'s role`
        : 'viewed two center cards';
    case 'robber':
      return targetId !== undefined
        ? `swapped roles with AI Player ${targetId}`
        : 'attempted to swap roles';
    case 'troublemaker':
      return 'swapped roles between two other players';
    case 'insomniac':
      return 'checked their final role';
    case 'minion':
      return 'identified the werewolves';
    case 'villager':
      return 'acknowledged their role';
    default:
      return 'performed night ability';
  }
};

const GameHistory = ({ history }) => {
  // 从GameContext获取完整游戏状态
  const { gameState } = useGame();
  
  // 创建历史记录的倒序副本
  const reversedHistory = history && history.length > 0 ? [...history].reverse() : [];
  
  // 获取玩家角色的方法
  const getPlayerRole = (playerId) => {
    // 首先尝试从gameState中找到玩家的角色
    if (gameState && gameState.players && Array.isArray(gameState.players)) {
      const player = gameState.players.find(p => p.player_id === playerId);
      if (player) {
        return player.current_role || player.original_role;
      }
    }
    
    // 如果gameState没有玩家信息，尝试直接从history中找
    for (const entry of history) {
      if (entry.player_id === playerId && entry.role) {
        return entry.role;
      }
      // 有些历史记录可能在action内包含角色信息
      if (entry.player_id === playerId && entry.action && entry.action.role) {
        return entry.action.role;
      }
    }
    
    return null; // 如果找不到角色
  };
  
  return (
    <Paper 
      elevation={4}
      sx={{ 
        backgroundColor: 'rgba(0,0,0,0.6)', 
        color: '#fff',
        border: '1px solid rgba(100,150,255,0.3)',
        borderRadius: 2,
        mb: 2,
        overflow: 'hidden'
      }} 
    >
      <Box sx={{ 
        p: 1, 
        backgroundColor: 'rgba(25,118,210,0.2)', 
        borderBottom: '1px solid rgba(100,150,255,0.3)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <Typography variant="h6" sx={{ color: '#90caf9', fontWeight: 'bold' }}>
          Operator History
        </Typography>
        <Tooltip title="Chronological record of all game actions">
          <IconButton size="small" sx={{ color: 'rgba(255,255,255,0.6)' }}>
            <InfoIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
      
      <Box sx={{ maxHeight: '250px', overflow: 'auto', p: 1 }}>
        <List disablePadding dense>
          {reversedHistory.length > 0 ? (
            reversedHistory.map((item, index) => {
              // 获取该历史记录关联的玩家角色
              const playerRole = item.player_id !== undefined ? getPlayerRole(item.player_id) : null;
              
              // 获取目标玩家ID（如果有）
              const targetId = item.action?.target_id;
              
              return (
                <React.Fragment key={index}>
                  <ListItem 
                    sx={{ 
                      backgroundColor: index % 2 === 0 ? 'rgba(25,118,210,0.05)' : 'transparent',
                      borderRadius: '6px',
                      mb: 0.75,
                      p: 1,
                      border: '1px solid rgba(255,255,255,0.05)',
                      transition: 'all 0.2s ease',
                      '&:hover': {
                        backgroundColor: 'rgba(25,118,210,0.1)',
                      }
                    }}
                  >
                    <Box sx={{ mr: 1 }}>
                      {item.phase === 'night' ? (
                        <Avatar sx={{ bgcolor: 'rgba(25,118,210,0.8)', width: 32, height: 32 }}>
                          <NightIcon fontSize="small" />
                        </Avatar>
                      ) : item.phase === 'day' ? (
                        <Avatar sx={{ bgcolor: 'rgba(255,152,0,0.8)', width: 32, height: 32 }}>
                          <DayIcon fontSize="small" />
                        </Avatar>
                      ) : item.phase === 'vote' ? (
                        <Avatar sx={{ bgcolor: 'rgba(156,39,176,0.8)', width: 32, height: 32 }}>
                          <VoteIcon fontSize="small" />
                        </Avatar>
                      ) : (
                        <Avatar sx={{ bgcolor: 'rgba(100,100,100,0.8)', width: 32, height: 32 }}>
                          <InfoIcon fontSize="small" />
                        </Avatar>
                      )}
                    </Box>
                    <Box sx={{ flexGrow: 1 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.3 }}>
                        <Typography variant="body2" sx={{ color: '#fff', fontWeight: 'bold' }}>
                          {item.phase === 'night' ? 'Night Phase' : 
                          item.phase === 'day' ? 'Day Phase' : 
                          item.phase === 'vote' ? 'Vote Phase' : 
                          item.phase} 
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          {item.action && (
                            <Chip
                              size="small"
                              label={getActionTypeLabel(item.action.action_type)}
                              sx={{ 
                                bgcolor: 
                                  item.action.action_type === 'NIGHT_ACTION' ? 'rgba(25,118,210,0.2)' : 
                                  item.action.action_type === 'DAY_SPEECH' ? 'rgba(255,152,0,0.2)' : 
                                  item.action.action_type === 'VOTE' ? 'rgba(156,39,176,0.2)' : 
                                  'rgba(100,100,100,0.2)',
                                color: '#fff',
                                fontSize: '0.65rem',
                                fontWeight: 'bold',
                                height: '18px',
                                '& .MuiChip-label': { px: 0.75 }
                              }}
                            />
                          )}
                          <Chip
                            size="small"
                            label={`Turn ${item.turn}`}
                            sx={{ 
                              bgcolor: 'rgba(255,255,255,0.1)', 
                              color: '#fff',
                              fontSize: '0.65rem',
                              height: '18px',
                              '& .MuiChip-label': { px: 0.75 }
                            }}
                          />
                        </Box>
                      </Box>
                      
                      <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                        {playerRole && (
                          <Avatar 
                            src={getRoleImage(playerRole)} 
                            sx={{ 
                              width: 20, 
                              height: 20, 
                              mr: 1,
                              border: '1px solid rgba(255,255,255,0.2)'
                            }}
                          />
                        )}
                        <Typography variant="caption" sx={{ color: '#e0e0e0', display: 'block' }}>
                          {item.action
                            ? `${formatRoleName(playerRole || 'unknown')} ${
                                item.action.action_type === 'NIGHT_ACTION' 
                                  ? getDetailedActionDescription(playerRole, item.action.action_name, targetId)
                                  : item.action.action_type === 'DAY_SPEECH' 
                                  ? 'made a speech' 
                                  : item.action.action_type === 'VOTE'
                                  ? 'voted for AI Player ' + item.action.target_id
                                  : 'performed an action'}`
                            : 'System event'
                          }
                        </Typography>
                      </Box>
                      
                      {/* Optional details that could be shown */}
                      {item.action && item.action.action_type === 'DAY_SPEECH' && item.action.content?.text && (
                        <Box sx={{ mt: 0.5, p: 0.5, bgcolor: 'rgba(255,255,255,0.05)', borderRadius: '4px', fontStyle: 'italic' }}>
                          <Typography variant="caption" sx={{ color: '#bdbdbd', fontSize: '0.7rem' }}>
                            "{item.action.content.text.substring(0, 60)}{item.action.content.text.length > 60 ? '...' : ''}"
                          </Typography>
                        </Box>
                      )}
                    </Box>
                  </ListItem>
                  {index < reversedHistory.length - 1 && <Divider sx={{ backgroundColor: 'rgba(255,255,255,0.05)', my: 0.5 }} />}
                </React.Fragment>
              );
            })
          ) : (
            <Box sx={{ 
              p: 2, 
              display: 'flex', 
              flexDirection: 'column', 
              alignItems: 'center', 
              justifyContent: 'center',
              color: '#9e9e9e',
              textAlign: 'center'
            }}>
              <InfoIcon sx={{ fontSize: '2rem', opacity: 0.5, mb: 1 }} />
              <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
                Game just started
              </Typography>
              <Typography variant="caption">
                History records will appear here as actions are performed
              </Typography>
            </Box>
          )}
        </List>
      </Box>
    </Paper>
  );
};

export default GameHistory; 