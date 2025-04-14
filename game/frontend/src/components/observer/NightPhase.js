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

// 夜晚阶段行动描述
const NightAction = ({ gameState }) => {
  if (!gameState || !gameState.history || gameState.history.length === 0) {
    return null;
  }

  // 获取最近的夜晚行动或当前角色
  const latestNightAction = [...gameState.history]
    .filter(action => action.phase === 'night' && action.action && action.action.action_type === 'NIGHT_ACTION')
    .pop();

  // 获取当前应该行动的玩家
  const actionPlayer = latestNightAction 
    ? gameState.players.find(p => p.player_id === latestNightAction.player_id)
    : gameState.players.find(p => p.player_id === gameState.current_player_id);
    
  if (!actionPlayer) {
    return null;
  }

  // 根据角色确定行动描述
  let actionDescription = getActionDescription(actionPlayer.current_role);
  
  // 返回行动UI
  return (
    <Paper sx={{ 
      p: 2, 
      mb: 3, 
      backgroundColor: 'rgba(0,0,0,0.75)', 
      color: '#fff',
      border: '1px solid rgba(100,150,255,0.5)'
    }}>
      <Typography variant="h6" gutterBottom sx={{ color: '#90caf9' }}>
        当前夜晚行动 - 回合 {gameState.turn || 1}
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
        <Avatar src={getRoleImage(actionPlayer.current_role)} sx={{ mr: 2, width: 56, height: 56 }} />
        <Box>
          <Typography variant="h6">
            AI玩家 {actionPlayer.player_id} ({actionPlayer.current_role})
          </Typography>
          <Typography variant="body1" sx={{ color: '#e0e0e0' }}>
            {actionDescription}
          </Typography>
          
          {/* 添加行动顺序提示 */}
          <Box sx={{ mt: 2, p: 1, backgroundColor: 'rgba(25,118,210,0.1)', borderRadius: '4px' }}>
            <Typography variant="caption" sx={{ display: 'block', mb: 0.5, color: '#90caf9' }}>
              夜晚行动顺序:
            </Typography>
            <Typography variant="caption" sx={{ color: '#e0e0e0' }}>
              狼人 → 预言家 → 强盗 → 捣蛋鬼 → 失眠者
            </Typography>
          </Box>
        </Box>
      </Box>
    </Paper>
  );
};

// 获取角色行动描述的辅助函数
const getActionDescription = (role) => {
  switch (role) {
    case 'werewolf':
      return '狼人正在确认队友的身份，并查看一张中央牌';
    case 'seer':
      return '预言家正在查看一名玩家或两张中央牌的身份';
    case 'robber':
      return '强盗正在与一名玩家交换角色，并查看自己的新角色';
    case 'troublemaker':
      return '捣蛋鬼正在交换两名其他玩家的角色';
    case 'insomniac':
      return '失眠者正在确认自己的最终角色';
    case 'minion':
      return '爪牙正在确认狼人的身份';
    default:
      return '玩家正在执行夜晚行动';
  }
};

// 夜晚行动结果展示组件
const NightActionResult = ({ gameState }) => {
  // 获取最近的行动历史
  const latestAction = gameState && gameState.history && gameState.history.length > 0
    ? gameState.history[gameState.history.length - 1]
    : null;
  
  if (!latestAction || latestAction.phase !== 'night') return null;
  
  // 获取执行行动的玩家
  const actionPlayer = gameState.players.find(p => p.player_id === latestAction.player_id);
  if (!actionPlayer) return null;
  
  // 获取行动类型和结果
  const action = latestAction.action;
  if (!action) return null;
  
  // 根据角色和行动获取结果描述
  const getActionResultDescription = () => {
    const role = actionPlayer.current_role;
    const actionName = action.action_name;
    
    // 渲染特定角色行动结果
    switch (role) {
      case 'werewolf':
        if (actionName === 'werewolf_action') {
          // 狼人查看队友
          const otherWerewolves = gameState.werewolf_info?.other_werewolves || [];
          const centerCard = gameState.werewolf_info?.center_card;
          
          return (
            <>
              <Typography variant="subtitle1" sx={{ color: '#ff9800', mt: 1 }}>
                行动结果:
              </Typography>
              <Box sx={{ ml: 2 }}>
                <Typography variant="body2">
                  {otherWerewolves.length > 0 
                    ? `发现其他狼人: ${otherWerewolves.map(id => `AI玩家 ${id}`).join(', ')}` 
                    : '没有发现其他狼人'}
                </Typography>
                {centerCard && (
                  <Typography variant="body2">
                    查看中央牌 {centerCard.index + 1}: {centerCard.role}
                  </Typography>
                )}
              </Box>
            </>
          );
        }
        break;
        
      case 'seer':
        if (actionName === 'seer_action') {
          const targetType = action.action_params.target_type;
          if (targetType === 'player') {
            const targetId = action.action_params.target_id;
            const targetPlayer = gameState.players.find(p => p.player_id === targetId);
            
            return (
              <>
                <Typography variant="subtitle1" sx={{ color: '#2196f3', mt: 1 }}>
                  行动结果:
                </Typography>
                <Box sx={{ ml: 2 }}>
                  <Typography variant="body2">
                    查看 AI玩家 {targetId} 的角色: {targetPlayer?.current_role || '未知'}
                  </Typography>
                </Box>
              </>
            );
          } else {
            const cardIndices = action.action_params.card_indices || [];
            return (
              <>
                <Typography variant="subtitle1" sx={{ color: '#2196f3', mt: 1 }}>
                  行动结果:
                </Typography>
                <Box sx={{ ml: 2 }}>
                  {cardIndices.map((index, i) => (
                    <Typography key={i} variant="body2">
                      查看中央牌 {index + 1}: {gameState.center_cards?.[index] || '未知'}
                    </Typography>
                  ))}
                </Box>
              </>
            );
          }
        }
        break;
        
      // 其他角色行动结果...
        
      default:
        return null;
    }
  };
  
  return (
    <Paper sx={{ 
      p: 2, 
      mb: 3, 
      backgroundColor: 'rgba(0,0,0,0.75)', 
      color: '#fff',
      border: '1px solid rgba(255,200,100,0.5)'
    }}>
      <Typography variant="h6" gutterBottom sx={{ color: '#ffeb3b' }}>
        行动结果
      </Typography>
      
      <Box sx={{ display: 'flex', alignItems: 'flex-start', mt: 1 }}>
        <Avatar src={getRoleImage(actionPlayer.current_role)} sx={{ mr: 2, mt: 1 }} />
        <Box>
          <Typography variant="subtitle1">
            AI玩家 {actionPlayer.player_id} ({actionPlayer.current_role})
          </Typography>
          <Typography variant="body2" sx={{ color: '#e0e0e0' }}>
            执行了: {getActionDescription(action.action_name)}
          </Typography>
          
          {getActionResultDescription()}
        </Box>
      </Box>
    </Paper>
  );
};

const NightPhase = ({ gameState }) => {
  const currentPlayer = gameState.current_player_id !== null 
    ? gameState.players.find(p => p.player_id === gameState.current_player_id)
    : null;

  return (
    <div style={{
      backgroundImage: `url("/first_night.png")`,
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      minHeight: '100vh',
      padding: '20px 0'
    }}>
      <Box sx={{ maxWidth: 'lg', mx: 'auto', px: 2 }}>
        {/* 夜晚阶段标题 */}
        <Card sx={{ mb: 3, bgcolor: 'rgba(0,0,0,0.75)', color: '#fff', p: 2 }}>
          <CardContent>
            <Typography variant="h5" gutterBottom sx={{ color: '#e3f2fd', fontWeight: 'bold', textShadow: '2px 2px 4px rgba(0,0,0,0.7)' }}>
              🌙 夜晚阶段
            </Typography>
            <Typography variant="body1" sx={{ color: '#e3f2fd', textShadow: '1px 1px 2px rgba(0,0,0,0.7)' }}>
              夜晚降临，特殊角色正在执行他们的能力...
            </Typography>
            
            <Box sx={{ mt: 2 }}>
              
              {currentPlayer && (
                <Typography variant="body1" sx={{ mt: 1, display: 'flex', alignItems: 'center' }}>
                  当前行动玩家: 
                  <Chip 
                    avatar={<Avatar src={getRoleImage(currentPlayer.current_role)} />}
                    label={`AI玩家 ${currentPlayer.player_id}`}
                    color="primary"
                    sx={{ ml: 1 }}
                  />
                </Typography>
              )}
            </Box>
          </CardContent>
        </Card>

        {/* 夜晚行动描述 */}
        <NightAction gameState={gameState} />
        
        {/* 玩家列表 */}
        <PlayerList players={gameState.players} currentPlayerId={gameState.current_player_id} />
        
        {/* 中央牌 */}
        <CenterCards centerCards={gameState.center_cards} />
        
        {/* 游戏历史 */}
        <GameHistory history={gameState.history} />

        {/* 行动结果展示 */}
        <NightActionResult gameState={gameState} />
      </Box>
    </div>
  );
};

export default NightPhase; 