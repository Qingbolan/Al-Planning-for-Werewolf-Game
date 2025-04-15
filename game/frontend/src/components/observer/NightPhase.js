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
  Container,
  Divider
} from '@mui/material';
import PlayerList from './PlayerList';
import CenterCards from './CenterCards';
import GameHistory from './GameHistory';

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

// Helper function to get role action description
const getActionDescription = (role) => {
  switch (role) {
    case 'werewolf':
      return 'The Werewolf is looking for other werewolves and viewing one center card';
    case 'seer':
      return 'The Seer is looking at one player\'s card or two center cards';
    case 'robber':
      return 'The Robber is swapping roles with another player and checking the new role';
    case 'troublemaker':
      return 'The Troublemaker is swapping roles between two other players';
    case 'insomniac':
      return 'The Insomniac is checking their final role';
    case 'minion':
      return 'The Minion is identifying the Werewolves';
    default:
      return 'Player is performing night action';
  }
};

// Improved NightActionResult component with better support for all roles
const NightActionResult = ({ gameState }) => {
  // Get the latest action history
  const latestAction = gameState && gameState.history && gameState.history.length > 0
    ? gameState.history[gameState.history.length - 1]
    : null;
  
  if (!latestAction || latestAction.phase !== 'night') return null;
  
  // Get the player who performed the action
  const actionPlayer = gameState.players.find(p => p.player_id === latestAction.player_id);
  if (!actionPlayer) return null;
  
  // Get action type and result
  const action = latestAction.action;
  if (!action) return null;
  
  // Get action result description based on role and action
  const getActionResultDescription = () => {
    const role = actionPlayer.current_role;
    const actionName = action.action_name;
    
    // Render specific role action result
    switch (role) {
      case 'werewolf':
        if (actionName === 'werewolf_action' || actionName === 'werewolf_check') {
          // Werewolf team check
          const otherWerewolves = gameState.werewolf_info?.other_werewolves || [];
          const centerCard = gameState.werewolf_info?.center_card;
          
          return (
            <>
              <Typography variant="subtitle2" sx={{ color: '#ff9800', mt: 0.5, fontWeight: 'bold' }}>
                Action Result:
              </Typography>
              <Box sx={{ ml: 1 }}>
                <Typography variant="caption" sx={{ display: 'block' }}>
                  {otherWerewolves.length > 0 
                    ? `Found other Werewolves: ${otherWerewolves.map(id => {
                      const player = gameState.players.find(p => p.player_id === id);
                      return player?.name || `AI Player ${id}`;
                    }).join(', ')}` 
                    : 'No other Werewolves found'}
                </Typography>
                {centerCard && (
                  <Typography variant="caption" sx={{ display: 'block' }}>
                    Viewed center card {centerCard.index + 1}: {centerCard.role}
                  </Typography>
                )}
              </Box>
            </>
          );
        }
        break;
        
      case 'seer':
        if (actionName === 'seer_action' || actionName === 'seer_check') {
          // Get seer info
          const seerInfo = gameState.seer_info || {};
          const checkedPlayer = seerInfo.checked_player;
          const checkedCards = seerInfo.checked_cards || [];
          
          return (
            <>
              <Typography variant="subtitle2" sx={{ color: '#2196f3', mt: 0.5, fontWeight: 'bold' }}>
                Action Result:
              </Typography>
              <Box sx={{ ml: 1 }}>
                {checkedPlayer && (
                  <Typography variant="caption" sx={{ display: 'block' }}>
                    Checked player {checkedPlayer.player_id}'s role: {checkedPlayer.role}
                  </Typography>
                )}
                {checkedCards.length > 0 && checkedCards.map((card, i) => (
                  <Typography key={i} variant="caption" sx={{ display: 'block' }}>
                    Checked center card {card.index + 1}: {card.role}
                  </Typography>
                ))}
              </Box>
            </>
          );
        }
        break;
        
      case 'robber':
        if (actionName === 'robber_action' || actionName === 'robber_swap') {
          // Get robber info
          const robberInfo = gameState.robber_info || {};
          const swappedWith = robberInfo.swapped_with;
          
          return (
            <>
              <Typography variant="subtitle2" sx={{ color: '#4caf50', mt: 0.5, fontWeight: 'bold' }}>
                Action Result:
              </Typography>
              <Box sx={{ ml: 1 }}>
                {swappedWith && (
                  <>
                    <Typography variant="caption" sx={{ display: 'block' }}>
                      Swapped with player {swappedWith.player_id}
                    </Typography>
                    <Typography variant="caption" sx={{ display: 'block' }}>
                      New role: {swappedWith.original_role}
                    </Typography>
                  </>
                )}
              </Box>
            </>
          );
        }
        break;
        
      case 'troublemaker':
        if (actionName === 'troublemaker_action' || actionName === 'troublemaker_swap') {
          // Get troublemaker info
          const troublemakerInfo = gameState.troublemaker_info || {};
          const swappedPlayers = troublemakerInfo.swapped_players;
          
          return (
            <>
              <Typography variant="subtitle2" sx={{ color: '#f44336', mt: 0.5, fontWeight: 'bold' }}>
                Action Result:
              </Typography>
              <Box sx={{ ml: 1 }}>
                {swappedPlayers && (
                  <Typography variant="caption" sx={{ display: 'block' }}>
                    Swapped player {swappedPlayers.player1.id} and player {swappedPlayers.player2.id}
                  </Typography>
                )}
              </Box>
            </>
          );
        }
        break;
        
      case 'insomniac':
        if (actionName === 'insomniac_action' || actionName === 'insomniac_check') {
          // Get insomniac info
          const insomniacInfo = gameState.insomniac_info || {};
          const finalRole = insomniacInfo.final_role;
          
          return (
            <>
              <Typography variant="subtitle2" sx={{ color: '#9c27b0', mt: 0.5, fontWeight: 'bold' }}>
                Action Result:
              </Typography>
              <Box sx={{ ml: 1 }}>
                {finalRole && (
                  <Typography variant="caption" sx={{ display: 'block' }}>
                    Final role: {finalRole}
                  </Typography>
                )}
              </Box>
            </>
          );
        }
        break;
        
      case 'minion':
        if (actionName === 'minion_action' || actionName === 'minion_check') {
          // Get minion info
          const minionInfo = gameState.minion_info || {};
          const werewolves = minionInfo.werewolves || [];
          
          return (
            <>
              <Typography variant="subtitle2" sx={{ color: '#795548', mt: 0.5, fontWeight: 'bold' }}>
                Action Result:
              </Typography>
              <Box sx={{ ml: 1 }}>
                <Typography variant="caption" sx={{ display: 'block' }}>
                  {werewolves.length > 0 
                    ? `Identified Werewolves: ${werewolves.map(id => {
                      const player = gameState.players.find(p => p.player_id === id);
                      return player?.name || `AI Player ${id}`;
                    }).join(', ')}` 
                    : 'No Werewolves found'}
                </Typography>
              </Box>
            </>
          );
        }
        break;
        
      default:
        return null;
    }
  };
  
  return (
    <Paper elevation={3} sx={{ 
      p: 1.5, 
      mb: 2, 
      backgroundColor: 'rgba(255,152,0,0.07)', 
      color: '#fff',
      border: '1px solid rgba(255,152,0,0.3)',
      borderRadius: 2
    }}>
      <Typography variant="subtitle1" gutterBottom sx={{ color: '#ffcc80', fontWeight: 'bold', mb: 0.5 }}>
        Action Result
      </Typography>
      
      <Box sx={{ display: 'flex', alignItems: 'flex-start' }}>
        <Avatar src={getRoleImage(actionPlayer.current_role)} sx={{ mr: 1.5, width: 32, height: 32, border: '2px solid rgba(255,152,0,0.5)' }} />
        <Box>
          <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
            {actionPlayer.name || `AI Player ${actionPlayer.player_id}`} ({actionPlayer.current_role})
          </Typography>
          <Typography variant="caption" sx={{ color: '#e0e0e0', display: 'block' }}>
            Performed: {getActionDescription(actionPlayer.current_role)}
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
    
  // Get action description for current player
  const actionDescription = currentPlayer ? getActionDescription(currentPlayer.current_role) : '';

  return (
    <div style={{
      backgroundImage: `linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url("/first_night.png")`,
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      height: '100vh',
      overflow: 'hidden'
    }}>
      <Container maxWidth="xl" sx={{ height: '100vh', pt: 2, pb: 2, overflow: 'hidden' }}>
        <Grid container spacing={2} sx={{ height: '100%' }}>
          {/* Left Column - Main Content (8/12) */}
          <Grid item xs={12} lg={8} sx={{ height: '100%', overflow: 'auto' }}>
            <Grid container spacing={2}>
              {/* Game Phase Header - Full Width */}
              <Grid item xs={12}>
                <Card elevation={4} sx={{ 
                  mb: 2, 
                  bgcolor: 'rgba(0,0,0,0.6)', 
                  color: '#fff', 
                  p: 1.5,
                  border: '1px solid rgba(25,118,210,0.3)',
                  borderRadius: 2
                }}>
                  <CardContent sx={{ py: 1.5, px: 2, "&:last-child": { pb: 1.5 } }}>
                    <Typography variant="h5" gutterBottom sx={{ 
                      color: '#e3f2fd', 
                      fontWeight: 'bold', 
                      textShadow: '2px 2px 4px rgba(0,0,0,0.7)',
                      mb: 0.5
                    }}>
                      🌙 Night Phase
                    </Typography>
                    <Typography variant="body2" sx={{ color: '#e3f2fd', textShadow: '1px 1px 2px rgba(0,0,0,0.7)' }}>
                      Night has fallen, special roles are using their abilities...
                    </Typography>
                    
                    <Box sx={{ mt: 1.5 }}>
                      {currentPlayer && (
                        <>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Typography variant="body2" sx={{ color: '#bdbdbd' }}>
                              Current Acting Player: 
                            </Typography>
                            <Chip 
                              size="small"
                              avatar={<Avatar src={getRoleImage(currentPlayer.current_role)} sx={{ width: 24, height: 24 }} />}
                              label={currentPlayer.name || `AI Player ${currentPlayer.player_id}`}
                              color="primary"
                              sx={{ ml: 1, height: '24px', '& .MuiChip-label': { fontSize: '0.75rem', px: 1 } }}
                            />
                          </Box>
                          <Typography variant="caption" sx={{ color: '#e0e0e0', mt: 0.5, display: 'block' }}>
                            {actionDescription}
                          </Typography>
                        </>
                      )}
                      
                      {/* Action order hint */}
                      <Box sx={{ mt: 1.5, p: 0.75, backgroundColor: 'rgba(25,118,210,0.15)', borderRadius: '6px' }}>
                        <Typography variant="caption" sx={{ display: 'block', color: '#90caf9', fontWeight: 'bold', fontSize: '0.7rem' }}>
                          Night Action Order:
                        </Typography>
                        <Typography variant="caption" sx={{ color: '#e0e0e0', fontSize: '0.7rem' }}>
                          Werewolf → Seer → Robber → Troublemaker → Insomniac
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
              
              {/* Game History - Full Width */}
              <Grid item xs={12}>
                <GameHistory history={gameState.history} />
              </Grid>
              
              {/* Night Action Results - Full Width */}
              <Grid item xs={12}>
                <NightActionResult gameState={gameState} />
              </Grid>
            </Grid>
          </Grid>

          {/* Right Column - Supporting Info (4/12) */}
          <Grid item xs={12} lg={4} sx={{ height: '100%', overflow: 'auto' }}>
            <Box>
              {/* Players Section */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle1" sx={{ mb: 1.5, color: '#fff', borderBottom: '1px solid rgba(255,255,255,0.1)', pb: 0.5 }}>
                  Game Participants
                </Typography>
                <PlayerList players={gameState.players} currentPlayerId={gameState.current_player_id} />
              </Box>
              
              {/* Divider */}
              <Divider sx={{ my: 2, backgroundColor: 'rgba(255,255,255,0.1)' }} />
              
              {/* Center Cards Section */}
              <Box>
                <Typography variant="subtitle1" sx={{ mb: 1.5, color: '#fff', borderBottom: '1px solid rgba(255,255,255,0.1)', pb: 0.5 }}>
                  Center Cards
                </Typography>
                <CenterCards centerCards={gameState.center_cards} />
              </Box>
            </Box>
          </Grid>
        </Grid>
      </Container>
    </div>
  );
};

export default NightPhase; 