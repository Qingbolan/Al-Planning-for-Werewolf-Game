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

// Day speech component
const DaySpeech = ({ gameState }) => {
  if (!gameState || !gameState.history || gameState.history.length === 0) {
    return null;
  }

  // Get latest day speech
  const latestDaySpeech = [...gameState.history]
    .filter(action => action.phase === 'day' && action.action && action.action.action_type === 'DAY_SPEECH')
    .pop();

  if (!latestDaySpeech) {
    return null;
  }

  // Get the speaking player
  const speechPlayer = gameState.players.find(p => p.player_id === latestDaySpeech.player_id);
  if (!speechPlayer) {
    return null;
  }

  // Determine speech type
  let speechType = '';
  let speechContent = '';
  
  if (latestDaySpeech.action.speech_type === 'CLAIM_ROLE') {
    speechType = 'Role Claim';
    speechContent = `Claims to be a ${latestDaySpeech.action.content?.role_claim || 'specific role'}`;
  } else if (latestDaySpeech.action.speech_type === 'ACCUSE') {
    speechType = 'Accusation';
    const targetPlayer = gameState.players.find(p => p.player_id === latestDaySpeech.action.content?.target_id);
    speechContent = `Accuses AI Player ${targetPlayer?.player_id || 'someone'} of being a Werewolf`;
  } else if (latestDaySpeech.action.speech_type === 'DEFEND') {
    speechType = 'Defense';
    const targetPlayer = gameState.players.find(p => p.player_id === latestDaySpeech.action.content?.target_id);
    speechContent = `Defends AI Player ${targetPlayer?.player_id || 'someone'}`;
  } else {
    speechType = 'Statement';
    speechContent = 'Shares information';
  }

  return (
    <Paper elevation={3} sx={{ 
      p: 2.5,
      mb: 3, 
      backgroundColor: 'rgba(255,152,0,0.07)', 
      color: '#fff',
      border: '1px solid rgba(255,152,0,0.3)',
      borderRadius: 2
    }}>
      <Typography variant="h6" gutterBottom sx={{ color: '#ffcc80', fontWeight: 'bold' }}>
        Current Speech - {speechType}
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'flex-start', mt: 1 }}>
        <Avatar src={getRoleImage(speechPlayer.current_role)} sx={{ 
          mr: 2, 
          mt: 1, 
          width: 56, 
          height: 56,
          border: '2px solid rgba(255,152,0,0.5)'
        }} />
        <Box sx={{ flexGrow: 1 }}>
          <Typography variant="h6" sx={{ fontWeight: 'medium' }}>
            {speechPlayer.name || `AI Player ${speechPlayer.player_id}`} ({speechPlayer.current_role})
          </Typography>
          <Typography variant="body1" sx={{ color: '#e0e0e0' }}>
            {speechContent}
          </Typography>
          
          {latestDaySpeech.action.content?.text && (
            <Card sx={{ 
              mt: 2, 
              p: 1, 
              bgcolor: 'rgba(255,152,0,0.1)', 
              borderRadius: '8px',
              border: '1px solid rgba(255,152,0,0.2)'
            }}>
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
      backgroundImage: `linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url("/day.png")`,
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      minHeight: '100vh',
      padding: '30px 0'
    }}>
      <Container maxWidth="xl">
        <Grid container spacing={4}>
          {/* Left Column - Main Content (8/12) */}
          <Grid item xs={12} lg={8}>
            <Grid container spacing={3}>
              {/* Game Phase Header - Full Width */}
              <Grid item xs={12}>
                <Card elevation={4} sx={{ 
                  mb: 3, 
                  bgcolor: 'rgba(0,0,0,0.6)', 
                  color: '#fff', 
                  p: 2,
                  border: '1px solid rgba(255,152,0,0.3)',
                  borderRadius: 2
                }}>
                  <CardContent>
                    <Typography variant="h4" gutterBottom sx={{ color: '#fff3e0', fontWeight: 'bold', textShadow: '2px 2px 4px rgba(0,0,0,0.7)' }}>
                      ☀️ Day Phase
                    </Typography>
                    <Typography variant="body1" sx={{ color: '#fff3e0', textShadow: '1px 1px 2px rgba(0,0,0,0.7)' }}>
                      Dawn has broken, players are sharing information. Speech round: {gameState.speech_round}/{gameState.max_speech_rounds}
                    </Typography>
                    
                    <Box sx={{ mt: 2 }}>
                      {currentPlayer && (
                        <Typography variant="body1" sx={{ mt: 1, display: 'flex', alignItems: 'center' }}>
                          Current Speaking Player: 
                          <Chip 
                            avatar={<Avatar src={getRoleImage(currentPlayer.current_role)} />}
                            label={currentPlayer.name || `AI Player ${currentPlayer.player_id}`}
                            color="warning"
                            sx={{ ml: 1 }}
                          />
                        </Typography>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
              
              {/* Game History - Full Width */}
              <Grid item xs={12}>
                <GameHistory history={gameState.history} />
              </Grid>
              
              {/* Day Speech Description - Full Width */}
              <Grid item xs={12}>
                <DaySpeech gameState={gameState} />
              </Grid>
            </Grid>
          </Grid>

          {/* Right Column - Supporting Info (4/12) */}
          <Grid item xs={12} lg={4}>
            <Box sx={{ position: 'sticky', top: '20px' }}>
              {/* Players Section */}
              <Box sx={{ mb: 4 }}>
                <Typography variant="h6" sx={{ mb: 2, color: '#fff', borderBottom: '1px solid rgba(255,255,255,0.1)', pb: 1 }}>
                  Game Participants
                </Typography>
                <PlayerList players={gameState.players} currentPlayerId={gameState.current_player_id} />
              </Box>
              
              {/* Divider */}
              <Divider sx={{ my: 3, backgroundColor: 'rgba(255,255,255,0.1)' }} />
              
              {/* Center Cards Section */}
              <Box>
                <Typography variant="h6" sx={{ mb: 2, color: '#fff', borderBottom: '1px solid rgba(255,255,255,0.1)', pb: 1 }}>
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

export default DayPhase; 