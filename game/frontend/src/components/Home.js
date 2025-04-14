import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useGame } from '../context/GameContext';
import {
  Container, Paper, Typography, Button,
  Grid, CircularProgress, Box, Snackbar, Alert,
  MenuItem, FormControl, InputLabel, Select
} from '@mui/material';
import { gameApi } from '../services/api';

const Home = () => {
  const navigate = useNavigate();
  const { createGame, joinGame, loading, error } = useGame();
  
  // Default player name
  const playerName = 'Player';
  const [numPlayers] = useState(6);
  const [showError, setShowError] = useState(false);
  const [testGameType, setTestGameType] = useState('heuristic');
  
  // Display error message
  useEffect(() => {
    if (error) {
      setShowError(true);
    }
  }, [error]);

  // Create standard game
  const handleCreateGame = async () => {
    try {
      // Configure new game
      const gameConfig = {
        num_players: Number(numPlayers),
        human_player_count: 1,
        max_speech_rounds: 3,
        ai_agent_type: "heuristic"
      };
      
      // Create game
      const result = await createGame(gameConfig);
      console.log("Game created successfully:", result);
      
      // Join the created game
      await joinGame(result.game_id, playerName);
      
      // Navigate to game page
      navigate('/game');
    } catch (error) {
      console.error('Failed to create game:', error);
    }
  };

  // Create test game
  const handleCreateTestGame = async () => {
    try {
      // 创建测试游戏 - 由于我们删除了 createObserverTestGame 函数，现在使用普通的 createTestGame
      const result = await gameApi.createTestGame(testGameType);
      console.log("Test game created successfully:", result);
      
      if (result.success) {
        // 由于现在使用普通测试游戏，我们需要导航到普通游戏页面
        navigate('/game');
      } else {
        console.error('Failed to create test game:', result.error);
      }
    } catch (error) {
      console.error('Failed to create test game:', error);
    }
  };

  return (
    <div style={{
      backgroundImage: 'url("/cover.png")',
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    }}>
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Box sx={{ textAlign: 'center', mb: 5 }}>
          <Typography 
            variant="h1" 
            sx={{ 
              color: '#f8e4b7', 
              fontFamily: "'Western', serif",
              fontSize: { xs: '3rem', sm: '4rem', md: '5rem' },
              fontWeight: 'bold',
              textShadow: '3px 3px 5px rgba(0,0,0,0.7)',
              letterSpacing: '0.05em',
              mb: 1
            }}
          >
            WEREWOLF AI
          </Typography>
          <Typography 
            variant="h5" 
            sx={{ 
              color: '#f8d692', 
              fontFamily: "'Courier New', monospace",
              letterSpacing: '0.15em',
              textShadow: '2px 2px 4px rgba(0,0,0,0.8)'
            }}
          >
            A MULTI-AGENT REINFORCEMENT LEARNING GAME
          </Typography>
        </Box>
        
        <Paper elevation={3} sx={{ 
          p: 4, 
          backgroundColor: 'rgba(20,20,30,0.8)', 
          color: 'white',
          borderRadius: '8px',
          boxShadow: '0 8px 20px rgba(0,0,0,0.4)'
        }}>
          <Grid container spacing={4}>
            {/* Standard Game */}
            <Grid item xs={12} md={6}>
              <Paper 
                elevation={2} 
                sx={{ 
                  p: 3, 
                  height: '100%', 
                  backgroundColor: 'rgba(26,41,54,0.9)',
                  border: '1px solid rgba(77,166,255,0.3)',
                  borderRadius: '8px'
                }}
              >
                <Typography variant="h5" gutterBottom sx={{ 
                  color: '#4da6ff',
                  fontFamily: "'Courier New', monospace",
                  letterSpacing: '0.05em'
                }}>
                  CREATE STANDARD GAME
                </Typography>
                <Typography variant="body2" sx={{ mb: 2, color: '#ccc' }}>
                Play as a human player against multiple Al agents
                </Typography>
                <Typography variant="body2" sx={{ mb: 3, color: '#aaa' }}>
                  • 6 players (1 human + 5 AI agents)<br />
                  • Roles: 2 werewolves, 1 seer, 3 villagers<br />
                  • Using heuristic AI agents
                </Typography>
                
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleCreateGame}
                  disabled={loading}
                  fullWidth
                  sx={{ 
                    mt: 1,
                    bgcolor: '#1a5f9c',
                    '&:hover': {
                      bgcolor: '#2a70ad'
                    },
                    fontFamily: "'Courier New', monospace",
                    letterSpacing: '0.05em'
                  }}
                >
                  {loading ? <CircularProgress size={24} /> : 'START GAME'}
                </Button>
              </Paper>
            </Grid>
            
            {/* Agent Test Game */}
            <Grid item xs={12} md={6}>
              <Paper 
                elevation={2} 
                sx={{ 
                  p: 3, 
                  height: '100%', 
                  backgroundColor: 'rgba(54,36,26,0.9)',
                  border: '1px solid rgba(255,153,102,0.3)',
                  borderRadius: '8px'
                }}
              >
                <Typography variant="h5" gutterBottom sx={{ 
                  color: '#ff9966',
                  fontFamily: "'Courier New', monospace",
                  letterSpacing: '0.05em'
                }}>
                  CREATE AGENT TEST GAME
                </Typography>
                
                <FormControl 
                  fullWidth 
                  variant="outlined" 
                  sx={{ 
                    mb: 2,
                    '& .MuiOutlinedInput-root': {
                      color: '#f8d692',
                      '& fieldset': {
                        borderColor: 'rgba(255,153,102,0.5)',
                      },
                      '&:hover fieldset': {
                        borderColor: 'rgba(255,153,102,0.8)',
                      },
                    },
                    '& .MuiInputLabel-root': {
                      color: '#f8d692',
                    },
                    '& .MuiSelect-icon': {
                      color: '#f8d692',
                    }
                  }}
                >
                  <InputLabel id="test-game-type-label">Game Mode</InputLabel>
                  <Select
                    labelId="test-game-type-label"
                    value={testGameType}
                    onChange={(e) => setTestGameType(e.target.value)}
                    label="Game Mode"
                  >
                    <MenuItem value="random">All Random Agents</MenuItem>
                    <MenuItem value="heuristic">All Heuristic Agents</MenuItem>
                    <MenuItem value="random_villager_heuristic_werewolf">Random Villagers, Heuristic Werewolves</MenuItem>
                    <MenuItem value="heuristic_villager_random_werewolf">Heuristic Villagers, Random Werewolves</MenuItem>
                    <MenuItem value="random_mix">Completely Random Agent Assignment</MenuItem>
                  </Select>
                </FormControl>
                
                <Typography variant="body2" sx={{ mb: 2, color: '#ccc' }}>
                  Create a test game with only AI agents (no human players)
                </Typography>
                <Typography variant="body2" sx={{ mb: 3, color: '#aaa' }}>
                  • Pure AI simulation game<br />
                  • Start automatically after creation<br />
                  • Watch how different AI agent types interact
                </Typography>
                
                <Button
                  variant="contained"
                  color="secondary"
                  onClick={handleCreateTestGame}
                  disabled={loading}
                  fullWidth
                  sx={{ 
                    mt: 1,
                    bgcolor: '#b35c2d',
                    '&:hover': {
                      bgcolor: '#c46d3e'
                    },
                    fontFamily: "'Courier New', monospace",
                    letterSpacing: '0.05em'
                  }}
                >
                  {loading ? <CircularProgress size={24} /> : 'START TEST GAME'}
                </Button>
              </Paper>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Error notification */}
        <Snackbar 
          open={showError} 
          autoHideDuration={6000} 
          onClose={() => setShowError(false)}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert onClose={() => setShowError(false)} severity="error" variant="filled">
            {error}
          </Alert>
        </Snackbar>
      </Container>
    </div>
  );
};

export default Home; 