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
    Divider,
    LinearProgress,
    Tooltip
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

// Format role name
const formatRoleName = (role) => {
    return role.charAt(0).toUpperCase() + role.slice(1);
};

// Vote results component
const VoteResults = ({ gameState }) => {
    if (!gameState || !gameState.history || gameState.history.length === 0) {
        return null;
    }

    // Calculate votes for each player
    const voteCount = {};
    gameState.players.forEach(player => {
        voteCount[player.player_id] = 0;
    });

    // Count votes
    gameState.history
        .filter(action => action.action && action.action.action_type === 'VOTE')
        .forEach(action => {
            const targetId = action.action.target_id;
            if (targetId !== undefined && voteCount[targetId] !== undefined) {
                voteCount[targetId]++;
            }
        });

    // Find eliminated player(s)
    const mostVotes = Math.max(...Object.values(voteCount));
    const eliminated = Object.keys(voteCount).filter(id => voteCount[id] === mostVotes && mostVotes > 0);

    // Get latest vote
    const latestVote = [...gameState.history]
        .filter(action => action.action && action.action.action_type === 'VOTE')
        .pop();

    return (
        <Paper elevation={3} sx={{
            p: 2.5,
            mb: 3,
            backgroundColor: 'rgba(156,39,176,0.07)',
            color: '#fff',
            border: '1px solid rgba(156,39,176,0.3)',
            borderRadius: 2
        }}>
            <Typography variant="h6" gutterBottom sx={{ color: '#ce93d8', fontWeight: 'bold' }}>
                Current Voting Status
            </Typography>

            {latestVote && (
                <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', p: 1, bgcolor: 'rgba(156,39,176,0.1)', borderRadius: 2 }}>
                    <Avatar
                        src={getRoleImage(gameState.players.find(p => p.player_id === latestVote.player_id)?.current_role)}
                        sx={{ 
                            mr: 2,
                            border: '2px solid rgba(156,39,176,0.4)'
                        }}
                    />
                    <Typography sx={{ fontWeight: 'medium' }}>
                        AI Player {latestVote.player_id} voted for AI Player {latestVote.action.target_id}
                    </Typography>
                </Box>
            )}

            {eliminated.length > 0 && mostVotes > 0 && (
                <Box sx={{ 
                    mb: 3, 
                    p: 2, 
                    backgroundColor: eliminated.length === 1 ? 'rgba(244,67,54,0.1)' : 'rgba(255,152,0,0.1)', 
                    borderRadius: 2,
                    border: eliminated.length === 1 ? '1px solid rgba(244,67,54,0.3)' : '1px solid rgba(255,152,0,0.3)',
                }}>
                    <Typography variant="subtitle1" sx={{ 
                        color: eliminated.length === 1 ? '#ef9a9a' : '#ffcc80',
                        fontWeight: 'bold',
                        mb: 1
                    }}>
                        {eliminated.length === 1
                            ? `Current Leader:`
                            : `Current Tie:`}
                    </Typography>
                    
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                        {eliminated.map(id => {
                            const player = gameState.players.find(p => p.player_id.toString() === id);
                            return (
                                <Chip 
                                    key={id}
                                    avatar={<Avatar src={getRoleImage(player?.current_role)} />}
                                    label={`AI Player ${id} (${mostVotes} votes)`}
                                    sx={{ 
                                        bgcolor: eliminated.length === 1 ? 'rgba(244,67,54,0.2)' : 'rgba(255,152,0,0.2)',
                                        color: '#fff',
                                        borderRadius: '16px',
                                        px: 1,
                                        '& .MuiChip-avatar': {
                                            border: eliminated.length === 1 ? '1px solid rgba(244,67,54,0.5)' : '1px solid rgba(255,152,0,0.5)',
                                        }
                                    }}
                                />
                            );
                        })}
                    </Box>
                </Box>
            )}

            <Grid container spacing={1}>
                {gameState.players.map((player) => {
                    const isEliminated = eliminated.includes(player.player_id.toString()) && mostVotes > 0;
                    const votePercentage = (voteCount[player.player_id] / gameState.players.length) * 100;
                    
                    return (
                        <Grid item xs={12} key={player.player_id}>
                            <Box sx={{
                                display: 'flex',
                                alignItems: 'center',
                                mb: 1.5,
                                backgroundColor: isEliminated ? 'rgba(156,39,176,0.1)' : 'rgba(0,0,0,0.3)',
                                p: 1.5,
                                borderRadius: '8px',
                                border: isEliminated ? '1px solid rgba(156,39,176,0.3)' : '1px solid rgba(255,255,255,0.05)',
                            }}>
                                <Avatar src={getRoleImage(player.current_role)} sx={{ 
                                    mr: 2,
                                    border: '2px solid rgba(255,255,255,0.2)'
                                }} />
                                <Box sx={{ flexGrow: 1 }}>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                                        <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                                            AI Player {player.player_id} ({formatRoleName(player.current_role)})
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
                                    
                                    <Box sx={{ mt: 1 }}>
                                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
                                            <Typography variant="caption" sx={{ color: '#bdbdbd', fontWeight: 'medium' }}>
                                                Votes: {voteCount[player.player_id]}
                                            </Typography>
                                            <Typography variant="caption" sx={{ color: '#bdbdbd' }}>
                                                {votePercentage.toFixed(0)}%
                                            </Typography>
                                        </Box>
                                        
                                        <LinearProgress 
                                            variant="determinate" 
                                            value={votePercentage} 
                                            sx={{ 
                                                height: 8, 
                                                borderRadius: 4,
                                                backgroundColor: 'rgba(255,255,255,0.1)',
                                                '& .MuiLinearProgress-bar': {
                                                    backgroundColor: isEliminated 
                                                        ? '#9c27b0' 
                                                        : voteCount[player.player_id] > 0 
                                                            ? '#2196f3' 
                                                            : '#9e9e9e'
                                                }
                                            }} 
                                        />
                                    </Box>
                                </Box>
                            </Box>
                        </Grid>
                    );
                })}
            </Grid>
        </Paper>
    );
};

const VotePhase = ({ gameState }) => {
    const currentPlayer = gameState.current_player_id !== null
        ? gameState.players.find(p => p.player_id === gameState.current_player_id)
        : null;

    return (
        <div style={{
            backgroundImage: `linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url("/voting.png")`,
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
                                    border: '1px solid rgba(156,39,176,0.3)',
                                    borderRadius: 2
                                }}>
                                    <CardContent>
                                        <Typography variant="h4" gutterBottom sx={{ color: '#e1bee7', fontWeight: 'bold', textShadow: '2px 2px 4px rgba(0,0,0,0.7)' }}>
                                            🗳️ Voting Phase
                                        </Typography>
                                        <Typography variant="body1" sx={{ color: '#e1bee7', textShadow: '1px 1px 2px rgba(0,0,0,0.7)' }}>
                                            Discussion ended, players are deciding who is the most suspicious...
                                        </Typography>

                                        <Box sx={{ mt: 2 }}>
                                            {currentPlayer && (
                                                <Typography variant="body1" sx={{ mt: 1, display: 'flex', alignItems: 'center' }}>
                                                    Current Voting Player:
                                                    <Chip
                                                        avatar={<Avatar src={getRoleImage(currentPlayer.current_role)} />}
                                                        label={`AI Player ${currentPlayer.player_id}`}
                                                        color="secondary"
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
                            
                            {/* Vote Results - Full Width */}
                            <Grid item xs={12}>
                                <VoteResults gameState={gameState} />
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

export default VotePhase; 