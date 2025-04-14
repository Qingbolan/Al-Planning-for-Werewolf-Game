import React from 'react';
import {
    Box,
    Typography,
    Card,
    CardContent,
    Avatar,
    Chip,
    Paper,
    Grid
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

// 投票结果组件
const VoteResults = ({ gameState }) => {
    if (!gameState || !gameState.history || gameState.history.length === 0) {
        return null;
    }

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

    // 获取最近的投票
    const latestVote = [...gameState.history]
        .filter(action => action.action && action.action.action_type === 'VOTE')
        .pop();

    return (
        <Paper sx={{
            p: 2,
            mb: 3,
            backgroundColor: 'rgba(0,0,0,0.75)',
            color: '#fff',
            border: '1px solid rgba(244,67,54,0.5)'
        }}>
            <Typography variant="h6" gutterBottom sx={{ color: '#ef9a9a' }}>
                当前投票情况
            </Typography>

            {latestVote && (
                <Box sx={{ mb: 3, display: 'flex', alignItems: 'center' }}>
                    <Avatar
                        src={getRoleImage(gameState.players.find(p => p.player_id === latestVote.player_id)?.current_role)}
                        sx={{ mr: 2 }}
                    />
                    <Typography>
                        AI玩家 {latestVote.player_id} 投票给了 AI玩家 {latestVote.action.target_id}
                    </Typography>
                </Box>
            )}

            {eliminated.length > 0 && mostVotes > 0 && (
                <Box sx={{ mb: 2, p: 1, backgroundColor: 'rgba(244,67,54,0.1)', borderRadius: '4px' }}>
                    <Typography variant="body1" sx={{ color: '#ff8a80' }}>
                        {eliminated.length === 1
                            ? `目前票数最高: AI玩家 ${eliminated[0]}（${mostVotes}票）`
                            : `目前平票: ${eliminated.map(id => `AI玩家 ${id}`).join('，')}（各${mostVotes}票）`}
                    </Typography>
                </Box>
            )}

            <Grid container spacing={1}>
                {gameState.players.map((player) => (
                    <Grid item xs={12} key={player.player_id}>
                        <Box sx={{
                            display: 'flex',
                            alignItems: 'center',
                            mb: 1,
                            backgroundColor: eliminated.includes(player.player_id.toString()) ? 'rgba(244,67,54,0.2)' : 'transparent',
                            p: 1,
                            borderRadius: '4px'
                        }}>
                            <Avatar src={getRoleImage(player.current_role)} sx={{ mr: 1 }} />
                            <Box sx={{ flexGrow: 1 }}>
                                <Typography variant="body1">
                                    AI玩家 {player.player_id} ({player.current_role})
                                </Typography>
                                <Box sx={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    mt: 0.5
                                }}>
                                    <Box sx={{
                                        width: `${(voteCount[player.player_id] / gameState.players.length) * 100}%`,
                                        height: '8px',
                                        backgroundColor: eliminated.includes(player.player_id.toString()) ? '#f44336' : '#2196f3',
                                        borderRadius: '4px',
                                        mr: 1,
                                        minWidth: '4px'
                                    }} />
                                    <Typography variant="body2" sx={{ color: '#bdbdbd' }}>
                                        {voteCount[player.player_id]} 票
                                    </Typography>
                                </Box>
                            </Box>
                            <Chip
                                label={`${player.team || '未知阵营'}`}
                                color={player.team === 'werewolf' ? 'error' : 'success'}
                                size="small"
                                sx={{ ml: 1 }}
                            />
                        </Box>
                    </Grid>
                ))}
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
            backgroundImage: `url("/voting.png")`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            minHeight: '100vh',
            padding: '20px 0'
        }}>
            <Box sx={{ maxWidth: 'lg', mx: 'auto', px: 2 }}>
                {/* 投票阶段标题 */}
                <Card sx={{ mb: 3, bgcolor: 'rgba(0,0,0,0.75)', color: '#fff', p: 2 }}>
                    <CardContent>
                        <Typography variant="h5" gutterBottom sx={{ color: '#fff3e0', fontWeight: 'bold', textShadow: '2px 2px 4px rgba(0,0,0,0.7)' }}>
                            🗳️ 投票阶段
                        </Typography>
                        <Typography variant="body1" sx={{ color: '#fff3e0', textShadow: '1px 1px 2px rgba(0,0,0,0.7)' }}>
                            讨论结束，玩家们正在决定谁是最可疑的...
                        </Typography>

                        <Box sx={{ mt: 2 }}>

                            {currentPlayer && (
                                <Typography variant="body1" sx={{ mt: 1, display: 'flex', alignItems: 'center' }}>
                                    当前投票玩家:
                                    <Chip
                                        avatar={<Avatar src={getRoleImage(currentPlayer.current_role)} />}
                                        label={`AI玩家 ${currentPlayer.player_id}`}
                                        color="error"
                                        sx={{ ml: 1 }}
                                    />
                                </Typography>
                            )}
                        </Box>
                    </CardContent>
                </Card>

                {/* 投票结果 */}
                <VoteResults gameState={gameState} />

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

export default VotePhase; 