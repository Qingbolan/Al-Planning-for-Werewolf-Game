import React, { useEffect, useState, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Button,
  Avatar,
  Chip,
  Box,
  Card,
  CardContent,
  TextField,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Divider,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import { useGame } from '../context/GameContext';

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

const Game = () => {
  const navigate = useNavigate();
  const { gameState, player, messages, performAction, sendChatMessage, disconnect } = useGame();

  // 本地状态管理
  const [chatMessage, setChatMessage] = useState('');
  const [actionDialogOpen, setActionDialogOpen] = useState(false);
  const [selectedAction, setSelectedAction] = useState(null);
  const [actionParams, setActionParams] = useState({});

  // 检查玩家是否已加入游戏
  useEffect(() => {
    if (!player) {
      navigate('/');
    }
  }, [player, navigate]);

  // 调试日志：记录游戏状态以及玩家信息
  useEffect(() => {
    console.log('Current game state:', gameState);
    console.log('Player information:', player);
  }, [gameState, player]);

  // 从 gameState 中查找当前登录玩家在游戏内的详细信息
  const currentUser = useMemo(() => {
    if (gameState && gameState.players && player) {
      return gameState.players.find((p) => p.player_id === player.playerId);
    }
    return null;
  }, [gameState, player]);

  // 处理游戏结束返回大厅
  const handleBackToLobby = useCallback(() => {
    disconnect();
    navigate('/');
  }, [disconnect, navigate]);

  // 发送聊天消息（支持 Enter 快捷键）
  const handleSendMessage = useCallback(() => {
    if (chatMessage.trim()) {
      sendChatMessage(chatMessage.trim());
      setChatMessage('');
    }
  }, [chatMessage, sendChatMessage]);

  // 处理输入框回车触发
  const handleChatKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // 打开操作对话框
  const handleOpenActionDialog = useCallback((action) => {
    setSelectedAction(action);
    setActionParams({}); // 重置操作参数
    setActionDialogOpen(true);
  }, []);

  // 执行所选操作
  const handlePerformAction = useCallback(() => {
    if (selectedAction) {
      // 整合操作信息与参数
      const actionPayload = {
        ...selectedAction,
        ...actionParams,
      };
      performAction(actionPayload);
      setActionDialogOpen(false);
    }
  }, [selectedAction, actionParams, performAction]);

  // 如果未获取到游戏状态则显示加载中提示
  if (!gameState) {
    return (
      <Container
        sx={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100vh',
        }}
      >
        <CircularProgress size={60} sx={{ mb: 3 }} />
        <Typography variant="h5" sx={{ mb: 2 }}>
          正在连接游戏...
        </Typography>
        <Typography variant="body1" color="text.secondary">
          正在等待服务器响应，请稍候...
        </Typography>
        {player && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2">玩家ID: {player.playerId}</Typography>
          </Box>
        )}
      </Container>
    );
  }

  // 当前行动的玩家信息
  const currentPlayerIdx = gameState.players.findIndex(
    (p) => p.player_id === gameState.current_player_id
  );
  const currentPlayer = currentPlayerIdx >= 0 ? gameState.players[currentPlayerIdx] : null;

  // 判断是否为当前玩家回合（仅针对人类玩家）
  const isMyTurn =
    gameState.current_player_id !== null &&
    player &&
    gameState.current_player_id === player.playerId;

  // 根据游戏阶段渲染不同内容
  const renderGameContent = () => {
    if (gameState.game_over) {
      return (
        <Card
          sx={{
            mb: 3,
            bgcolor: 'info.light',
            color: 'info.contrastText',
            p: 2,
          }}
        >
          <CardContent>
            <Typography variant="h4" gutterBottom>
              Game Over!
            </Typography>
            <Typography variant="h5">
              {gameState.winner === 'werewolf'
                ? 'Werewolves Win!'
                : gameState.winner === 'villager'
                ? 'Villagers Win!'
                : 'Game ended, no winner'}
            </Typography>
            <Button variant="contained" color="primary" onClick={handleBackToLobby} sx={{ mt: 2 }}>
              Return to Lobby
            </Button>
          </CardContent>
        </Card>
      );
    }

    return (
      <>
        <Card
          sx={{
            mb: 3,
            bgcolor: 'primary.light',
            color: 'primary.contrastText',
            p: 2,
          }}
        >
          <CardContent>
            <Typography variant="h5" gutterBottom>
              {gameState.phase === 'NIGHT'
                ? '🌙 Night Phase'
                : gameState.phase === 'DAY'
                ? '☀️ Day Phase'
                : gameState.phase === 'VOTE'
                ? '🗳️ Voting Phase'
                : gameState.phase}
            </Typography>
            {currentPlayer && (
              <Typography variant="body1">
                Current Player: {currentPlayer.name} {currentPlayer.player_id === player?.playerId && ' (Your Turn)'}
              </Typography>
            )}
            {gameState.speech_round && (
              <Typography variant="body2">
                Speech Round: {gameState.speech_round}/{gameState.max_speech_rounds}
              </Typography>
            )}
          </CardContent>
        </Card>

        {isMyTurn && gameState.possible_actions && gameState.possible_actions.length > 0 && (
          <Card sx={{ mb: 3, bgcolor: 'success.light', p: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                It's your turn to act!
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mt: 1 }}>
                {gameState.possible_actions.map((action, index) => (
                  <Button
                    key={index}
                    variant="contained"
                    color="primary"
                    onClick={() => handleOpenActionDialog(action)}
                  >
                    {action.action_type === 'NIGHT_ACTION'
                      ? 'Perform Night Action'
                      : action.action_type === 'DAY_SPEECH'
                      ? 'Make a Statement'
                      : action.action_type === 'VOTE'
                      ? 'Choose a Target to Vote'
                      : action.action_type}
                  </Button>
                ))}
              </Box>
            </CardContent>
          </Card>
        )}
      </>
    );
  };

  return (
    <div className="game-container">
      <Container maxWidth="lg" sx={{ py: 3 }}>
        <Grid container spacing={3}>
          {/* 游戏标题和ID */}
          <Grid item xs={12}>
            <Typography variant="h4" gutterBottom>
              Werewolf Game
            </Typography>
            <Typography variant="body1" gutterBottom>
              Game ID: {gameState.game_id}
            </Typography>
          </Grid>

          {/* 游戏内容与玩家信息 */}
          <Grid item xs={12} md={8}>
            {/* 游戏状态 */}
            {renderGameContent()}

            {/* 玩家列表 */}
            <Paper sx={{ p: 2, mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                Players ({gameState.players.length})
              </Typography>
              <Grid container spacing={2}>
                {gameState.players.map((p, index) => (
                  <Grid item xs={6} sm={4} md={3} key={p.player_id || index}>
                    <Card
                      sx={{
                        border:
                          gameState.current_player_id === p.player_id
                            ? '2px solid #f50057'
                            : 'none',
                      }}
                    >
                      <CardContent>
                        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                          <Avatar src={getRoleImage(p.current_role)} sx={{ width: 60, height: 60, mb: 1 }} />
                          <Typography variant="subtitle1" noWrap>
                            {p.name}
                          </Typography>
                          <Chip
                            size="small"
                            label={p.is_human ? 'Player' : 'AI'}
                            color={p.is_human ? 'primary' : 'default'}
                            sx={{ mt: 0.5 }}
                          />
                          {p.current_role && (
                            <Typography variant="body2" sx={{ mt: 1 }}>
                              Role: {p.current_role}
                            </Typography>
                          )}
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Paper>

            {/* 游戏历史 */}
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Game History
              </Typography>
              <List>
                {gameState.history && gameState.history.length > 0 ? (
                  gameState.history.map((item, index) => {
                    // 获取执行操作的玩家名称
                    const actingPlayer =
                      item.player_id &&
                      gameState.players.find((p) => p.player_id === item.player_id)?.name;
                    return (
                      <React.Fragment key={index}>
                        <ListItem>
                          <ListItemText
                            primary={`Phase: ${item.phase} - Turn: ${item.turn}`}
                            secondary={
                              item.action
                                ? `Player ${actingPlayer || item.player_id} performed ${item.action.action_type} action`
                                : 'System event'
                            }
                          />
                        </ListItem>
                        {index < gameState.history.length - 1 && <Divider />}
                      </React.Fragment>
                    );
                  })
                ) : (
                  <ListItem>
                    <ListItemText primary="Game just started, no history yet" />
                  </ListItem>
                )}
              </List>
            </Paper>
          </Grid>

          {/* 聊天与信息面板 */}
          <Grid item xs={12} md={4}>
            {/* 个人信息面板 */}
            <Paper sx={{ p: 2, mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                Your Information
              </Typography>
              {currentUser ? (
                <>
                  <Typography variant="body1">Name: {currentUser.name}</Typography>
                  <Typography variant="body1">
                    Player ID: {player.playerId.slice(0, 8)}...
                  </Typography>
                  {currentUser.current_role && (
                    <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                      <Typography variant="body1" sx={{ mr: 1 }}>
                        Your Role:
                      </Typography>
                      <Chip
                        label={currentUser.current_role}
                        color={currentUser.team === 'werewolf' ? 'error' : 'success'}
                      />
                    </Box>
                  )}
                </>
              ) : (
                <Typography variant="body2">Loading your info...</Typography>
              )}
            </Paper>

            {/* 聊天区域 */}
            <Paper
              sx={{
                p: 2,
                height: '60vh',
                display: 'flex',
                flexDirection: 'column',
              }}
            >
              <Typography variant="h6" gutterBottom>
                Game Chat
              </Typography>
              <Box sx={{ flexGrow: 1, overflow: 'auto', mb: 2 }}>
                <List>
                  {messages && messages.length > 0 ? (
                    messages.map((msg, index) => (
                      <ListItem key={index} alignItems="flex-start">
                        <ListItemAvatar>
                          <Avatar>{msg.name.charAt(0)}</Avatar>
                        </ListItemAvatar>
                        <ListItemText
                          primary={msg.name}
                          secondary={msg.message}
                          secondaryTypographyProps={{
                            component: 'div',
                            sx: { wordBreak: 'break-word' },
                          }}
                        />
                      </ListItem>
                    ))
                  ) : (
                    <ListItem>
                      <ListItemText primary="No chat messages yet, start chatting!" />
                    </ListItem>
                  )}
                </List>
              </Box>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                  fullWidth
                  variant="outlined"
                  placeholder="Type a message..."
                  size="small"
                  value={chatMessage}
                  onChange={(e) => setChatMessage(e.target.value)}
                  onKeyDown={handleChatKeyDown}
                />
                <Button variant="contained" color="primary" onClick={handleSendMessage}>
                  Send
                </Button>
              </Box>
            </Paper>
          </Grid>
        </Grid>

        {/* 操作对话框 */}
        <Dialog
          open={actionDialogOpen}
          onClose={() => setActionDialogOpen(false)}
          maxWidth="sm"
          fullWidth
        >
          <DialogTitle>
            {selectedAction?.action_type === 'NIGHT_ACTION'
              ? 'Perform Night Action'
              : selectedAction?.action_type === 'DAY_SPEECH'
              ? 'Make a Statement'
              : selectedAction?.action_type === 'VOTE'
              ? 'Choose a Target to Vote'
              : 'Perform Action'}
          </DialogTitle>
          <DialogContent>
            {selectedAction?.action_type === 'NIGHT_ACTION' && (
              <>
                <Typography variant="body1" gutterBottom>
                  Selected Action: {selectedAction.action_name}
                </Typography>
                <Typography variant="subtitle2" gutterBottom>
                  Choose Target:
                </Typography>
                <Grid container spacing={1} sx={{ mt: 1 }}>
                  {selectedAction.targets?.map((targetId) => (
                    <Grid item key={targetId}>
                      <Button
                        variant={
                          actionParams.action_params?.target_id === targetId ? 'contained' : 'outlined'
                        }
                        onClick={() =>
                          setActionParams({
                            action_name: selectedAction.action_name,
                            action_params: { target_id: targetId },
                          })
                        }
                      >
                        {gameState.players.find((p) => p.player_id === targetId)?.name ||
                          `Central Card ${targetId + 1}`}
                      </Button>
                    </Grid>
                  ))}
                </Grid>
              </>
            )}

            {selectedAction?.action_type === 'DAY_SPEECH' && (
              <>
                <Typography variant="subtitle2" gutterBottom>
                  Choose Speech Type:
                </Typography>
                <Grid container spacing={1} sx={{ mb: 2 }}>
                  {selectedAction.speech_types?.map((type) => (
                    <Grid item key={type}>
                      <Button
                        variant={actionParams.speech_type === type ? 'contained' : 'outlined'}
                        onClick={() =>
                          setActionParams((prev) => ({
                            ...prev,
                            speech_type: type,
                          }))
                        }
                      >
                        {type === 'CLAIM'
                          ? 'Role Claim'
                          : type === 'ACCUSE'
                          ? 'Accuse'
                          : type === 'DEFEND'
                          ? 'Defend'
                          : type === 'QUESTION'
                          ? 'Question'
                          : 'General Comment'}
                      </Button>
                    </Grid>
                  ))}
                </Grid>

                <Typography variant="subtitle2" gutterBottom>
                  Statement Content:
                </Typography>
                <TextField
                  fullWidth
                  multiline
                  rows={3}
                  variant="outlined"
                  placeholder="Enter your statement..."
                  sx={{ mb: 2 }}
                  value={actionParams.content?.text || ''}
                  onChange={(e) =>
                    setActionParams((prev) => ({
                      ...prev,
                      content: { ...prev.content, text: e.target.value },
                    }))
                  }
                />

                {actionParams.speech_type === 'CLAIM' && (
                  <>
                    <Typography variant="subtitle2" gutterBottom>
                      Claim Role:
                    </Typography>
                    <Grid container spacing={1}>
                      {['villager', 'werewolf', 'seer', 'robber', 'troublemaker'].map((role) => (
                        <Grid item key={role}>
                          <Button
                            variant={
                              actionParams.content?.role_claim === role ? 'contained' : 'outlined'
                            }
                            onClick={() =>
                              setActionParams((prev) => ({
                                ...prev,
                                content: { ...prev.content, role_claim: role },
                              }))
                            }
                          >
                            {role}
                          </Button>
                        </Grid>
                      ))}
                    </Grid>
                  </>
                )}

                {(actionParams.speech_type === 'ACCUSE' ||
                  actionParams.speech_type === 'DEFEND' ||
                  actionParams.speech_type === 'QUESTION') && (
                  <>
                    <Typography variant="subtitle2" gutterBottom>
                      Choose Target Player:
                    </Typography>
                    <Grid container spacing={1}>
                      {gameState.players
                        .filter((p) => p.player_id !== gameState.current_player_id)
                        .map((p) => (
                          <Grid item key={p.player_id}>
                            <Button
                              variant={
                                actionParams.content?.target_id === p.player_id ? 'contained' : 'outlined'
                              }
                              onClick={() =>
                                setActionParams((prev) => ({
                                  ...prev,
                                  content: { ...prev.content, target_id: p.player_id },
                                }))
                              }
                            >
                              {p.name}
                            </Button>
                          </Grid>
                        ))}
                    </Grid>
                  </>
                )}
              </>
            )}

            {selectedAction?.action_type === 'VOTE' && (
              <>
                <Typography variant="subtitle2" gutterBottom>
                  Choose Voting Target:
                </Typography>
                <Grid container spacing={1}>
                  {gameState.players
                    .filter((p) => p.player_id !== gameState.current_player_id && p.is_alive)
                    .map((p) => (
                      <Grid item key={p.player_id}>
                        <Button
                          variant={
                            actionParams.target_id === p.player_id ? 'contained' : 'outlined'
                          }
                          onClick={() =>
                            setActionParams({
                              target_id: p.player_id,
                            })
                          }
                        >
                          {p.name}
                        </Button>
                      </Grid>
                    ))}
                </Grid>
              </>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setActionDialogOpen(false)}>Cancel</Button>
            <Button
              onClick={handlePerformAction}
              variant="contained"
              color="primary"
              disabled={
                (selectedAction?.action_type === 'NIGHT_ACTION' &&
                  !actionParams.action_params?.target_id) ||
                (selectedAction?.action_type === 'DAY_SPEECH' &&
                  (!actionParams.speech_type || !actionParams.content?.text)) ||
                (selectedAction?.action_type === 'VOTE' && !actionParams.target_id)
              }
            >
              Execute
            </Button>
          </DialogActions>
        </Dialog>
      </Container>
    </div>
  );
};

export default Game;
