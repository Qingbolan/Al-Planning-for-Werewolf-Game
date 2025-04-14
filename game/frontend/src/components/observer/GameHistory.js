import React from 'react';
import {
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  Divider
} from '@mui/material';

const GameHistory = ({ history }) => {
  return (
    <Paper sx={{ p: 2, backgroundColor: 'rgba(0,0,0,0.75)', color: '#fff' }}>
      <Typography variant="h6" gutterBottom>
        游戏历史
      </Typography>
      <List>
        {history.map((item, index) => (
          <React.Fragment key={index}>
            <ListItem sx={{ 
              backgroundColor: index % 2 === 0 ? 'rgba(30,30,40,0.5)' : 'transparent',
              borderRadius: '4px'
            }}>
              <ListItemText
                primary={
                  <Typography sx={{ color: '#fff' }}>
                    {item.phase === 'night' ? '🌙 夜晚阶段' : 
                     item.phase === 'day' ? '☀️ 白天阶段' : 
                     item.phase === 'vote' ? '🗳️ 投票阶段' : 
                     item.phase} - 回合: {item.turn}
                  </Typography>
                }
                secondary={
                  <Typography sx={{ color: '#ccc' }}>
                    {item.action
                      ? `AI玩家 ${item.player_id} ${
                          item.action.action_type === 'NIGHT_ACTION' 
                            ? '执行了夜晚能力' 
                            : item.action.action_type === 'DAY_SPEECH' 
                            ? '发表了言论' 
                            : item.action.action_type === 'VOTE'
                            ? '投票给了 AI玩家 ' + item.action.target_id
                            : '执行了操作'}`
                      : '系统事件'
                  }
                  </Typography>
                }
              />
            </ListItem>
            {index < history.length - 1 && <Divider sx={{ backgroundColor: 'rgba(255,255,255,0.1)' }} />}
          </React.Fragment>
        ))}
        {(!history || history.length === 0) && (
          <ListItem>
            <ListItemText 
              primary="游戏刚刚开始，暂无历史记录" 
              primaryTypographyProps={{ color: '#ccc' }}
            />
          </ListItem>
        )}
      </List>
    </Paper>
  );
};

export default GameHistory; 