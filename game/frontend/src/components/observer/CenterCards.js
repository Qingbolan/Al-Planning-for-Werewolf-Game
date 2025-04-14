import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Avatar,
  Box,
  Card,
  CardContent
} from '@mui/material';

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

const CenterCards = ({ centerCards }) => {
  if (!centerCards || centerCards.length === 0) {
    return null;
  }

  return (
    <Paper sx={{ p: 2, mb: 3, backgroundColor: 'rgba(0,0,0,0.75)', color: '#fff' }}>
      <Typography variant="h6" gutterBottom>
        中央牌
      </Typography>
      <Grid container spacing={2}>
        {centerCards.map((role, index) => (
          <Grid item xs={4} key={index}>
            <Card sx={{ backgroundColor: 'rgba(30,30,40,0.9)', color: '#fff' }}>
              <CardContent>
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <Avatar
                    src={getRoleImage(role)}
                    sx={{ width: 50, height: 50, mb: 1 }}
                  />
                  <Typography variant="body2">
                    中央牌 {index + 1}: {role}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Paper>
  );
};

export default CenterCards; 