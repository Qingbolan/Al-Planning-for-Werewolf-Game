import React from 'react';
import {
  Grid,
  Typography,
  Avatar,
  Box,
  Card,
  CardContent,
  Tooltip,
  Zoom
} from '@mui/material';

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

// Role name formatting
const formatRoleName = (role) => {
  return role.charAt(0).toUpperCase() + role.slice(1);
};

const CenterCards = ({ centerCards }) => {
  if (!centerCards || centerCards.length === 0) {
    return null;
  }

  return (
    <Box>
      <Grid container spacing={2}>
        {centerCards.map((role, index) => (
          <Grid item xs={4} key={index}>
            <Tooltip 
              title={`Center Card ${index + 1}: ${formatRoleName(role)}`}
              TransitionComponent={Zoom}
              placement="top"
              arrow
            >
              <Card 
                elevation={2}
                sx={{ 
                  backgroundColor: 'rgba(30,30,40,0.7)', 
                  color: '#fff', 
                  height: '100%',
                  position: 'relative',
                  border: '1px solid rgba(255,255,255,0.12)',
                  borderRadius: 2,
                  overflow: 'hidden',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    backgroundColor: 'rgba(30,30,40,0.9)',
                    transform: 'translateY(-3px)',
                    boxShadow: '0 6px 12px rgba(0,0,0,0.3)'
                  }
                }}
              >
                <Box 
                  sx={{ 
                    position: 'absolute', 
                    top: 0, 
                    left: 0, 
                    bgcolor: 'rgba(0,0,0,0.5)', 
                    color: '#fff',
                    px: 1,
                    py: 0.3,
                    borderBottomRightRadius: '8px',
                    fontSize: '0.7rem',
                    fontWeight: 'bold'
                  }}
                >
                  #{index + 1}
                </Box>
                <CardContent sx={{ 
                  p: 2, 
                  "&:last-child": { pb: 2 },
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <Avatar
                    src={getRoleImage(role)}
                    alt={role}
                    sx={{ 
                      width: 52, 
                      height: 52, 
                      mb: 1,
                      border: '2px solid rgba(255,255,255,0.2)',
                      boxShadow: '0 3px 8px rgba(0,0,0,0.3)'
                    }}
                  />
                  <Typography 
                    variant="subtitle2" 
                    align="center"
                    sx={{ 
                      fontWeight: 'bold',
                      fontSize: '0.85rem'
                    }}
                  >
                    {formatRoleName(role)}
                  </Typography>
                </CardContent>
              </Card>
            </Tooltip>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default CenterCards; 