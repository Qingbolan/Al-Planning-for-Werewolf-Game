import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { GameProvider } from './context/GameContext';
import Home from './components/Home';
import Game from './components/Game';
import GameObserver from './components/observer/GameObserver';
import './App.css';

function App() {
  return (
    <GameProvider>
      <Router>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/game" element={<Game />} />
          <Route path="/game/observe" element={<GameObserver />} />
        </Routes>
      </Router>
    </GameProvider>
  );
}

export default App; 