import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// 在入口文件增加清理代码
// 这可以帮助在开发时避免使用过时的localStorage数据
if (process.env.NODE_ENV === 'development') {
  // 清除所有可能导致问题的旧数据
  // localStorage.clear(); // 如果需要完全清理，可以使用这行
  
  // 或者，只清除特定的键
  const keysToCheck = ['gameId', 'playerId', 'playerName'];
  keysToCheck.forEach(key => {
    const value = localStorage.getItem(key);
    if (value) {
      console.log(`已保存的${key}:`, value);
    }
  });
}

// 在应用启动时立即清除可能导致问题的localStorage项
localStorage.removeItem('observerGameId'); 