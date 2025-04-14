from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

# 创建FastAPI应用
app = FastAPI(
    title="Werewolf Game API",
    description="API for the One Night Ultimate Werewolf game",
    version="1.0.0"
)

# 添加CORS中间件 - 允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # React开发服务器
        "http://localhost:8080",    # Vue.js开发服务器
        "http://localhost:4200",    # Angular开发服务器
        "http://localhost:5000",    # 其他可能的前端服务器
        "http://localhost:8000"     # 同一域名访问
    ],  
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# 导入游戏API路由
try:
    from game.backend.api import router as game_router
    from game.backend.game_manager import GameManager
    
    # 创建游戏管理器实例
    game_manager = GameManager()
    
    # 注册API路由 - 注意：api.py中的路由前缀已经包含/api，所以这里不需要再添加
    app.include_router(game_router)
    
    print("Successfully registered game API routes from api.py")
except Exception as e:
    import traceback
    print(f"Error loading game API routes: {e}")
    traceback.print_exc()

# 添加根路由，提供API信息
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Werewolf Game API",
        "documentation": "/docs",
        "openapi": "/openapi.json"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=18000)
