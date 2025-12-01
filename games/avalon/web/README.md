# Avalon Game Web Interface

网页应用，支持观察模式和参与模式。

## 功能

### 观察模式 (Observe Mode)
- 观察主持人和所有agent的发言
- 实时查看游戏状态
- 只读模式，不参与游戏

### 参与模式 (Participate Mode)
- 扮演user agent参与游戏
- 只能看到agent observe的信息
- 可以输入发言和做出决策

## 安装依赖

确保已安装以下Python包：
```bash
pip install fastapi uvicorn websockets
```

## 使用方法

### 启动Web服务器

#### 观察模式
```bash
python games/avalon/web/run_web_game.py --mode observe --auto-start
```

#### 参与模式
```bash
python games/avalon/web/run_web_game.py --mode participate --user-agent-id 0 --auto-start
```

### 命令行参数

- `--mode`: 游戏模式，`observe` 或 `participate` (默认: `observe`)
- `--user-agent-id`: 参与模式下，用户扮演的agent ID (默认: 0)
- `--num-players`: 玩家数量 (默认: 5)
- `--language`: 语言，`en` 或 `zh`/`cn`/`chinese` (默认: `en`)
- `--host`: 服务器主机地址 (默认: `0.0.0.0`)
- `--port`: 服务器端口 (默认: 8000)
- `--auto-start`: 自动启动游戏（如果不设置，需要手动触发游戏开始）

### 访问Web界面

启动服务器后，在浏览器中访问：

- 主页: `http://localhost:8000`
- 观察模式: `http://localhost:8000/observe`
- 参与模式: `http://localhost:8000/participate`

## 架构说明

### 后端组件

1. **WebUserInput**: 处理前端用户输入
2. **WebUserAgent**: 扮演用户的agent，同步观察信息到前端
3. **ObserveAgent**: 观察模式下，同步所有消息到前端
4. **GameStateManager**: 管理游戏状态和WebSocket连接
5. **Web服务器**: FastAPI + WebSocket服务器

### 前端组件

1. **观察模式界面**: 显示所有消息
2. **参与模式界面**: 显示user agent的观察，并提供输入功能

### 数据流

#### 观察模式
```
游戏流程 → MsgHub广播消息 → ObserveAgent.observe() → WebSocket → 前端显示
```

#### 参与模式
```
游戏流程 → MsgHub广播消息 → WebUserAgent.observe() → WebSocket → 前端显示
用户输入 → WebSocket → WebUserInput → WebUserAgent.reply() → 游戏流程
```

## 技术细节

### 消息格式

#### 前端到后端
```json
{
  "type": "user_input",
  "agent_id": "agent_id",
  "content": "用户输入内容"
}
```

#### 后端到前端
```json
{
  "type": "message",
  "sender": "Moderator|Player0|...",
  "content": "消息内容",
  "role": "assistant|user",
  "timestamp": "2024-01-01T00:00:00"
}
```

### 游戏流程集成

`game.py`中的`avalon_game`函数已支持Web模式：
- 添加了`web_mode`和`web_observe_agent`参数
- 在观察模式下，将`ObserveAgent`添加到所有`MsgHub`的participants中
- 保持向后兼容，不影响RL Rollout Engine的使用

## 注意事项

1. 确保设置了正确的环境变量（如`MODEL_NAME`和`API_KEY`）
2. Web模式下的游戏日志默认关闭（`log_dir=None`）
3. 观察模式下，`ObserveAgent`会接收到所有消息，但不参与游戏逻辑
4. 参与模式下，`WebUserAgent`只同步自己观察到的信息到前端

## 开发

### 文件结构

```
games/avalon/web/
├── ARCHITECTURE.md          # 架构设计文档
├── README.md                # 本文件
├── __init__.py
├── web_user_input.py        # WebUserInput类
├── web_agent.py             # WebUserAgent和ObserveAgent类
├── server.py                # Web服务器
├── game_state_manager.py    # 游戏状态管理器
├── run_web_game.py          # Web游戏启动脚本
└── static/                  # 前端静态文件
    ├── index.html           # 主页面
    ├── observe.html         # 观察模式页面
    ├── participate.html     # 参与模式页面
    ├── css/
    │   └── style.css
    └── js/
        ├── websocket.js     # WebSocket客户端
        ├── observe.js       # 观察模式逻辑
        └── participate.js   # 参与模式逻辑
```

