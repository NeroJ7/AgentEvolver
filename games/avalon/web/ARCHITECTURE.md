# Avalon Web应用架构设计

## 概述

本架构设计用于在avalon游戏中实现网页应用，支持两种模式：
1. **观察模式**：用户可以在前端观察主持人以及所有agent的发言
2. **参与模式**：用户扮演user agent，只能看到agent observe的信息，并可以参与发言

## 设计原则

1. **最小改动原则**：尽量少改动`game.py`中的workflow，因为这部分会在RL Rollout Engine中复用
2. **可扩展性**：通过agent的`observe`方法来同步信息，不侵入游戏逻辑
3. **实时通信**：使用WebSocket实现前后端实时通信

## 架构组件

### 后端组件

#### 1. WebUserInput (`web_user_input.py`)
- **职责**：等待和处理前端用户输入（替代terminal input）
- **实现**：继承`UserInputBase`，通过WebSocket队列等待前端输入
- **接口**：
  - `async def __call__(agent_id, agent_name, structured_model)`: 等待并返回用户输入

#### 2. WebUserAgent (`web_agent.py`)
- **职责**：扮演用户，在observe方法中同步信息到前端
- **实现**：继承或包装`UserAgent`，重写`observe`方法
- **功能**：
  - 使用`WebUserInput`作为输入方法
  - 在`observe`方法中，将观察到的信息通过WebSocket发送到前端
  - 只发送该agent观察到的信息（参与模式）

#### 3. ObserveAgent (`web_agent.py`)
- **职责**：在观察模式下，参与到各个msghub，在observe方法中向前端同步信息
- **实现**：继承`AgentBase`，实现`observe`方法
- **功能**：
  - 参与到所有MsgHub中
  - 在`observe`方法中，将所有消息（主持人+所有agent）通过WebSocket发送到前端
  - 不参与游戏逻辑，只观察

#### 4. Web服务器 (`server.py`)
- **职责**：提供WebSocket服务和HTTP服务
- **实现**：使用FastAPI + WebSocket
- **功能**：
  - WebSocket连接管理
  - 消息队列管理（用户输入队列、消息广播队列）
  - 游戏状态管理
  - 提供静态文件服务

#### 5. 游戏状态管理器 (`game_state_manager.py`)
- **职责**：管理游戏状态和消息队列
- **功能**：
  - 管理用户输入队列（按agent_id索引）
  - 管理消息广播队列
  - 管理WebSocket连接

### 前端组件

#### 1. 观察模式界面
- **显示内容**：
  - 主持人（Moderator）的所有发言
  - 所有agent的发言
  - 游戏状态信息（当前阶段、任务结果等）
- **交互**：只读，不提供输入功能

#### 2. 参与模式界面
- **显示内容**：
  - 只显示user agent观察到的信息
  - 游戏状态信息
- **交互**：
  - 提供输入框，用户可以输入发言
  - 在需要用户输入时（如投票、选择团队等），显示相应的输入界面

## 数据流

### 观察模式
```
游戏流程 (game.py)
  ↓
MsgHub广播消息
  ↓
ObserveAgent.observe()
  ↓
WebSocket发送到前端
  ↓
前端显示所有消息
```

### 参与模式
```
游戏流程 (game.py)
  ↓
MsgHub广播消息
  ↓
WebUserAgent.observe()
  ↓
WebSocket发送到前端（仅该agent观察到的信息）
  ↓
前端显示user agent的观察
  ↓
用户输入
  ↓
WebSocket发送到后端
  ↓
WebUserInput返回输入
  ↓
WebUserAgent.reply()
  ↓
游戏流程继续
```

## 实现细节

### 1. 消息格式

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

```json
{
  "type": "game_state",
  "phase": "Team Selection|Team Voting|Quest Voting|Assassination",
  "mission_id": 1,
  "round_id": 1,
  "leader": 0
}
```

```json
{
  "type": "user_input_request",
  "agent_id": "agent_id",
  "prompt": "请输入你的选择..."
}
```

### 2. 游戏流程集成

在`game.py`中，需要做最小改动：
- 在观察模式下，将`ObserveAgent`添加到所有`MsgHub`的participants中
- 在参与模式下，使用`WebUserAgent`替代`UserAgent`

可以通过参数控制：
```python
async def avalon_game(
    agents: list[AgentBase], 
    config: AvalonBasicConfig,
    log_dir: str | None = None,
    language: str = "en",
    web_mode: str = None,  # "observe" | "participate" | None
    web_observe_agent: AgentBase | None = None,  # 观察模式下的观察agent
) -> bool:
```

### 3. WebSocket连接管理

- 每个游戏实例维护一个WebSocket连接
- 使用`asyncio.Queue`管理消息队列
- 用户输入队列：`{agent_id: Queue}`
- 消息广播队列：`Queue`

## 文件结构

```
games/avalon/web/
├── ARCHITECTURE.md          # 架构设计文档
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
        ├── observe.js      # 观察模式逻辑
        └── participate.js   # 参与模式逻辑
```

## 使用方式

### 启动Web服务器
```bash
python games/avalon/web/run_web_game.py --mode observe --port 8000
python games/avalon/web/run_web_game.py --mode participate --port 8000 --user-agent-id 0
```

### 访问前端
打开浏览器访问 `http://localhost:8000`

