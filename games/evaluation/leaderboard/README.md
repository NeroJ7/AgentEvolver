# Arena Leaderboard

Multi-game evaluation system with persistent storage, fair model assignment, and Elo-based rankings.

## Features

- **Fair Assignment**: Weighted random selection balances game distribution across models
- **Elo Ratings**: Pairwise Elo ratings track model performance
- **Persistent Storage**: Thread-safe JSON database with incremental updates
- **Multi-Game Support**: Avalon and Diplomacy (extensible via lazy loading)
- **Role Statistics**: Win rates tracked per role (game-specific)
- **API Rate Limiting**: Configurable delays to prevent API overload

## Quick Start

### Supported Games

- **Avalon**: 5 players (Merlin, Servant, Assassin, Minion)
- **Diplomacy**: 7 players (one per power: AUSTRIA, ENGLAND, FRANCE, GERMANY, ITALY, RUSSIA, TURKEY)

### Run Evaluation

```bash
# Avalon
python games/evaluation/leaderboard/run_arena.py \
    --game avalon \
    --config games/games/avalon/configs/arena_config.yaml \
    --num-games 200 \
    --max-workers 10

# Diplomacy
python games/evaluation/leaderboard/run_arena.py \
    --game diplomacy \
    --config games/games/diplomacy/configs/arena_config.yaml \
    --num-games 100 \
    --max-workers 10
```

### Continue or Add Models

The system automatically loads existing leaderboard data. To add models or continue:

1. Add models to config:
```yaml
arena:
  models:
    - qwen-plus
    - qwen3-max
    - new-model-name  # Add here
```

2. Run evaluation (existing data is preserved):
```bash
python games/evaluation/leaderboard/run_arena.py \
    --game avalon \
    --config games/games/avalon/configs/arena_config.yaml \
    --num-games 100
```

## Command-Line Options

| Option | Short | Default | Description |
|--------|-------|----------|-------------|
| `--game` | `-g` | *required* | Game name (`avalon` or `diplomacy`) |
| `--config` | `-c` | *required* | Path to arena config YAML |
| `--num-games` | `-n` | 200 | Number of games to run |
| `--max-workers` | `-w` | 10 | Maximum parallel workers |
| `--experiment-name` | | `arena_leaderboard_{game}` | Experiment name for logs |
| `--leaderboard-db` | | `games/evaluation/leaderboard/leaderboard_{game}.json` | Database file path |
| `--api-call-interval` | | 0.0 | Seconds between API calls (0.0 = no limit) |

### Rate Limiting

To prevent API rate limit errors with high concurrency:

```bash
--api-call-interval 0.5  # Recommended: 0.5-0.6s for 10 workers
```

**Recommended intervals** (for qwen-max, RPM=1200):
- 5 workers: `0.3-0.4s`
- 10 workers: `0.5-0.6s`
- 20 workers: `1.0-1.2s`

## Configuration

Config files inherit from `default_config.yaml` using Hydra. Key sections:

### Arena Section
```yaml
arena:
  models: [qwen-plus, qwen3-max, ...]  # Models to evaluate
  seed: 42                              # Random seed (offset by game_id)
  elo_initial: 1500                    # Initial Elo rating
  elo_k: 32                            # Elo K-factor
```

### Game Section
```yaml
game:
  name: avalon                          # or diplomacy
  num_players: 5                        # Avalon: 5, Diplomacy: 7
  language: en                         # en or zh
  log_dir: games/logs/arena
  # Diplomacy-specific:
  # power_names: [AUSTRIA, ENGLAND, ...]
  # max_phases: 20
  # negotiation_rounds: 3
```

### Default Role Section
```yaml
default_role:
  trainable: false
  act_by_user: false
  model:
    url:  # From OPENAI_BASE_URL
    temperature: 0.7
    max_tokens: 2048
  agent:
    type: ThinkingReActAgent
    kwargs: {}  # Diplomacy: add memory config
```

See `games/games/{game}/configs/arena_config.yaml` for complete examples.

## Leaderboard Data

**Storage**: `games/evaluation/leaderboard/leaderboard_{game_name}.json`

**Contents**:
- Model statistics (Elo, games, wins, role-specific stats)
- Game history with timestamps
- Elo configuration (initial rating, K-factor)
- Balance statistics (computed on-the-fly)

**Features**:
- Thread-safe incremental updates
- Automatic loading on startup
- Supports adding models mid-evaluation
- Resumable after interruption

## Output Format

The leaderboard displays:
- Rankings by Elo (descending)
- Overall win rate and total games per model
- Role-specific win rates (e.g., Merlin, Servant for Avalon)
- Row/column averages
- Balance statistics (warnings if ratio < 0.8)
- Models with insufficient games marked with `*` (< 80% of max)

## How It Works

### Fair Model Assignment
- Weighted selection: `weight = 1 / (game_count + 1)`
- Ensures diversity (no duplicates when possible)
- Real-time game count updates for fairness
- Balance ratio monitored (warnings if < 0.8)

### Elo Rating System
- **Formula**: `new_elo = old_elo + k * (actual_score - expected_score)`
- **Expected score**: `1 / (1 + 10^((opponent_elo - my_elo) / 400))`
- Pairwise updates for all model pairs in each game
- Handles binary (0/1) and continuous scores via normalization

### Thread Safety
All database operations use locks, enabling:
- Concurrent game execution
- Safe incremental updates
- Real-time statistics without corruption

## Workflow

1. Load/create leaderboard database
2. Register models from config (new models get initial Elo)
3. For each game:
   - Query current game counts
   - Calculate weights (fewer games = higher weight)
   - Assign models to roles (weighted random, ensures diversity)
   - Execute game
   - Update statistics and Elo ratings
   - Save database (thread-safe)
4. Display final leaderboard

## Testing

```bash
python games/evaluation/leaderboard/test_arena.py
```

Tests database operations, Elo calculations, and leaderboard generation.

## Files

- `run_arena.py`: Main entry point
- `arena_workflow.py`: Model assignment and game execution
- `leaderboard_db.py`: Thread-safe persistent storage
- `leaderboard.py`: Calculation and display utilities
- `rate_limiter.py`: API rate limiting
- `test_arena.py`: Test suite
