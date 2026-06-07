# ♟️ Reinforcement Learning for Chess — Deep Q-Network (DQN)

A complete implementation of a Deep Q-Network agent trained to play Chess via self-play, built with PyTorch and python-chess.

---

## 📁 Project Structure

```
chess-dqn/
├── chess_env.py       # Chess environment (state encoding, action mask, rewards)
├── dqn_agent.py       # Replay buffer, ResNet DQN model, DQN agent
├── train.py           # Training loop, self-play, evaluation CLI
├── requirements.txt   # Python dependencies
└── checkpoints/       # Auto-created during training
```

---

## 🧠 Algorithm Overview

| Component | Detail |
|---|---|
| Algorithm | Deep Q-Network (DQN) |
| State | (8 × 8 × 14) float32 tensor — 14 binary planes |
| Action space | 4,096 slots (64 × 64 from/to squares) |
| Network | Convolutional ResNet — 6 residual blocks, ~4.2M parameters |
| Exploration | ε-greedy with illegal-move masking |
| Stability | Experience replay + hard target-network updates |
| Training | Self-play (agent plays both White and Black) |

### State Encoding (14 planes)

| Planes | Meaning |
|---|---|
| 0 – 5 | White pieces: Pawn, Knight, Bishop, Rook, Queen, King |
| 6 – 11 | Black pieces: Pawn, Knight, Bishop, Rook, Queen, King |
| 12 | Side to move (all-1s = White, all-0s = Black) |
| 13 | En-passant square |

### Reward Function

| Event | Reward |
|---|---|
| Checkmate (win) | +1.0 |
| Checkmate (loss) | −1.0 |
| Draw (stalemate / repetition / 75-move / insufficient material) | 0.0 |
| Each step | −0.001 (encourages decisive play) |

---

## ⚙️ Installation

**Requirements:** Python 3.9+, pip

```bash
# Clone / download the project files
# Then install dependencies:
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
numpy>=1.24.0
python-chess>=1.10.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

GPU is optional but significantly speeds up training. PyTorch will auto-detect CUDA.

---

## 🚀 Usage

### Train from scratch

```bash
python train.py --episodes 5000
```

### Resume from a checkpoint

```bash
python train.py --episodes 10000 --checkpoint checkpoints/dqn_chess_ep5000.pt
```

### Evaluate agent against a random opponent

```bash
python train.py --mode eval --checkpoint dqn_chess_final.pt --eval_games 200
```

### Full list of CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `train` | `train` or `eval` |
| `--checkpoint` | _(none)_ | Path to `.pt` file to load |
| `--episodes` | `5000` | Number of self-play games |
| `--max_moves` | `200` | Max moves per game before draw |
| `--lr` | `1e-4` | Adam learning rate |
| `--gamma` | `0.99` | Discount factor |
| `--epsilon_start` | `1.0` | Initial exploration rate |
| `--epsilon_end` | `0.05` | Minimum exploration rate |
| `--epsilon_decay` | `0.9995` | Multiplicative decay per step |
| `--batch_size` | `64` | Replay buffer sample size |
| `--buffer_size` | `50000` | Replay buffer capacity |
| `--target_update` | `500` | Steps between target-net syncs |
| `--log_every` | `50` | Episodes between console logs |
| `--save_every` | `500` | Episodes between checkpoints |
| `--eval_games` | `100` | Games to play in eval mode |

---

## 📈 Expected Training Dynamics

| Phase | Episodes | Description |
|---|---|---|
| Random Play | 1 – 500 | High ε; diverse but noisy experience fills the buffer |
| Early Learning | 500 – 3,000 | Agent avoids one-move blunders; win rate vs. random climbs to ~70% |
| Refinement | 3,000+ | Basic tactics emerge; win rate reaches ~85–90% vs. random |

Training output example:
```
Ep   50 | Avg Reward: -0.012 | Avg Loss: 0.7823 | ε: 0.975 | Steps: 143 | W/D/L: 24/3/23
Ep  500 | Avg Reward: +0.104 | Avg Loss: 0.3241 | ε: 0.779 | Steps: 112 | W/D/L: 267/18/215
Ep 2000 | Avg Reward: +0.312 | Avg Loss: 0.1872 | ε: 0.368 | Steps:  87 | W/D/L: 1124/72/804
```

---

## 🏗️ Architecture

```
Input: (B, 8, 8, 14)
       ↓  permute → (B, 14, 8, 8)
Entry Conv2d(14→128, 3×3) + BatchNorm + ReLU
       ↓
ResBlock × 6  [Conv → BN → ReLU → Conv → BN → skip add → ReLU]
       ↓
Conv2d(128→32, 1×1) + BatchNorm + ReLU
       ↓
Flatten → Linear(2048→256) → ReLU → Linear(256→4096)
       ↓
Output: Q-values for all 4,096 (from, to) actions
```

Illegal moves are masked with −∞ before argmax — the agent always plays legal Chess.

---

## 🔭 Extending the Project

| Improvement | Benefit |
|---|---|
| **Double DQN** | Reduces Q-value overestimation |
| **Dueling DQN** | Separate value + advantage streams |
| **Prioritised Replay (PER)** | Learn faster from surprising transitions |
| **Monte Carlo Tree Search** | Look-ahead planning (AlphaZero style) |
| **Opponent pool** | Diverse, non-stationary opponents prevent strategy collapse |
| **Full AlphaZero encoding** | Castling rights, repetition planes, 50-move counter |
| **Transformer encoder** | Attention over 64 squares instead of convolutions |

---

## 📚 References

1. Mnih et al. (2015) — *Human-level control through deep reinforcement learning*. Nature.
2. Silver et al. (2017) — *Mastering Chess and Shogi by self-play with a general RL algorithm*. arXiv:1712.01815.
3. van Hasselt et al. (2016) — *Deep Reinforcement Learning with Double Q-learning*. AAAI.
4. Schaul et al. (2016) — *Prioritized Experience Replay*. ICLR.
5. [python-chess documentation](https://python-chess.readthedocs.io)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
