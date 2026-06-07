"""
Training & Evaluation Script for Chess DQN Agent
Run:   python train.py --episodes 5000 --eval_every 500
"""

import argparse
import time
import json
import os
import random
import numpy as np
from collections import deque

from chess_env import ChessEnv
from dqn_agent import DQNAgent


# ──────────────────────────────────────────────────────────────────────────────
# Self-Play Training
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    env   = ChessEnv(max_moves=args.max_moves)
    agent = DQNAgent(
        state_shape    = env.state_shape,
        action_size    = env.action_size,
        lr             = args.lr,
        gamma          = args.gamma,
        epsilon_start  = args.epsilon_start,
        epsilon_end    = args.epsilon_end,
        epsilon_decay  = args.epsilon_decay,
        batch_size     = args.batch_size,
        buffer_size    = args.buffer_size,
        target_update  = args.target_update,
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)

    history = {
        "episode":      [],
        "reward":       [],
        "loss":         [],
        "epsilon":      [],
        "game_length":  [],
        "wins":         0,
        "draws":        0,
        "losses":       0,
    }

    recent_rewards = deque(maxlen=100)
    recent_losses  = deque(maxlen=100)

    print(f"Training on device: {agent.device}")
    print(f"Episodes: {args.episodes}  |  Batch: {args.batch_size}  "
          f"|  Buffer: {args.buffer_size}\n")

    for ep in range(1, args.episodes + 1):
        state = env.reset()
        total_reward = 0.0
        losses       = []
        step         = 0

        while True:
            mask   = env.legal_action_mask()
            action = agent.select_action(state, mask)

            next_state, reward, done, info = env.step(action)
            next_mask = env.legal_action_mask() if not done else np.zeros(env.action_size, dtype=bool)

            agent.store(state, action, reward, next_state, done, next_mask)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            state         = next_state
            total_reward += reward
            step         += 1

            if done:
                break

        # Track outcomes
        result = info.get("reason", "")
        if total_reward > 0.5:
            history["wins"] += 1
        elif total_reward < -0.5:
            history["losses"] += 1
        else:
            history["draws"] += 1

        avg_loss = float(np.mean(losses)) if losses else 0.0
        recent_rewards.append(total_reward)
        recent_losses.append(avg_loss)

        history["episode"].append(ep)
        history["reward"].append(total_reward)
        history["loss"].append(avg_loss)
        history["epsilon"].append(agent.epsilon)
        history["game_length"].append(step)

        # Logging
        if ep % args.log_every == 0:
            print(f"Ep {ep:5d} | "
                  f"Avg Reward: {np.mean(recent_rewards):+.3f} | "
                  f"Avg Loss: {np.mean(recent_losses):.4f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Steps: {step:3d} | "
                  f"W/D/L: {history['wins']}/{history['draws']}/{history['losses']}")

        # Checkpoint
        if ep % args.save_every == 0:
            agent.save(f"checkpoints/dqn_chess_ep{ep}.pt")
            with open("training_history.json", "w") as f:
                json.dump({k: v for k, v in history.items()
                           if isinstance(v, list)}, f)

    agent.save("dqn_chess_final.pt")
    print("\nTraining complete. Final model saved.")
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation: Agent vs Random
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(agent_path: str, num_games: int = 100):
    import chess

    env   = ChessEnv(max_moves=200)
    agent = DQNAgent(state_shape=env.state_shape, action_size=env.action_size)
    agent.load(agent_path)
    agent.epsilon = 0.0   # greedy

    wins = draws = losses = 0

    for _ in range(num_games):
        state = env.reset()
        done  = False
        agent_color = chess.WHITE   # agent always plays White

        while not done:
            if env.board.turn == agent_color:
                mask   = env.legal_action_mask()
                action = agent.select_action(state, mask)
            else:
                # Random opponent
                legal = list(env.board.legal_moves)
                move  = random.choice(legal)
                action = move.from_square * 64 + move.to_square

            state, reward, done, _ = env.step(action)

        if env.board.is_checkmate():
            if env.board.turn != agent_color:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1

    print(f"\nEvaluation over {num_games} games (Agent=White vs Random):")
    print(f"  Wins:   {wins}  ({wins/num_games*100:.1f}%)")
    print(f"  Draws:  {draws} ({draws/num_games*100:.1f}%)")
    print(f"  Losses: {losses}({losses/num_games*100:.1f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess DQN Training")

    parser.add_argument("--mode",          type=str,   default="train",
                        choices=["train", "eval"])
    parser.add_argument("--checkpoint",    type=str,   default="",
                        help="Path to existing checkpoint to resume from")
    parser.add_argument("--episodes",      type=int,   default=5000)
    parser.add_argument("--max_moves",     type=int,   default=200)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--gamma",         type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end",   type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=0.9995)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--buffer_size",   type=int,   default=50_000)
    parser.add_argument("--target_update", type=int,   default=500)
    parser.add_argument("--log_every",     type=int,   default=50)
    parser.add_argument("--save_every",    type=int,   default=500)
    parser.add_argument("--eval_games",    type=int,   default=100)

    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        if not args.checkpoint:
            print("Provide --checkpoint path for eval mode.")
        else:
            evaluate(args.checkpoint, args.eval_games)
