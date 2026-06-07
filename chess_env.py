"""
Chess Environment Wrapper for Reinforcement Learning
Uses python-chess for game logic; exposes OpenAI Gym-like interface.
"""

import chess
import numpy as np
from typing import Tuple, List, Optional


class ChessEnv:
    """
    Chess environment compatible with DQN training.

    State:  (8, 8, 14) float32 tensor
            Planes 0-5:  White pieces  (P, N, B, R, Q, K)
            Planes 6-11: Black pieces  (P, N, B, R, Q, K)
            Plane 12:    Side to move  (all 1s = White, all 0s = Black)
            Plane 13:    En-passant square (1 at ep square, else 0)

    Action: Integer index into a flat list of all 4672 possible UCI moves
            (from_sq * 64 + to_sq * ... simplified to from_sq*64 + to_sq).
            Illegal moves are masked out before action selection.

    Reward:
        +1   Win
        -1   Loss
         0   Draw / ongoing
        -0.001 per step (encourages shorter games)
    """

    PIECE_TO_PLANE = {
        chess.PAWN:   0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK:   3,
        chess.QUEEN:  4,
        chess.KING:   5,
    }

    def __init__(self, max_moves: int = 200):
        self.board = chess.Board()
        self.max_moves = max_moves
        self.move_count = 0
        # Build full action space (from_sq 0-63, to_sq 0-63) = 4096 slots
        # We ignore promotions for simplicity (always promote to queen)
        self.action_size = 64 * 64

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.board = chess.Board()
        self.move_count = 0
        return self._get_state()

    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        from_sq = action // 64
        to_sq   = action  % 64

        # Build move (auto-queen promotion)
        move = chess.Move(from_sq, to_sq)
        promo = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)

        if promo in self.board.legal_moves:
            move = promo
        elif move not in self.board.legal_moves:
            # Illegal move penalty – agent should never reach here if masking used
            return self._get_state(), -1.0, True, {"reason": "illegal"}

        self.board.push(move)
        self.move_count += 1

        reward, done = self._evaluate()
        return self._get_state(), reward, done, {}

    # ------------------------------------------------------------------
    def legal_action_mask(self) -> np.ndarray:
        """Returns a boolean mask over the 4096 action space."""
        mask = np.zeros(self.action_size, dtype=bool)
        for move in self.board.legal_moves:
            mask[move.from_square * 64 + move.to_square] = True
        return mask

    # ------------------------------------------------------------------
    def _evaluate(self) -> Tuple[float, bool]:
        if self.board.is_checkmate():
            # The side that just moved wins
            reward = 1.0 if self.board.turn == chess.BLACK else -1.0
            return reward, True
        if (self.board.is_stalemate() or
                self.board.is_insufficient_material() or
                self.board.is_seventyfive_moves() or
                self.board.is_fivefold_repetition()):
            return 0.0, True
        if self.move_count >= self.max_moves:
            return 0.0, True
        return -0.001, False   # small step penalty

    # ------------------------------------------------------------------
    def _get_state(self) -> np.ndarray:
        """Encode board as (14, 8, 8) then transpose to (8, 8, 14)."""
        planes = np.zeros((14, 8, 8), dtype=np.float32)

        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece is None:
                continue
            rank, file = divmod(sq, 8)
            plane = self.PIECE_TO_PLANE[piece.piece_type]
            if piece.color == chess.BLACK:
                plane += 6
            planes[plane, rank, file] = 1.0

        # Side to move
        if self.board.turn == chess.WHITE:
            planes[12] = 1.0

        # En passant
        if self.board.ep_square is not None:
            ep_rank, ep_file = divmod(self.board.ep_square, 8)
            planes[13, ep_rank, ep_file] = 1.0

        return planes.transpose(1, 2, 0)   # (8, 8, 14)

    # ------------------------------------------------------------------
    @property
    def state_shape(self) -> Tuple:
        return (8, 8, 14)
