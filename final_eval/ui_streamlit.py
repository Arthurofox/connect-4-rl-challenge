from __future__ import annotations


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from inspect import signature
from typing import Any, Dict

import numpy as np
import streamlit as st

from src.board import ConnectFourBoard, InvalidMoveError

try:
    from .arena import fight
except ImportError:
    from final_eval.arena import fight

def _load_muzero_agent(checkpoint: str, device: str):
    from pathlib import Path as _Path
    from src.agents.muzero.agent import MuZeroAgent, resolve_device

    resolved_device = resolve_device(device)
    loader = getattr(MuZeroAgent, "load_from_checkpoint", None)
    if callable(loader):
        return loader(checkpoint, device=resolved_device)
    return MuZeroAgent.from_checkpoint(_Path(checkpoint), device=resolved_device)



def _board_to_numpy(board: ConnectFourBoard) -> np.ndarray:
    """Return a visualization grid with row 0 as the bottom row."""
    return board.render_grid_bottom_first()


def _legal_moves_mask(board: ConnectFourBoard) -> np.ndarray:
    mask = np.zeros(board.columns, dtype=bool)
    for col in board.valid_moves():
        mask[col] = True
    return mask


def _apply_move(board: ConnectFourBoard, column: int) -> dict[str, Any]:
    result = board.drop(column)
    winner = result.winner
    done = result.board_full or winner is not None
    return {"done": done, "winner": winner}


def render_board_svg(board: ConnectFourBoard) -> None:
    """Render the board with the bottom row at index 0."""
    G = _board_to_numpy(board)
    rows, cols = G.shape

    cell = 70
    pad = 14
    disc_radius = 26
    hole_radius = 28
    width = cols * cell + pad * 2
    height = rows * cell + pad * 2

    parts: list[str] = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
    ]
    parts += [
        '<defs>',
        '  <mask id="holes">',
        f'    <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'    <g transform="translate(0,{height}) scale(1,-1)">'
    ]
    for rr in range(rows):
        cy = pad + rr * cell + cell / 2
        for cc in range(cols):
            cx = pad + cc * cell + cell / 2
            parts.append(
                f'      <circle cx="{cx}" cy="{cy}" r="{hole_radius}" fill="black"/>'
            )
    parts += [
        '    </g>',
        '  </mask>',
        '</defs>',
        f'<g transform="translate(0,{height}) scale(1,-1)">'
    ]

    for rr in range(rows):
        cy = pad + rr * cell + cell / 2
        for cc in range(cols):
            value = int(G[rr, cc])
            if value == 0:
                continue
            cx = pad + cc * cell + cell / 2
            fill = '#ef4444' if value == 1 else '#facc15'
            parts.append(
                f'<circle cx="{cx}" cy="{cy}" r="{disc_radius}" fill="{fill}" stroke="#111827" stroke-width="1.5"/>'
            )

    parts.append(
        f'<rect x="{pad/2}" y="{pad/2}" width="{width - pad}" height="{height - pad}" rx="22" ry="22" fill="#1E3A8A" mask="url(#holes)" />'
    )
    parts.append(
        f'<rect x="{pad/2}" y="{pad/2}" width="{width - pad}" height="{height - pad}" rx="22" ry="22" fill="none" stroke="#0b1b5a" stroke-width="4"/>'
    )
    parts.append('</g>')
    parts.append('</svg>')

    svg = ''.join(parts)
    st.markdown(
        f'<div style="display:flex;justify-content:center">{svg}</div>',
        unsafe_allow_html=True,
    )



def _select_ai_move(agent: Any, board: ConnectFourBoard, sims: int) -> int:
    name = getattr(agent, "name", agent.__class__.__name__).lower()
    if name == "muzero":
        kwargs: Dict[str, Any] = {"training": False}
        params = signature(agent.select_action).parameters
        if "mcts_simulations" in params:
            kwargs["mcts_simulations"] = sims
        else:
            kwargs["simulations"] = sims
        return agent.select_action(board, **kwargs)
    try:
        return agent.select_action(board, training=False)
    except TypeError:
        return agent.select_action(board)



def main():
    st.set_page_config(page_title="Connect-4 Final Evaluation", page_icon="🎮", layout="centered")
    st.title("Connect-4 Final Evaluation")

    # ---- CLI args (leave as-is if you already parse) ----
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--muzero")
    parser.add_argument("--dqn")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sims", type=int, default=160)
    args, _ = parser.parse_known_args()

    # ---- Initialize session_state once ----
    if "board" not in st.session_state:
        st.session_state.board = ConnectFourBoard()
        st.session_state.msg = ""
        st.session_state.player_red = "human"        # P1
        st.session_state.player_yellow = "muzero"    # P2
        st.session_state.muzero_ckpt = args.muzero or "final_eval/Arthur_final_best.pt"
        st.session_state.dqn_ckpt    = args.dqn or "final_eval/Roumouz_final_best.pt"
        st.session_state.device      = args.device
        st.session_state.sims        = args.sims
        st.session_state.agents      = {}            # loaded agents cache

    s = st.session_state   # convenience alias

    # ---- Sidebar controls ----
    s.player_red    = st.sidebar.selectbox("P1 (🔴)", ["human","muzero","dqn","alphabeta"], index=["human","muzero","dqn","alphabeta"].index(s.player_red))
    s.player_yellow = st.sidebar.selectbox("P2 (🟡)", ["human","muzero","dqn","alphabeta"], index=["human","muzero","dqn","alphabeta"].index(s.player_yellow))
    s.device        = st.sidebar.selectbox("Device", ["cpu","mps"], index=["cpu","mps"].index(s.device) if s.device in ["cpu","mps"] else 0)
    s.sims          = st.sidebar.slider("MuZero simulations", 32, 512, int(s.sims), step=32)

    col1, col2 = st.sidebar.columns(2)
    s.muzero_ckpt = col1.text_input("MuZero checkpoint", s.muzero_ckpt)
    s.dqn_ckpt    = col2.text_input("DQN checkpoint", s.dqn_ckpt)

    if st.sidebar.button("Load/Reload agents"):
        agents: Dict[str, Any] = {}
        try:
            if "muzero" in (s.player_red, s.player_yellow):
                agents["muzero"] = _load_muzero_agent(s.muzero_ckpt, s.device)
            if "dqn" in (s.player_red, s.player_yellow):
                from final_eval.dqn_agent import DQNAgent

                agents["dqn"] = DQNAgent.load_from_checkpoint(s.dqn_ckpt, device=s.device)
            if "alphabeta" in (s.player_red, s.player_yellow):
                from src.agents.alphabeta import AlphaBetaAgent

                agents["alphabeta"] = AlphaBetaAgent(depth=7)
        except FileNotFoundError as exc:
            st.error(f"Checkpoint not found: {exc}")
        except AttributeError as exc:
            st.error(f"Failed to load agent: {exc}")
        else:
            s.agents = agents
            st.success("Agents ready.")

    # ---- Top controls ----
    c1, c2, c3 = st.columns(3)
    if c1.button("↩️ Reset"):
        s.board = ConnectFourBoard()
        s.msg = ""
    if c2.button("🤖 Next AI Move"):
        if not s.agents:
            st.warning("Load agents first.")
        else:
            pl = s.board.to_move
            who = s.player_red if pl == 1 else s.player_yellow
            if who == "human":
                st.info("It's human's turn.")
            else:
                agent = s.agents[who]
                sims_value = s.sims if getattr(agent, "name", "") == "muzero" else 0
                move = _select_ai_move(agent, s.board, sims_value)
                outcome = _apply_move(s.board, move)
                if outcome["done"]:
                    winner = outcome["winner"]
                    s.msg = "Draw." if winner is None else ("🔴 wins!" if winner == 1 else "🟡 wins!")
                else:
                    s.msg = ""
    auto_n = c3.number_input("Auto games", 10, 1000, 100, step=10)
    if st.button("🏟️ Auto-play (swap first)"):
        if not s.agents:
            st.warning("Load agents first.")
        else:
            from final_eval.arena import fight
            a_map = s.agents
            get = lambda role: a_map.get(role) if role in a_map else None
            a1 = get(s.player_red); a2 = get(s.player_yellow)
            if a1 is None or a2 is None:
                st.warning("Both sides must be AI for auto-play.")
            else:
                res = fight(a1, a2, games=int(auto_n), swap_first=True, sims1=s.sims, sims2=s.sims)
                st.success(f"Wins 🔴: {res['wins_agent1']} | Wins 🟡: {res['wins_agent2']} | Draws: {res['draws']}")

    st.subheader("Make a move")
    cols = st.columns(7)
    legal = _legal_moves_mask(s.board)
    move_applied = False
    for i in range(7):
        disabled = not bool(legal[i])
        if cols[i].button(f"Col {i}", disabled=disabled):
            try:
                result = _apply_move(s.board, i)
            except InvalidMoveError:
                st.warning("Illegal move.")
            else:
                winner = result["winner"]
                if result["done"]:
                    s.msg = "Draw." if winner is None else ("🔴 wins!" if winner == 1 else "🟡 wins!")
                else:
                    s.msg = ""
                move_applied = True

    st.subheader("Board")
    render_board_svg(s.board)
    st.caption("Columns: 0 1 2 3 4 5 6")

    if s.msg:
        st.success(s.msg)


if __name__ == "__main__":
    main()