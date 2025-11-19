import os
import json
import time
import re
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

import torch
import torch.nn as nn
import ccxt

import plotly.graph_objects as go
import plotly.express as px

from stable_baselines3 import PPO

# Project Imports
from src.data.loader import load_all_symbols
from src.data.features import add_features, align_all_assets
from src.envs.microstruct_env import MicrostructureEnv

from src.utils.metrics import evaluate_on_env

from src.agents.ppo_runner import (
    _select_first_n_assets,
    DEFAULT_WINDOW as PPO_WINDOW
)

from src.qrl.qrl_3assets_runner import (
    QACAgent as QAC3,
    QRLModelWrapper,
    build_train_val_envs as build_qrl_train_val_envs,
    DEFAULT_N_QUBITS,
    DEFAULT_N_LAYERS,
    DEFAULT_WINDOW as QRL_WINDOW,
)

# PATHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PPO_MODEL_PATH = os.path.join(BASE_DIR, "models", "ppo_20251117_092420", "best_model.zip")
QRL_MODEL_PATH = os.path.join(BASE_DIR, "models", "qrl_3asset_20251117_012801", "final_qrl_agent.pt")

PPO_RESULTS_DIR = os.path.join(BASE_DIR, "results", "ppo_20251117_092420")
QRL_RESULTS_DIR = os.path.join(BASE_DIR, "results", "qrl_3asset_20251117_012801")

PPO_FINAL_JSON = os.path.join(PPO_RESULTS_DIR, "final_metrics.json")
PPO_QRL_COMPARISON_CSV = os.path.join(BASE_DIR, "results", "ppo_vs_qrl.csv")
PPO_HISTORY_CSV = os.path.join(PPO_RESULTS_DIR, "history.csv")

FINETUNED_QRL_PATH = os.path.join(BASE_DIR, "models", "qrl_3asset_finetuned_live.pt")

LOGS_DIR = os.path.join(BASE_DIR, "logs")

# HELPERS

def _format_hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_latest_qrl_log_path() -> str | None:
    if not os.path.isdir(LOGS_DIR):
        return None
    candidates = [
        f for f in os.listdir(LOGS_DIR)
        if f.startswith("qrl_3asset_") and f.endswith(".log")
    ]
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(LOGS_DIR, candidates[-1])


def parse_qrl_log_episode_rewards(log_path: str) -> pd.DataFrame:
    episodes = []
    steps_list = []
    rewards = []

    if (log_path is None) or (not os.path.exists(log_path)):
        return pd.DataFrame(columns=["episode", "steps", "reward"])

    ep_pattern = re.compile(r"^\[EP\s+(\d+)\]")
    r_pattern = re.compile(r"R=([\-0-9\.]+)")
    steps_pattern = re.compile(r"steps=([\d,]+)")

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m_ep = ep_pattern.match(line)
            if not m_ep:
                continue

            try:
                ep = int(m_ep.group(1))
            except:
                continue

            m_r = r_pattern.search(line)
            m_s = steps_pattern.search(line)

            if m_r is None or m_s is None:
                continue

            try:
                reward = float(m_r.group(1))
                steps = int(m_s.group(1).replace(",", ""))
            except:
                continue

            episodes.append(ep)
            steps_list.append(steps)
            rewards.append(reward)

    return pd.DataFrame({
        "episode": episodes,
        "steps": steps_list,
        "reward": rewards
    })

# CACHE LOADERS

@st.cache_resource
def load_ppo_model() -> PPO:
    return PPO.load(PPO_MODEL_PATH, device="cpu")


@st.cache_resource
def load_qrl_agent_and_env():
    train_env, val_env, train_scaled = build_qrl_train_val_envs(
        window=QRL_WINDOW,
        split_ratio=0.8,
        seed=42,
        target_symbols=None,
    )

    action_dim = len(train_scaled)

    agent = QAC3(
        observation_space=train_env.observation_space,
        action_dim=action_dim,
        n_qubits=DEFAULT_N_QUBITS,
        n_layers=DEFAULT_N_LAYERS,
        lr=1e-4,
    )

    state_dict = torch.load(QRL_MODEL_PATH, map_location="cpu")
    agent.load_state_dict(state_dict)

    wrapper = QRLModelWrapper(agent, torch.device("cpu"))

    return agent, wrapper, train_env, val_env

# LIVE UTILITY FUNCTIONS

def reinit_actor_for_new_assets(agent: QAC3, n_assets: int):
    agent.actor = nn.Sequential(
        nn.Linear(agent.actor[0].in_features, n_assets),
        nn.Softplus(),
    )
    return agent


def get_binance_client():
    return ccxt.binance({"enableRateLimit": True})


def fetch_live_ohlcv(symbol: str, limit: int = 300):
    client = get_binance_client()
    data = client.fetch_ohlcv(symbol, "1h", limit=limit)
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def prepare_live_feature_dict(symbols: List[str], window: int):
    raw_dict = {}
    limit = max(window + 200, 300)

    for sym in symbols:
        df = fetch_live_ohlcv(sym, limit)
        feat = add_features(df)
        raw_dict[sym] = feat

    return align_all_assets(raw_dict)

# QRL FINE-TUNING FUNCTION

def finetune_qrl_on_live(
    agent,
    env,
    episodes: int = 25,
    max_steps: int = 400,
    gamma: float = 0.995,
):

    device = torch.device("cpu")
    agent.to(device)

    progress = st.progress(0)
    status_box = st.empty()
    reward_plot = st.empty()
    qubit_map = st.empty()
    alloc_plot = st.empty()
    entropy_plot = st.empty()
    eff_plot = st.empty()
    grad_plot = st.empty()
    speed_box = st.empty()

    episode_rewards = []
    alloc_history = []
    qubit_history = []
    entropy_history_ep = []
    efficiency_history = []
    grad_update_norms = []

    best_reward = -1e18
    start_time = time.time()

    for ep in range(episodes):
        ep_start = time.time()

        obs, _ = env.reset()
        traj_logp, traj_val, traj_rew = [], [], []
        done = False
        step = 0
        ep_reward = 0.0

        ep_allocs, ep_entropies = [], []

        while not done and step < max_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            action, logp, value = agent.act(obs_t)
            alloc_vec = action.detach().cpu().numpy().flatten()

            ep_allocs.append(alloc_vec)

            w = np.clip(alloc_vec, 1e-12, 1.0)
            w /= w.sum()
            ep_entropies.append(float(-(w * np.log(w)).sum()))

            with torch.no_grad():
                q_out = agent._encode(obs_t).squeeze(0).cpu().numpy()
            qubit_history.append(q_out)

            nxt, reward, terminated, truncated, _ = env.step(alloc_vec)
            done = terminated or truncated

            traj_logp.append(logp.squeeze())
            traj_val.append(value.squeeze())
            traj_rew.append(float(reward))

            obs = nxt
            step += 1
            ep_reward += float(reward)

        before = np.concatenate([p.detach().cpu().numpy().ravel() for p in agent.vqc.parameters()])
        agent.update({"logp": traj_logp, "value": traj_val, "reward": traj_rew}, gamma)
        after = np.concatenate([p.detach().cpu().numpy().ravel() for p in agent.vqc.parameters()])
        grad_update_norms.append(float(np.linalg.norm(after - before)))

        episode_rewards.append(ep_reward)
        alloc_history.append(np.array(ep_allocs))
        entropy_history_ep.append(ep_entropies)

        vol = float(np.std(traj_rew) + 1e-12)
        efficiency_history.append(float(ep_reward / vol))

        if ep_reward > best_reward:
            best_reward = ep_reward
            os.makedirs(os.path.dirname(FINETUNED_QRL_PATH), exist_ok=True)
            torch.save(agent.state_dict(), FINETUNED_QRL_PATH)

        pct = int((ep + 1) / episodes * 100)
        progress.progress(pct)

        elapsed = time.time() - start_time
        eta = (time.time() - ep_start) * (episodes - ep - 1)

        status_box.markdown(
            f"""
            ### Fine-tuning Episode {ep+1}/{episodes}
            **Episode Reward:** `{ep_reward:.3f}`  
            **Steps:** `{step}`  
            **Elapsed:** `{_format_hms(elapsed)}`  
            **ETA:** `{_format_hms(eta)}`  
            """
        )

        reward_plot.plotly_chart(
            px.line(y=episode_rewards, title="Fine-tuning Episode Rewards (Y=Reward, X=Episode)"),
            use_container_width=True
        )

        if len(qubit_history) >= 20:
            qmat = np.stack(qubit_history[-80:], axis=0)
            df_q = pd.DataFrame(qmat, columns=[f"q{i}" for i in range(qmat.shape[1])])
            corr = df_q.corr().values
            ent = np.abs(corr - np.eye(corr.shape[0]))
            ent_strength = float(ent.sum() / (ent.size - ent.shape[0]))

            qmap_fig = px.imshow(
                df_q.T,
                color_continuous_scale="Viridis",
                title=f"Qubit Activation Map — Entanglement Proxy: {ent_strength:.3f}"
            )
            qubit_map.plotly_chart(qmap_fig, use_container_width=True)

        drift_df = pd.DataFrame(
            alloc_history[-1],
            columns=[f"A{i}" for i in range(alloc_history[-1].shape[1])]
        )
        alloc_plot.plotly_chart(px.line(drift_df, title="Allocation Drift"), use_container_width=True)

        entropy_plot.plotly_chart(
            px.line(y=entropy_history_ep[-1], title="Action Entropy"), use_container_width=True
        )

        eff_plot.plotly_chart(
            px.line(y=efficiency_history, title="Reward-to-Volatility Efficiency"),
            use_container_width=True
        )

        grad_plot.plotly_chart(
            px.line(y=grad_update_norms, title="Quantum Circuit Update Norm"),
            use_container_width=True
        )

        if len(episode_rewards) > 2:
            score = float(
                0.5 * np.tanh((episode_rewards[-1] - episode_rewards[0]) / 1000.0)
                + 0.3 * np.tanh(efficiency_history[-1] / 100.0)
                + 0.2 * np.tanh(grad_update_norms[-1])
            )
            speed_box.success(f"⚡ QRL Learning Speed Score: **{score:.3f}**")

    status_box.success("Fine-tuning complete. Best model saved.")
    return agent

# STREAMLIT UI

st.set_page_config(page_title="Quantum RL Portfolio Manager", layout="wide")
st.title("Quantum Reinforcement Learning for Crypto Portfolio Management")

tab_hist, tab_live = st.tabs(["Historical Backtest", "Live Allocation"])

# TAB 1 — HISTORICAL (PPO + QRL + TRAINING CURVES)
with tab_hist:

    st.header("Historical Model Comparison (PPO vs QRL)")

    assets = sorted(load_all_symbols().keys())
    sel_asset = st.selectbox("Choose Asset", assets)

    df = load_all_symbols()[sel_asset]

    st.subheader(f"{sel_asset} — OHLCV (last 200 rows)")
    st.dataframe(df.tail(200), use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Close"))
    if "rsi_14" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["rsi_14"], name="RSI"))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.header("PPO Evaluation")
        if os.path.exists(PPO_FINAL_JSON):
            m = json.load(open(PPO_FINAL_JSON))
            st.metric("Cumulative Return", f"{m['cumulative_return']*100:.2f}%")
            st.metric("Sharpe Ratio", f"{m['sharpe']:.3f}")
        else:
            st.info("PPO final_metrics.json not found.")

    with col2:
        st.header("QRL Evaluation")
        qrl_final_json = os.path.join(QRL_RESULTS_DIR, "final.json")
        if os.path.exists(qrl_final_json):
            m = json.load(open(qrl_final_json))
            st.metric("Cumulative Return", f"{m['cumulative_return']*100:.2f}%")
            st.metric("Sharpe Ratio", f"{m['sharpe']:.3f}")
        else:
            st.info("QRL final.json not found.")

    st.header("Training Progress")

    ppo_col, qrl_col = st.columns(2)

    with ppo_col:
        st.subheader("PPO Training Curve")

        if os.path.exists(PPO_HISTORY_CSV):
            hist_df = pd.read_csv(PPO_HISTORY_CSV)

            x = hist_df["step"] if "step" in hist_df.columns else hist_df.index

            ppo_fig = go.Figure()

            if "cumulative_return" in hist_df.columns:
                ppo_fig.add_trace(go.Scatter(
                    x=x,
                    y=hist_df["cumulative_return"],
                    name="Cumulative Return"
                ))

            if "sharpe" in hist_df.columns:
                ppo_fig.add_trace(go.Scatter(
                    x=x,
                    y=hist_df["sharpe"],
                    name="Sharpe Ratio",
                    yaxis="y2"
                ))
                ppo_fig.update_layout(
                    yaxis=dict(title="Cumulative Return"),
                    yaxis2=dict(title="Sharpe", overlaying="y", side="right"),
                    xaxis_title="Step"
                )
            else:
                ppo_fig.update_layout(xaxis_title="Step", yaxis_title="Metric")

            st.plotly_chart(ppo_fig, use_container_width=True)

        else:
            st.info("PPO history.csv not found.")

    with qrl_col:
        st.subheader("QRL Episode Reward Curve")

        qrl_log_path = None
        if os.path.isdir(LOGS_DIR):
            for f in sorted(os.listdir(LOGS_DIR)):
                if f.startswith("qrl_3asset") and f.endswith(".log"):
                    qrl_log_path = os.path.join(LOGS_DIR, f)

        if qrl_log_path is None:
            st.info("No QRL log file found.")
        else:
            df_ep = parse_qrl_log_episode_rewards(qrl_log_path)

            if df_ep.empty:
                st.info("Could not parse episode rewards from QRL log.")
            else:
                st.plotly_chart(
                    px.line(
                        df_ep,
                        x="episode",
                        y="reward",
                        title=f"QRL Episode Rewards (log: {os.path.basename(qrl_log_path)})"
                    ),
                    use_container_width=True
                )

        st.subheader("QRL Training — Cumulative Return & Sharpe")

        val_files = sorted([
            f for f in os.listdir(QRL_RESULTS_DIR)
            if f.startswith("val_") and f.endswith(".json")
        ])

        if len(val_files) == 0:
            st.info("No QRL validation snapshots found.")
        else:
            vals = []
            for f in val_files:
                step = int(f.replace("val_", "").replace(".json", ""))
                data = json.load(open(os.path.join(QRL_RESULTS_DIR, f)))
                vals.append({
                    "step": step,
                    "cumulative_return": data.get("cumulative_return", None),
                    "sharpe": data.get("sharpe", None)
                })

            df_val = pd.DataFrame(vals).sort_values("step")

            fig_qrl = go.Figure()
            fig_qrl.add_trace(go.Scatter(
                x=df_val["step"],
                y=df_val["cumulative_return"],
                name="Cumulative Return"
            ))
            fig_qrl.add_trace(go.Scatter(
                x=df_val["step"],
                y=df_val["sharpe"],
                name="Sharpe Ratio",
                yaxis="y2"
            ))
            fig_qrl.update_layout(
                title="QRL Training Performance",
                xaxis_title="Training Steps",
                yaxis=dict(title="Cumulative Return"),
                yaxis2=dict(title="Sharpe", overlaying="y", side="right")
            )
            st.plotly_chart(fig_qrl, use_container_width=True)

    st.header("PPO vs QRL Summary")

    if os.path.exists(PPO_QRL_COMPARISON_CSV):
        st.dataframe(
            pd.read_csv(PPO_QRL_COMPARISON_CSV),
            use_container_width=True
        )
    else:
        st.info("Comparison CSV not found.")

# TAB 2 — LIVE ALLOCATION

with tab_live:

    st.header("Live 3-Asset Allocation (QRL Only)")

    colA, colB, colC = st.columns(3)
    a1 = colA.text_input("Asset 1", "BTC/USDT")
    a2 = colB.text_input("Asset 2", "ETH/USDT")
    a3 = colC.text_input("Asset 3", "BNB/USDT")

    capital = st.number_input("Total Capital (USDT)", value=1000.0)

    st.subheader("QRL Fine-Tuning Settings")
    do_ft = st.checkbox("Fine-tune QRL on these live assets?", value=True)
    ft_episodes = st.slider("Fine-tune episodes", 5, 100, 25)
    ft_steps = st.slider("Max steps per episode", 100, 2000, 400)

    if st.button("Run QRL Live Allocation"):

        try:
            st.info("Fetching live data...")
            data_dict = prepare_live_feature_dict([a1, a2, a3], window=QRL_WINDOW)
            st.success("Live data ready.")

            agent, wrapper, train_env, _ = load_qrl_agent_and_env()

            # reinitialize actor head for new assets
            agent = reinit_actor_for_new_assets(agent, 3)

            # >>> FIX: rebuild optimizer so new actor parameters are trainable
            if hasattr(agent, "_build_optimizer"):
                agent._build_optimizer()

            live_env = MicrostructureEnv(data_dict=data_dict, window=QRL_WINDOW)

            if do_ft:
                st.subheader("QRL Fine-Tuning Dashboard")
                agent = finetune_qrl_on_live(
                    agent,
                    live_env,
                    episodes=ft_episodes,
                    max_steps=ft_steps,
                )

            obs, _ = live_env.reset()
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_w = agent.act(obs_t)[0].squeeze(0).detach().numpy()

            q_w = np.clip(q_w, 0, 1)
            q_w /= q_w.sum()

            q_alloc = (q_w * capital).tolist()

            st.subheader("Final QRL Allocation")
            st.dataframe(pd.DataFrame({
                "asset": [a1, a2, a3],
                "weight": q_w,
                "amount_usdt": q_alloc
            }))

            if os.path.exists(FINETUNED_QRL_PATH):
                with open(FINETUNED_QRL_PATH, "rb") as f:
                    st.download_button(
                        "Download Fine-Tuned QRL Model",
                        data=f,
                        file_name="qrl_3asset_finetuned_live.pt",
                        mime="application/octet-stream"
                    )

        except Exception as e:
            st.error(f"Live QRL Allocation Failed: {e}")
