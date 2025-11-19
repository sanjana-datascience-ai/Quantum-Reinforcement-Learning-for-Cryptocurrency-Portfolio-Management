"""
MicrostructureEnv
"""

from typing import Dict, List, Tuple, Optional, Any
import math

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from .execution import simulate_execution_lob, DEFAULTS as EXEC_DEFAULTS


class MicrostructureEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        window: int = 24 * 7,
        initial_balance: float = 100_000.0,
        max_participation: float = 0.05,
        taker_fee: float = 0.001,
        maker_fee: float = 0.0002,
        spread_floor: float = 0.0002,
        min_trade_notional: float = 10.0,
        reward_params: Optional[Dict] = None,
        risk_window: int = 24 * 7,
        max_drawdown: Optional[float] = None,
        rebalance_freq: int = 1,
        verbose: bool = False,
        exec_params: Optional[Dict] = None,
        risk_free_rate_annual: float = 0.0,
        steps_per_year: int = 24 * 365,
    ):
        """
        data_dict: dict[symbol] -> DataFrame with:
          - timestamp (datetime-like)
          - OHLCV columns
          - feature columns
        """
        super().__init__()

        if not isinstance(data_dict, dict) or len(data_dict) == 0:
            raise ValueError("MicrostructureEnv requires a non-empty dict of {symbol: DataFrame}.")

        self.raw_data = data_dict
        self.assets = list(data_dict.keys())
        self.n_assets = len(self.assets)

        self.window = int(window)
        self.initial_balance = float(initial_balance)
        self.max_participation = float(max_participation)
        self.taker_fee = float(taker_fee)
        self.maker_fee = float(maker_fee)
        self.spread_floor = float(spread_floor)
        self.min_trade_notional = float(min_trade_notional)
        self.reward_params = reward_params or {"lambda_vol": 0.0, "eta_turn": 0.1, "gamma_cvar": 0.0}
        self.risk_window = int(risk_window)
        self.max_drawdown = float(max_drawdown) if max_drawdown is not None else None
        self.rebalance_freq = int(rebalance_freq)
        self.verbose = bool(verbose)

        self.risk_free_rate_annual = float(risk_free_rate_annual)
        self.steps_per_year = int(steps_per_year)
        if self.steps_per_year <= 0:
            self.steps_per_year = 24 * 365

        # local RNG (do not rely on global np.random)
        self.rng = np.random.default_rng()

        # execution params (LOB simulator)
        self.exec_params = EXEC_DEFAULTS.copy()
        if exec_params:
            self.exec_params.update(exec_params)

        # align and precompute feature/price tensors
        self._align_by_intersection()

        # features: (T, n_assets, n_features)
        self.n_features = self.features.shape[2]

        # observation includes feature channels + 1 allocation channel per asset
        obs_shape = (self.n_assets, self.n_features + 1, self.window)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32,
        )

        # action = target allocation per asset (long-only, sum<=1; remainder is cash)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        # internal state
        self.t = None
        self.balance = None
        self.holdings = None
        self.nav = None
        self.prev_nav = None
        self.episode_pvs: List[float] = []
        self.trade_ledger: List[Dict[str, Any]] = []
        self.cum_turnover = 0.0
        self.steps = 0
        self.enable_trade_logging = False

    # -----------------------
    # INTERNAL DATA ALIGNMENT
    # -----------------------

    def _align_by_intersection(self):
        dfs: Dict[str, pd.DataFrame] = {}

        for s, df in self.raw_data.items():
            if "timestamp" not in df.columns:
                raise ValueError(f"[MicrostructureEnv] {s} missing 'timestamp' column.")

            d = df.copy()
            d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
            d = d.dropna(subset=["timestamp"]).set_index("timestamp")

            required = {"open", "high", "low", "close", "volume"}
            if not required.issubset(d.columns):
                raise ValueError(f"[MicrostructureEnv] {s} missing OHLCV columns {required}.")

            dfs[s] = d

        common_idx = None
        for d in dfs.values():
            common_idx = d.index if common_idx is None else common_idx.intersection(d.index)

        if common_idx is None or len(common_idx) == 0:
            raise RuntimeError("[MicrostructureEnv] No common timestamps across assets.")

        common_idx = common_idx.sort_values()

        price_list, vol_list, feat_list = [], [], []
        for s in self.assets:
            d = dfs[s].loc[common_idx]

            price_list.append(d["close"].values.astype(float))
            vol_list.append(d["volume"].values.astype(float))

            feat_cols = [c for c in d.columns if c not in ["open", "high", "low", "close", "volume", "symbol"]]
            if len(feat_cols) == 0:
                feat_arr = d[["close"]].values.astype(float)
            else:
                feat_arr = d[feat_cols].values.astype(float)

            feat_list.append(feat_arr)

        prices = np.vstack(price_list).T.astype(np.float32)  # (T, n_assets)
        volumes = np.vstack(vol_list).T.astype(np.float32)   # (T, n_assets)

        feat_dims = {arr.shape[1] for arr in feat_list}
        if len(feat_dims) != 1:
            raise RuntimeError(f"[MicrostructureEnv] Feature dimension mismatch across assets: {feat_dims}")

        features = np.stack(feat_list, axis=1).astype(np.float32)  # (T, n_assets, n_features)

        T = prices.shape[0]
        safe_prices = np.clip(prices, 1e-12, None)

        # approximate log returns for vol estimation
        log_rets = np.zeros_like(safe_prices)
        log_rets[1:] = np.log(safe_prices[1:] / safe_prices[:-1] + 1e-12)

        hourly_vol = np.zeros_like(prices)
        for t in range(T):
            if t == 0:
                hourly_vol[t, :] = 1e-6
            else:
                lb = min(24, t)
                slice_rets = log_rets[max(0, t - lb):t, :]
                hourly_vol[t, :] = np.std(slice_rets, axis=0) + 1e-12 if slice_rets.size else 1e-6

        hourly_vol = hourly_vol.astype(np.float32)
        self.timestamps = np.array(common_idx)
        self.prices = prices.astype(np.float32, copy=False)
        self.features = features.astype(np.float32, copy=False)
        self.volumes = volumes.astype(np.float32, copy=False)
        self.hourly_vol = hourly_vol.astype(np.float32, copy=False)
        self.T = T

    # -----------------------
    # GYM API
    # -----------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        if self.T <= self.window + 2:
            raise RuntimeError(
                f"[MicrostructureEnv] Not enough data for window={self.window}, T={self.T}."
            )

        self.t = self.window
        self.balance = float(self.initial_balance)
        self.holdings = np.zeros(self.n_assets, dtype=float)
        self.nav = float(self.initial_balance)
        self.prev_nav = float(self.initial_balance)
        self.episode_pvs = [float(self.nav)]
        self.trade_ledger = []
        self.cum_turnover = 0.0
        self.steps = 0

        obs = self._get_observation()
        info = {}
        return obs, info

    def _get_observation(self):
        start = self.t - self.window
        end = self.t

        feat_slice = self.features[start:end].transpose(1, 2, 0)  # (n_assets, n_features, window)

        last_prices = self.prices[self.t - 1]
        nav = self.nav if self.nav > 0 else (self.initial_balance + 1e-12)
        alloc = (self.holdings * last_prices) / (nav + 1e-12)
        alloc = np.clip(alloc, 0.0, 1.0)
        alloc_chan = np.repeat(alloc.reshape(self.n_assets, 1, 1), feat_slice.shape[2], axis=2).astype(np.float32, copy=False)

        obs = np.concatenate([feat_slice, alloc_chan], axis=1)
        # Avoid creating another copy unless needed
        if obs.dtype != np.float32:
            obs = obs.astype(np.float32, copy=False)
        return obs


    # -----------------------
    # STEP LOGIC
    # -----------------------

    def _apply_risk_free_yield(self):
        if self.risk_free_rate_annual <= 0.0:
            return
        # per-step compounding
        r_step = (1.0 + self.risk_free_rate_annual) ** (1.0 / self.steps_per_year) - 1.0
        self.balance *= (1.0 + r_step)

    def step(self, action):
        # risk-free accrual on cash before trading
        self._apply_risk_free_yield()

        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape[0] != self.n_assets:
            raise ValueError(f"[MicrostructureEnv] action dim={action.shape[0]}, expected {self.n_assets}.")

        # constrain to [0,1] and sum<=1 (long-only portfolio, remainder is cash)
        action = np.clip(action, 0.0, 1.0)
        s = action.sum()
        if s > 1.0:
            action /= (s + 1e-12)

        # rebalance frequency
        if (self.steps % self.rebalance_freq) != 0:
            # hold current allocations
            prices_t = self.prices[self.t]
            current_val = self.balance + (self.holdings * prices_t).sum()
            if current_val <= 0:
                target_alloc = np.zeros_like(action)
            else:
                target_alloc = (self.holdings * prices_t) / (current_val + 1e-12)
        else:
            target_alloc = action

        prices = self.prices[self.t]
        total_val = self.balance + (self.holdings * prices).sum()

        if total_val <= 0:
            # bankrupt → no trades, reward = large negative
            obs = self._get_observation()
            reward = -1e3
            self.t += 1
            self.steps += 1
            terminated = True
            truncated = True
            info = {
                "portfolio_value": float(self.nav),
                "balance": float(self.balance),
                "holdings": self.holdings.copy(),
                "turnover": 0.0,
                "total_fees": 0.0,
                "timestamp": str(self.timestamps[self.t - 1]) if (self.t - 1) < len(self.timestamps) else None,
                "reward_components": {
                    "log_ret": float(-1.0),
                    "realized_vol": 0.0,
                    "turnover_pen": 0.0,
                    "cvar_penalty": 0.0,
                },
                "trade_summary": [],
                "executed_alloc": np.zeros_like(self.holdings),
            }
            return obs, float(reward), terminated, truncated, info

        target_value = target_alloc * total_val

        self.holdings = np.asarray(self.holdings, dtype=float)
        executed_holdings = self.holdings.copy()
        total_fees = 0.0
        turnover_notional = 0.0
        trade_info: List[Dict[str, Any]] = []

        # per-asset trade simulation
        for i, sym in enumerate(self.assets):
            current_price = float(np.float32(prices[i]))
            current_qty = executed_holdings[i]
            desired_qty = target_value[i] / (current_price + 1e-12)
            delta_qty = desired_qty - current_qty

            if abs(delta_qty) < 1e-12:
                trade_info.append(
                    {
                        "asset": sym,
                        "requested_notional": 0.0,
                        "executed_notional": 0.0,
                        "exec_price_vwap": None,
                        "exec_qty": 0.0,
                        "fee": 0.0,
                        "fill_ratio": 1.0,
                        "diag": None,
                    }
                )
                continue

            side = 1 if delta_qty > 0 else -1
            requested_notional = abs(delta_qty) * current_price

            # liquidity-based cap
            max_partic_not = (
                self.max_participation * float(self.volumes[self.t, i]) * (current_price + 1e-12)
            )
            trade_notional = min(requested_notional, max_partic_not)

            if trade_notional < self.min_trade_notional:
                trade_info.append(
                    {
                        "asset": sym,
                        "requested_notional": float(requested_notional),
                        "executed_notional": 0.0,
                        "exec_price_vwap": None,
                        "exec_qty": 0.0,
                        "fee": 0.0,
                        "fill_ratio": 0.0,
                        "diag": None,
                    }
                )
                continue

            vol = float(self.hourly_vol[self.t, i])
            avg_vol = float(
                np.nanmean(self.hourly_vol[max(0, self.t - 24): self.t + 1, i]) + 1e-12
            )
            spread_est = max(
                self.spread_floor,
                0.5 * (vol / (avg_vol + 1e-12)) * self.spread_floor,
            )

            vwap, exec_notional, diag = simulate_execution_lob(
                side=side,
                mid_price=current_price,
                trade_notional=float(trade_notional),
                hourly_volume_base=float(self.volumes[self.t, i]),
                volatility=float(vol),
                spread=float(spread_est),
                params=self.exec_params,
                rng=self.rng,
            )

            if exec_notional <= 0.0:
                trade_info.append(
                    {
                        "asset": sym,
                        "requested_notional": float(requested_notional),
                        "executed_notional": 0.0,
                        "exec_price_vwap": None,
                        "exec_qty": 0.0,
                        "fee": 0.0,
                        "fill_ratio": 0.0,
                        "diag": diag,
                    }
                )
                continue

            exec_qty = (exec_notional / (vwap + 1e-12)) * side

            # single source of trading fee (no double-counting)
            fee = exec_notional * self.taker_fee

            executed_holdings[i] = current_qty + exec_qty
            total_fees += fee
            turnover_notional += exec_notional

            trade_info.append(
                {
                    "asset": sym,
                    "requested_notional": float(requested_notional),
                    "executed_notional": float(exec_notional),
                    "exec_price_vwap": float(vwap) if vwap is not None else None,
                    "exec_qty": float(exec_qty),
                    "fee": float(fee),
                    "fill_ratio": float(diag.get("fill_ratio", 1.0)) if isinstance(diag, dict) else 0.0,
                    "diag": diag,
                }
            )

        executed_val = (executed_holdings * prices).sum()
        new_balance = total_val - executed_val - total_fees

        # prevent negative cash by scaling down trades and fees
        if new_balance < -1e-6 and executed_val > 0:
            scale = max(0.0, (total_val - 1e-8) / (executed_val + total_fees + 1e-12))
            executed_holdings *= scale
            executed_val = (executed_holdings * prices).sum()
            total_fees *= scale
            turnover_notional *= scale

        # final safety: recompute and clamp
        new_balance = total_val - executed_val - total_fees
        new_balance = max(0.0, float(new_balance))

        self.holdings = executed_holdings
        self.balance = float(new_balance)
        self.nav = float(new_balance + executed_val)

        prev_nav = float(self.prev_nav) if self.prev_nav is not None else float(self.initial_balance)
        log_ret = math.log((self.nav + 1e-12) / (prev_nav + 1e-12))
        self.prev_nav = float(self.nav)

        self.episode_pvs.append(float(self.nav))
        if len(self.episode_pvs) > 2000:
            self.episode_pvs.pop(0)

        self.cum_turnover += float(turnover_notional)
        if self.enable_trade_logging:
            self.trade_ledger.append({"t": int(self.t), "trades": trade_info})

        # risk stats over rolling window
        window_pvs_raw = self.episode_pvs[-(self.risk_window + 1):]
        window_pvs_np = np.array(window_pvs_raw, dtype=np.float32)
        if window_pvs_np.size < 2:
            rets = np.array([], dtype=float)
        else:
            rets = (np.diff(window_pvs_np) / (window_pvs_np[:-1] + 1e-12)).astype(np.float32)

        realized_vol = float(np.std(rets)) if rets.size > 1 else 0.0

        # CVaR penalty
        cvar_pen = 0.0
        gamma_cvar = float(self.reward_params.get("gamma_cvar", 0.0))
        if gamma_cvar > 0 and rets.size >= 10:
            losses = (-rets).astype(np.float32)
            k = max(1, int(0.05 * len(losses)))
            tail = np.sort(losses)[-k:] if len(losses) >= k else losses
            cvar = float(np.mean(tail)) if tail.size > 0 else 0.0
            cvar_pen = gamma_cvar * cvar

        # turnover penalty (use current NAV instead of initial balance)
        eta_turn = float(self.reward_params.get("eta_turn", 0.0))
        nav_now = max(self.nav, 1e-12)
        turnover_pen = eta_turn * min(1.0, turnover_notional / nav_now)

        # volatility penalty
        lambda_vol = float(self.reward_params.get("lambda_vol", 0.0))

        # scale log_ret for numeric stability
        reward = float(1000.0 * log_ret - lambda_vol * realized_vol - turnover_pen - cvar_pen)

        # time update
        self.t += 1
        self.steps += 1
        truncated = self.t >= (self.T - 1)
        done = bool(truncated)

        if self.max_drawdown is not None:
            peak = max(self.episode_pvs) if self.episode_pvs else float(self.initial_balance)
            dd = 1.0 - (self.nav / (peak + 1e-12))
            if dd > self.max_drawdown:
                done = True

        info = {
            "portfolio_value": float(self.nav),
            "balance": float(self.balance),
            "holdings": self.holdings.copy(),
            "turnover": float(turnover_notional),
            "total_fees": float(total_fees),
            "timestamp": str(self.timestamps[self.t - 1]) if (self.t - 1) < len(self.timestamps) else None,
            "reward_components": {
                "log_ret": float(log_ret),
                "realized_vol": float(realized_vol),
                "turnover_pen": float(turnover_pen),
                "cvar_penalty": float(cvar_pen),
            },
            "trade_summary": trade_info,
            "executed_alloc": (self.holdings * self.prices[self.t - 1]) / (self.nav + 1e-12),
        }

        terminated = bool(done)
        return self._get_observation(), float(reward), terminated, bool(truncated), info

    # -----------------------
    # UTILS
    # -----------------------

    def render(self, mode: str = "human"):
        peak = max(self.episode_pvs) if self.episode_pvs else float(self.initial_balance)
        dd = 1.0 - (self.nav / (peak + 1e-12))
        print(
            f"[t={self.t}] NAV={self.nav:.2f}  cash={self.balance:.2f}  "
            f"peak={peak:.2f}  drawdown={dd*100:.2f}%  turnover={self.cum_turnover:.2f}"
        )

    def get_trade_ledger(self):
        return self.trade_ledger
