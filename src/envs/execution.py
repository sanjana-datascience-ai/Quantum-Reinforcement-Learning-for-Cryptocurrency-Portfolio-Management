"""
LOB execution simulator.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import math

DEFAULTS = {
    "n_levels": 80,
    "depth_decay": 0.8,
    "level_price_step": 0.0005,
    "min_level_qty_frac": 0.002,
    "eta_temp": 1e-6,
    "gamma_perm": 5e-7,
    "alpha": 0.5,
    "recovery_tau": 3.0,
    "noise_scale": 0.02,
    "slice_participation": 0.02,  # fraction of hourly base-volume per slice
    "max_slices": 20,
    "latency_sec": 0.2,
    "adv_select_prob": 0.05,
}


def build_synthetic_lob(
    mid_price: float,
    hourly_volume_base: float,
    volatility: float,
    spread: float,
    side: int = 1,
    n_levels: int = 80,
    depth_decay: float = 0.8,
    level_price_step: Optional[float] = None,
    min_level_qty_frac: float = 0.002,
) -> List[Tuple[float, float, float]]:
    """
    Build synthetic LOB for a given side.

    Args:
        mid_price: current mid price (quote)
        hourly_volume_base: hourly traded volume in base units
        volatility: used for shaping depth (currently unused)
        spread: bid-ask spread as fraction of mid
        side: +1 for buy (hit asks), -1 for sell (hit bids)

    Returns:
        List of levels: (price, qty_base, cumulative_qty_base)
    """
    if hourly_volume_base <= 0 or mid_price <= 0:
        return []

    if level_price_step is None:
        level_price_step = max(0.0005, spread / 4.0)

    side = 1 if side >= 0 else -1

    prices: List[float] = []
    qtys: List[float] = []

    top_level_frac = 0.03
    top_qty = max(1.0, hourly_volume_base * top_level_frac)

    for lvl in range(n_levels):
        # For buys → we consume asks above mid
        # For sells → we consume bids below mid
        if side == 1:
            price_lvl = mid_price * (
                1.0 + (spread / 2.0) + (lvl + 0.5) * level_price_step
            )
        else:
            price_lvl = mid_price * (
                1.0 - (spread / 2.0) - (lvl + 0.5) * level_price_step
            )
            # Ensure strictly positive prices
            price_lvl = max(price_lvl, 1e-12)

        qty = top_qty * (depth_decay**lvl)
        qty = max(qty, hourly_volume_base * min_level_qty_frac)
        prices.append(price_lvl)
        qtys.append(qty)

    cum = np.cumsum(qtys)
    levels = [(float(prices[i]), float(qtys[i]), float(cum[i])) for i in range(n_levels)]
    return levels


def consume_lob_for_notional(
    levels: List[Tuple[float, float, float]],
    notional: float,
) -> Tuple[float, float, List[Tuple[int, float, float]]]:
    """
    Consume LOB until requested notional (quote currency) is filled.
    Returns:
        vwap_price: float
        executed_notional: float
        consumed: list of (level_index, qty_base_filled, notional_filled)
    """
    if not levels or notional <= 0:
        if levels:
            return float(levels[0][0]), 0.0, []
        else:
            return 0.0, 0.0, []

    remaining_notional = float(notional)
    executed_notional = 0.0
    executed_base = 0.0
    consumed: List[Tuple[int, float, float]] = []

    for i, (price, qty_base, _) in enumerate(levels):
        level_notional = price * qty_base
        if remaining_notional <= 0:
            break

        if level_notional >= remaining_notional:
            qty_base_fill = remaining_notional / (price + 1e-12)
            executed_notional += price * qty_base_fill
            executed_base += qty_base_fill
            consumed.append((i, float(qty_base_fill), float(price * qty_base_fill)))
            remaining_notional = 0.0
            break
        else:
            executed_notional += level_notional
            executed_base += qty_base
            consumed.append((i, float(qty_base), float(level_notional)))
            remaining_notional -= level_notional

    if executed_notional == 0:
        return float(levels[0][0]), 0.0, []

    vwap = executed_notional / (executed_base + 1e-12)
    # VWAP should always be strictly positive
    vwap = max(float(vwap), 1e-12)
    return vwap, float(executed_notional), consumed


def simulate_execution_lob(
    side: int,
    mid_price: float,
    trade_notional: float,
    hourly_volume_base: float,
    volatility: float,
    spread: float,
    params: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    LOB-aware execution with simple impact model.

    Args:
        side: +1 for buy, -1 for sell
        mid_price: current mid price
        trade_notional: desired notional to trade (quote currency)
        hourly_volume_base: hourly traded volume in base units
        volatility: approx volatility (for noise scaling)
        spread: bid-ask spread fraction
        params: dict overriding DEFAULTS
        rng: np.random.Generator (for reproducible simulations)

    Returns:
        global_vwap: float
        executed_notional: float
        diagnostics: dict
    """
    if rng is None:
        rng = np.random.default_rng()

    if params is None:
        params = DEFAULTS.copy()
    else:
        p = DEFAULTS.copy()
        p.update(params)
        params = p

    side = 1 if side >= 0 else -1
    trade_notional = float(max(0.0, trade_notional))

    if trade_notional == 0.0 or hourly_volume_base <= 0.0 or mid_price <= 0.0:
        return float(mid_price), 0.0, {
            "slices": [],
            "vwap": float(mid_price),
            "executed_notional": 0.0,
            "fill_ratio": 0.0,
            "cum_perm_impact": 0.0,
        }

    executed_total = 0.0
    executed_base_total = 0.0
    slices_info: List[Dict[str, Any]] = []

    # convert slice participation from fraction of base-volume -> notional per slice
    slice_part = float(params.get("slice_participation", DEFAULTS["slice_participation"]))
    max_slice_base = max(1.0, slice_part * (hourly_volume_base + 1e-12))
    max_slices = int(params.get("max_slices", DEFAULTS["max_slices"]))

    max_slice_notional = max_slice_base * (mid_price + 1e-12)

    if trade_notional <= max_slice_notional:
        n_slices = 1
        slice_notional = trade_notional
    else:
        n_slices = int(math.ceil(trade_notional / max_slice_notional))
        n_slices = min(n_slices, max_slices)
        slice_notional = trade_notional / n_slices

    recovery_tau = float(params.get("recovery_tau", DEFAULTS["recovery_tau"]))
    eta_temp = float(params.get("eta_temp", DEFAULTS["eta_temp"]))
    gamma_perm = float(params.get("gamma_perm", DEFAULTS["gamma_perm"]))
    alpha = float(params.get("alpha", DEFAULTS["alpha"]))
    noise_scale = float(params.get("noise_scale", DEFAULTS["noise_scale"]))
    depth_decay = float(params.get("depth_decay", DEFAULTS["depth_decay"]))
    level_price_step = float(params.get("level_price_step", DEFAULTS["level_price_step"]))
    n_levels = int(params.get("n_levels", DEFAULTS["n_levels"]))
    adv_select_prob = float(params.get("adv_select_prob", DEFAULTS["adv_select_prob"]))

    cum_perm_impact = 0.0

    # ensure volatility non-zero for noise scaling
    volatility = max(volatility, 1e-6)

    for sl in range(n_slices):
        remaining = trade_notional - executed_total
        this_slice_notional = min(slice_notional, remaining)
        if this_slice_notional <= 0:
            break

        effective_mid = mid_price * (1.0 + cum_perm_impact)
        lob = build_synthetic_lob(
            effective_mid,
            hourly_volume_base,
            volatility,
            spread,
            side=side,
            n_levels=n_levels,
            depth_decay=depth_decay,
            level_price_step=level_price_step,
            min_level_qty_frac=params.get("min_level_qty_frac", DEFAULTS["min_level_qty_frac"]),
        )

        if not lob:
            break

        vwap, executed_notional_slice, consumed = consume_lob_for_notional(lob, this_slice_notional)
        if executed_notional_slice <= 0:
            break

        # impact depends on participation
        denom = hourly_volume_base * (mid_price + 1e-12)
        share = executed_notional_slice / (denom + 1e-12)

        temp_impact_frac = eta_temp * (share**alpha)
        perm_impact_frac = gamma_perm * share

        # permanent impact sign depends on side
        if side == 1:
            cum_perm_impact += perm_impact_frac
        else:
            cum_perm_impact -= perm_impact_frac

        # apply temporary impact in direction of trade
        exec_price_slice = vwap * (1.0 + temp_impact_frac * (1.0 if side == 1 else -1.0))

        # noisy slippage
        noise = rng.normal(scale=volatility * noise_scale * math.sqrt(max(0.0, share)))
        exec_price_slice *= (1.0 + noise)

        # Clamp execution price to strictly positive to avoid numerical issues
        if not math.isfinite(exec_price_slice) or exec_price_slice <= 0.0:
            exec_price_slice = max(vwap, 1e-12)

        executed_total += executed_notional_slice
        executed_base_total += executed_notional_slice / (exec_price_slice + 1e-12)

        slices_info.append(
            {
                "slice_index": int(sl),
                "exec_price_slice": float(exec_price_slice),
                "executed_notional_slice": float(executed_notional_slice),
                "consumed_levels": consumed,
                "temp_impact_frac": float(temp_impact_frac),
                "perm_impact_frac": float(perm_impact_frac),
            }
        )

        # recovery and occasional adverse selection
        if sl < n_slices - 1:
            decay = math.exp(-1.0 / max(1e-6, recovery_tau))
            cum_perm_impact *= decay

            if rng.random() < adv_select_prob:
                adverse_frac = 0.5 * temp_impact_frac
                if side == -1:
                    cum_perm_impact += adverse_frac
                else:
                    cum_perm_impact -= adverse_frac

    if executed_total > 0:
        global_vwap = sum(
            s["exec_price_slice"] * s["executed_notional_slice"] for s in slices_info
        ) / (executed_total + 1e-12)
        # Clamp VWAP as well
        if not math.isfinite(global_vwap) or global_vwap <= 0.0:
            global_vwap = max(mid_price, 1e-12)
    else:
        global_vwap = mid_price

    diagnostics = {
        "slices": slices_info,
        "vwap": float(global_vwap),
        "executed_notional": float(executed_total),
        "fill_ratio": float(executed_total / (trade_notional + 1e-12)),
        "cum_perm_impact": float(cum_perm_impact),
    }
    return float(global_vwap), float(executed_total), diagnostics
