import numpy as np
import pandas as pd

def example_data(
    n_users: int = 20,
    n_quarters: int = 8,   # e.g. 8 quarters = 2 years
    seed: int = 42,
    end_date: "pd.Timestamp | str | None" = None,
    global_trend: "float | None" = None,
) -> pd.DataFrame:
    """
    Return a synthetic longitudinal lifestyle/wellness dataset tailored for demos/tests.

    Scenario
    --------
    A startup is piloting a lifestyle monitoring gadget with many users over time.
    Each user contributes **weekly** measurements (on a random day of the week) for
    lifestyle factors and a simple emotion label. The dataset is designed to feel coherent in charts:
      - **Temporal structure**: regular weekly observations, ~2 years.
      - **Users** (default 20) with **age bands** and **mindsets** (categoricals).
      - **Participation windows** per user (`start_date`, `end_date`) so some users
        only appear for part of the period.
      - **Correlated variables** generated in sequence with interpretable direction:
          global_trend → mindset → work_stress → alcohol → sleep_hours → steps → nutrition → emotion.
      - **Hidden drivers**:
          * global_trend (scalar slope shared by everyone),
          * season (calendar quarter per row).

    Relevance
    ---------
    The data intentionally mixes numeric and categorical columns (both low- and
    high-cardinality), time, and correlated signals. It’s great for timeseries,
    small-multiple faceting by user/mindset/age_band, distribution plots, and
    dashboards showing trends and cohort effects.

    Parameters
    ----------
    n_users : int
        Number of distinct users (default 20).
    n_quarters : int
        Number of calendar quarters to cover (default 8 ≈ 2 years).
    seed : int
        RNG seed for reproducibility.
    end_date : pd.Timestamp | str | None
        Inclusive end of the series. If None, defaults to the **end of the previous
        calendar quarter** relative to now (e.g., if today is 2025-08-24, default is
        2025-06-30). Must align to or precede the final weekly timestamp.
    global_trend : float | None
        Hidden scalar trend shared by all users/dates. Positive values indicate
        overall improvement (alcohol & stress tend down; sleep, steps, nutrition up).
        If None, draws from N(0, 1).

    Notes
    -----
    - **Weekly cadence** at the start of the week (Mondays) using `freq='W-MON'`.
    - `start_date`/`end_date` are user-level but repeated on each row for convenience.
    - Emotion is a **'happy'/'neutral'/'sad'** label based on a hidden energy score
      plus rules (e.g., very high stress or very low sleep → 'sad').
    """
    # -----------------------------
    # Setup & date grid (weekly)
    # -----------------------------

    rng = np.random.default_rng(seed)

    # End-of-previous-quarter default (e.g., if today=2025-08-24 → 2025-06-30)
    if end_date is None:
        today = pd.Timestamp.today()
        prev_q = (today.to_period("Q") - 1)
        end_date = prev_q.end_time.normalize()
    else:
        end_date = pd.to_datetime(end_date).normalize()

    # Compute start_date aligned to the beginning of the quarter n_quarters ago
    # Example: with n_quarters=8 and end_date=2025-06-30 → start_date=2023-07-01
    start_date = (end_date - pd.offsets.QuarterEnd(n_quarters)).normalize() + pd.offsets.Day(1)

    # Use one week earlier as the grid’s end so random offsets can’t spill past end_date
    grid_end = end_date - pd.Timedelta(days=6)

    # Weekly Mondays covering the entire quarter-aligned range
    dates = pd.date_range(start=start_date, end=grid_end, freq="W-MON")
    
    # Standardized time index in [-1, 1] across the global date axis
    t_by_date = pd.Series(np.linspace(-1.0, 1.0, num=len(dates)), index=dates)

    # Hidden seasonal (not returned): quarter per row
    # Season contributes signed offsets for each variable.
    # season = pd.Series(dates).dt.quarter.values  # 1..4
    season_ws = {1: -0.3, 2: -0.1, 3: 0.2,  4: 0.4}   # work_stress offset
    season_al = {1: -0.1, 2: 0.0,  3: 0.1,  4: 0.3}   # alcohol offset
    season_sl = {1:  0.1, 2: 0.0,  3: -0.1, 4: -0.2}  # sleep_hours offset
    season_st = {1:  200, 2: 300,  3: -200, 4: -300}  # steps offset
    season_nq = {1:  0.2, 2: 0.1,  3: -0.1, 4: -0.2}  # nutrition offset

    # -----------------------------
    # Users, demographics, mindsets
    # -----------------------------
    users = np.array([f"user{i:02d}" for i in range(1, n_users + 1)])

    # Mindset affects a per-user slope over time: growth (+), neutral (~0), fixed (-)
    mindset_labels = np.array(["growth", "neutral", "fixed"])
    user_mindset = rng.choice(mindset_labels, size=n_users, replace=True)
    mindset_slope = np.select(
        [user_mindset == "growth", user_mindset == "neutral", user_mindset == "fixed"],
        [0.5, 0.0, -0.5]
    )

    # Age band: sample ages 20–80, bin into 5-year groups
    ages = rng.integers(20, 81, size=n_users)
    bins = np.arange(20, 85, 5)  # 20,25,...,80
    age_labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]  # "20-24", ...
    age_band = pd.cut(ages, bins=bins, labels=age_labels, right=False, include_lowest=True)

    # -----------------------------
    # Participation windows per user
    # -----------------------------
    window_start, window_end = dates.min(), dates.max()
    window_mid = window_start + (window_end - window_start) / 2

    # Pick a membership "midpoint" near the window midpoint.
    midpoint_offsets_w = rng.normal(0, len(dates)/3, size=n_users)  # in weeks
    mids = (window_mid + pd.to_timedelta(midpoint_offsets_w, unit="W")).normalize()

    # Draw long-ish durations so many users span the whole window,
    # with some partial coverage (clipped to the window edges).
    dur_weeks = np.clip(rng.normal(loc=len(dates)*1.2, scale=len(dates)*0.4, size=n_users), 8, len(dates)*2)
    starts = (mids - pd.to_timedelta(dur_weeks/2, unit="W"))
    ends   = (mids + pd.to_timedelta(dur_weeks/2, unit="W"))
    starts = pd.Series(starts).clip(lower=window_start, upper=window_end).dt.normalize()
    ends   = pd.Series(ends).clip(lower=window_start, upper=window_end).dt.normalize()

    # Hidden global trend (shared). Positive → better: stress/alcohol down, sleep/steps/nutrition up.
    if global_trend is None:
        global_trend = float(rng.normal(0.0, 1.0))

    # -----------------------------
    # Build panel skeleton (date × user) and attach user metadata
    # -----------------------------
    user_meta = pd.DataFrame({
        "user": users,
        "mindset": user_mindset,
        "mindset_slope": mindset_slope,
        "age_band": age_band.astype(str),
        "start_date": starts.values,
        "end_date": ends.values,
        # Per-user baselines to add persistent heterogeneity
        "ws_baseline": rng.normal(5.0, 0.8, size=n_users),      # work_stress
        "al_baseline": rng.normal(1.2, 0.5, size=n_users),      # alcohol
        "sl_baseline": rng.normal(7.0, 0.6, size=n_users),      # sleep
        "st_baseline": rng.normal(6000, 1500, size=n_users),    # steps
        "nq_baseline": rng.normal(6.5, 0.7, size=n_users),      # nutrition
    })

    # Cross-join weekly dates × users, then keep only active rows per user window
    panel = (
        pd.MultiIndex.from_product([dates, users], names=["date", "user"])
        .to_frame(index=False)
        .merge(user_meta, on="user", how="left")
    )
    active = (panel["date"] >= panel["start_date"]) & (panel["date"] <= panel["end_date"])
    panel = panel.loc[active].reset_index(drop=True)

    # Attach per-row hidden drivers
    panel["quarter"] = panel["date"].dt.quarter
    panel["t"] = panel["date"].map(t_by_date)

    # Global & mindset time components
    g = global_trend * panel["t"].to_numpy()
    m = panel["mindset_slope"].to_numpy() * panel["t"].to_numpy()

    # -----------------------------
    # Sequential/correlated generation
    # -----------------------------
    # 1) Work stress (0..10): lower with positive g/m; seasonal offsets
    ws = (
        panel["ws_baseline"].to_numpy()
        - 0.9 * g
        - 0.7 * m
        + panel["quarter"].map(season_ws).to_numpy()
        + rng.normal(0, 0.7, size=len(panel))
    )
    ws = np.clip(ws, 0, 10)

    # 2) Alcohol (integer 0..8): rises with stress; lowered by positive g/m; seasonal offsets
    stress_z = (ws - 5.0) / 2.0
    al = (
        panel["al_baseline"].to_numpy()
        - 0.6 * g
        - 0.4 * m
        + 0.45 * stress_z
        + panel["quarter"].map(season_al).to_numpy()
        + rng.normal(0, 0.5, size=len(panel))
    )
    al = np.clip(al, 0, 8)
    al = np.rint(al).astype(int)

    # 3) Sleep hours (3..11): improves with positive g/m; worsens with stress & alcohol; seasonal offsets
    sl = (
        panel["sl_baseline"].to_numpy()
        + 0.6 * g
        + 0.4 * m
        - 0.35 * stress_z
        - 0.20 * al
        + panel["quarter"].map(season_sl).to_numpy()
        + rng.normal(0, 0.4, size=len(panel))
    )
    sl = np.clip(sl, 3, 11)

    # 4) Steps (200..20_000): better with sleep; worse with stress & alcohol; seasonal offsets
    st = (
        panel["st_baseline"].to_numpy()
        + 1200 * g
        + 900 * m
        + 900 * (sl - 7.0)
        - 500 * stress_z
        - 180 * al
        + panel["quarter"].map(season_st).to_numpy()
        + rng.normal(0, 800, size=len(panel))
    )
    st = np.clip(st, 200, 20000).astype(int)

    # 5) Nutrition quality (0..10): better with sleep & steps; worse with stress & alcohol; seasonal offsets
    nq = (
        panel["nq_baseline"].to_numpy()
        + 0.5 * g
        + 0.4 * m
        + 0.18 * (sl - 7.0)
        + 0.00025 * (st - 6000)
        - 0.28 * stress_z
        - 0.20 * al
        + panel["quarter"].map(season_nq).to_numpy()
        + rng.normal(0, 0.5, size=len(panel))
    )
    nq = np.clip(nq, 0, 10)

    # Hidden continuous energy score
    energy = (
        0.45 * sl
        + 0.001 * st
        + 0.5 * nq
        - 0.45 * ws
        - 0.35 * al
        + rng.normal(0, 0.9, size=len(panel))
    )

    # 6) Emotion label: rules + quantiles for a good mix
    emotion = np.array(["neutral"] * len(panel), dtype=object)
    sad_mask = (ws >= 8.0) | (sl < 5.0) | (al >= 5)
    emotion[sad_mask] = "sad"
    remaining = ~sad_mask
    if remaining.any():
        q25, q75 = np.quantile(energy[remaining], [0.25, 0.75])
        emotion[remaining & (energy >= q75)] = "happy"
        emotion[remaining & (energy < q25)] = "sad"

    # -----------------------------
    # Random weekday offset
    # -----------------------------
    
    panel["offset_days"] = rng.integers(0, 7, size=len(panel))
    panel["date"] = panel["date"] + pd.to_timedelta(panel["offset_days"], unit="D")

    # -----------------------------
    # Assemble final DataFrame
    # -----------------------------

    df = pd.DataFrame({
        "date": panel["date"].values,
        "user": panel["user"].values,
        "age_band": panel["age_band"].values,
        "mindset": panel["mindset"].values,
        "sleep_hours": sl.round(2),
        "steps": st,
        "alcohol": al,
        "work_stress": ws.round(2),
        "nutrition": nq.round(2),
        "emotion": pd.Categorical(emotion, categories=["sad", "neutral", "happy"], ordered=True),
    })

    # Helpful calendar categoricals
    df["weekday"] = df["date"].dt.day_name()

    return df