from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union, Sequence, Tuple, Dict, Callable, Literal, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties
from textwrap import wrap
from math import ceil

# Logging
from lazychart.log import create_logger
logger = create_logger(__name__)

# -----------------------------------------------------------------------------
# Common docstring added to all methods
# -----------------------------------------------------------------------------

def add_docstring(extra: str):
    """Decorator: append a shared docstring chunk to a function's __doc__."""
    def deco(func):
        func.__doc__ = (func.__doc__ or "") + "\n\n" + extra
        return func
    return deco

COMMON_DOCSTRING = """
    Configuration for a single chart call.

    Core selection / aggregation
    ----------------------------
    data : pd.DataFrame
        Source data (required for most charts).
    x : str, optional
        Categorical or datetime column for the x-axis (bar/line).
    y : str | Sequence[str], optional
        Value column(s) to aggregate. If omitted, counts are used.
    group_by : str, optional
        Sub-category column (produces grouped bars / multiple lines).
    aggfunc : str | callable, default 'sum'
        Aggregation function ('sum', 'mean', etc.).
    x_period : {'month','quarter','year','week','day'}, optional
        If set and `x` is datetime-like, bins to the chosen period.
    show_gaps : bool, default: True
        Where x_period is used, fill in any gaps in the series

    Bar
    ------------------------
    stacking : {'stacked','proportion','none'}, default 'stacked'
    
    Pie
    ------------------------
    values : str, optional
        Numeric column of slice sizes.
    names : str, optional
        Category column for pie slices.
    show_labels : bool, default False
        Whether to label wedges (in addition or alternative to using legend)

    Ordering
    ------------------------
    sort_x : {'label','value','none'} | sequence, optional
    sort_group_by : {'label','value','none'} | sequence, optional
    sort_names : {'label','value','none'} | sequence, optional
    *_ascending : bool, optional
        Ascending/descending for the corresponding sort_*.

    Axis limits & formatting
    ------------------------
    x_min, x_max, y_min, y_max : float, optional
        Axis bounds; omitted side stays automatic.
    x_axis_format, y_axis_format : {'percent','comma','short'} or None
        Built-in numeric formatters. 'short' ~ 1.2k/3.4M/5.6B.
    decimals : int, default 0
    xtick_rotation, ytick_rotation : int, optional
        Explicit tick rotations (x auto-rotates when labels are long).

    Legend / grid / figure
    ----------------------
    legend : {'right','bottom','none'}, default 'right'
    grid_x : bool, default False
    grid_y : bool, default True
    resize_fig : bool, default True
        Expand the figure to fit the legend and title without reducing data axes size.
    fig_size : (w,h), default (6.4, 4.8)
    save_path : str, optional
        If provided, the figure is saved to this path after drawing.
    show_fig : bool, default True
        Call plt.show() after drawing and return None instead of fig, ax, chart_data
        (this keeps the API clean and avoids extra outputs in Jupyter notebooks)
    return_values : bool, default False
        Return fig, ax, chart_data after drawing.
        (useful if you want to extract specific figures from the chart data)

    Labels
    --------------------
    title : str, optional
        Figure-level title (wrapped to fit).
    subtitle : str, optional
    x_label : str, optional
    y_label : str, optional
    legend_label : str, optional
    title_size : int | {'small','medium','large','x-large'}, optional
    subtitle_size : int | {'small','medium','large','x-large'}, optional
    x_label_size : int | {'small','medium','large','x-large'}, optional
    y_label_size : int | {'small','medium','large','x-large'}, optional
    tick_size : int | {'small','medium','large','x-large'}, optional

    Label helpers
    ------------------------
    label_map : dict, default {}
        Map raw category labels to display labels (applies to x/group_by/names).
    group_threshold : int | float, optional
        If int: keep top-N categories, group the rest as 'Other'.
        If 0< float <=1: keep categories cumulatively until this proportion.
    group_other_name : str, default 'Other'
        Label for grouped small categories.

    Other
    --------------------
    sticky : bool, default False
        Whether to store arguments passed for the current chart for future charts
    use_sticky : bool, default True
        Whether to apply existing sticky arguments to the current chart
    palette : str | Sequence[str], optional
        Per-chart color cycle (overrides global palette for this chart only).
    target_x : float, optional
        Adds a vertical dotted line at a given x value
    target_y : float, optional
        Adds a horizontal dotted line at a given y value
    target_x_label : str, optional
    target_y_label : str, optional

    """

# -----------------------------------------------------------------------------
# Palettes
# -----------------------------------------------------------------------------

RAINBOW_PALETTE = [
    '#e41a1c',  # Red
    '#ff7f00',  # Orange
    '#ffff33',  # Yellow
    '#4daf4a',  # Green
    '#377eb8',  # Blue
    '#984ea3',  # Purple
    '#f781bf',  # Pink
    '#a65628',  # Brown
    '#999999',  # Gray
    '#66c2a5'   # Aqua
]

COLORBLIND_PALETTE = [
    '#E69F00',  # Orange
    '#56B4E9',  # Sky Blue
    '#009E73',  # Bluish Green
    '#F0E442',  # Yellow
    '#0072B2',  # Blue
    '#D55E00',  # Vermillion
    '#CC79A7',  # Reddish Purple
    '#999999',  # Gray
    '#000000',  # Black
    '#FFFFFF'   # White
]

def _resolve_palette(palette: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    """
    Resolve a palette specification into a concrete list of colors.

    This function is designed to be flexible:
    - If `palette` is None, returns None. This allows passing the result directly
      to plotting functions (e.g., `plot(color=_resolve_palette(...))`) without extra checks.
    - If `palette` is a string and it matches a known named palette, returns that palette's list of colors.
    - If `palette` is already a list/tuple of colors, it is returned as a list.
    - Raises ValueError if the palette cannot be resolved.

    Returns:
        A list of colors, or None if no palette was provided.
    """
    if palette is None:
        return None

    named_palettes = {
        "default": RAINBOW_PALETTE,
        "rainbow": RAINBOW_PALETTE,
        "colorblind": COLORBLIND_PALETTE,
    }

    if isinstance(palette, str):
        if palette.lower() in named_palettes:
            return list(named_palettes[palette.lower()])
        else:
            raise ValueError(f"Unrecognized palette string '{palette}'")

    # Already a list/tuple of colors
    try:
        return list(palette)
    except Exception as e:
        raise ValueError(f"Invalid palette type: {type(palette)} ({e})")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

AxisFormat = Optional[Literal["percent", "comma", "short"]]
LegendPosition = Literal["right", "bottom", "none"]

@dataclass
class ChartConfig:

    # ---- Core selection / aggregation ----
    data: Optional[pd.DataFrame] = None
    x: Optional[str] = None
    y: Optional[Union[str, Sequence[str]]] = None
    group_by: Optional[str] = None
    aggfunc: Union[str, Callable] = "sum"
    x_period: Optional[Literal["month", "quarter", "year", "week", "day"]] = None
    show_gaps: bool = True

    # ---- Bar ----
    stacking: Literal["stacked", "proportion", "none"] = "stacked"
    
    # ---- Pie ----
    values: Optional[str] = None
    names: Optional[str] = None
    show_labels: bool = False

    # ---- Grouping ----
    group_threshold: Optional[Union[int, float]] = None
    group_other_name: str = "Other"

    # ---- Ordering ----
    sort_x: Optional[Union[Literal["label", "value", "none"], Sequence[Any]]] = None
    sort_x_ascending: Optional[bool] = None
    sort_group_by: Optional[Union[Literal["label", "value", "none"], Sequence[Any]]] = None
    sort_group_by_ascending: Optional[bool] = None
    sort_names: Optional[Union[Literal["label", "value", "none"], Sequence[Any]]] = None
    sort_names_ascending: Optional[bool] = None

    # ---- Axis formats ----
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    x_axis_format: AxisFormat = None
    y_axis_format: AxisFormat = None
    decimals: int = 0
    xtick_rotation: Optional[int] = None
    ytick_rotation: Optional[int] = None

    # ---- Legend & grid ----
    legend: LegendPosition = "right"
    grid_x: bool = False
    grid_y: bool = True
    resize_fig: bool = True

    # ---- Figure ----
    fig_size: Tuple[float, float] = (6.4, 4.8)
    save_path: Optional[str] = None
    show_fig: bool = True
    return_values: bool = False

    # ---- Labels ----
    title: Optional[str] = None
    subtitle: Optional[str] = None
    x_label : Optional[str] = None
    y_label : Optional[str] = None
    legend_label: Optional[str] = None
    label_map: Dict[Any, Any] = field(default_factory=dict)
    title_size: Optional[Union[int, str]] = None
    subtitle_size: Optional[Union[int, str]] = None
    x_label_size: Optional[Union[int, str]] = None
    y_label_size: Optional[Union[int, str]] = None
    tick_size: Optional[Union[int, str]] = None
    show_percent: Union[bool, str] = True

    # ---- Other ----
    palette: Optional[Union[str, Sequence[str]]] = None
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    target_x_label: Optional[str] = None
    target_y_label: Optional[str] = None

    def _normalize_x_period(self, x_period: str | None, *, week_anchor: str = "W-MON") -> str | None:
        """
        Map human-friendly period names to pandas frequency codes.
        Returns canonical pandas freq (e.g., 'Y','Q','M','W-MON','D') or None.
        """

        _PERIOD_SYNONYMS: dict[str, str] = {
            "d": "D", "day": "D", "daily": "D",
            "w": "W-MON", "week": "W-MON", "weekly": "W-MON",
            "m": "M", "mon": "M", "month": "M", "monthly": "M",
            "q": "Q", "qtr": "Q", "quarter": "Q", "quarterly": "Q",
            "y": "Y", "yr": "Y", "year": "Y", "annual": "Y", "yearly": "Y",
        }

        if x_period is None:
            return None
        key = str(x_period).strip().lower()
        if key in _PERIOD_SYNONYMS:
            freq = _PERIOD_SYNONYMS[key]

            # If caller prefers a different week anchor, swap it in
            if freq.startswith("W-") and week_anchor:
                return week_anchor
            return freq

        # If the user already passed a pandas-like code, accept it (e.g., 'Y','W-SUN')
        if key.upper() in {"Y", "Q", "M", "D"}:
            return key.upper()
        if key.upper().startswith("W-"):
            return key.upper()

        raise ValueError(
            f"Invalid x_period={x_period!r}. "
            f"Use one of: day/week/month/quarter/year or codes: D, W-MON, M, Q, Y."
        )

    def __post_init__(self) -> None:
        """Check/fix arguments, apply synonyms etc"""

        # Legend placement
        if isinstance(self.legend, str):
            leg = self.legend.lower()
            if leg in {"hide", "none", "off"}:
                self.legend = "none"  # type: ignore[assignment]
            elif leg in {"right", "r"}:
                self.legend = "right"  # type: ignore[assignment]
            elif leg in {"bottom", "b"}:
                self.legend = "bottom"  # type: ignore[assignment]

        # Decimals
        if self.decimals is None or self.decimals < 0:
            self.decimals = 0

        # Treat empty string as False for 'show_percent'
        if isinstance(self.show_percent, str) and not self.show_percent.strip():
            self.show_percent = False

        # Ensure label_map is a dict
        if self.label_map is None:
            self.label_map = {}

        # Check Literal values
        if self.stacking not in ("stacked", "proportion", "none"):
            raise ValueError(f"Invalid stacking: {self.stacking}")
        
        # X periods
        self.x_period = self._normalize_x_period(self.x_period)
        if self.x_period and self.show_gaps is None:
            self.show_gaps = True
        

# -----------------------------------------------------------------------------
# Styling helpers
# -----------------------------------------------------------------------------

def _resolve_fontsize(size: Union[int, str, None], label: str) -> int:
    """Determine font size by looking up a category preset if not passed explicitly

    Parameters
    ----------
    value : int | {'small','medium','large','x-large'}, optional
    category : str {'title','subtitle','x_label','y_label','tick'}
    """

    FONT_PRESETS = {
        "small":    {"title": 13, "subtitle": 11, "x_label": 10, "y_label": 10, "tick": 9},
        "medium":   {"title": 15, "subtitle": 13, "x_label": 12, "y_label": 12, "tick": 11},
        "large":    {"title": 17, "subtitle": 15, "x_label": 14, "y_label": 14, "tick": 13},
        "x-large":  {"title": 19, "subtitle": 17, "x_label": 16, "y_label": 16, "tick": 15},
    }

    if isinstance(size, int):
        return size
    elif isinstance(size, str):
        preset = FONT_PRESETS.get(size.lower())
        if preset:
            return preset[label]
    return FONT_PRESETS["medium"][label] # use medium by default

def _format_axis(ax: Axes, which: str, fmt: AxisFormat, decimals: int) -> None:
    axis = ax.yaxis if which == "y" else ax.xaxis
    if fmt == "percent":
        axis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=decimals))
    elif fmt == "comma":
        axis.set_major_formatter(mticker.StrMethodFormatter(f"{{x:,.{decimals}f}}"))
    elif fmt == "short":
        def short(x, pos=None):
            if abs(x) >= 1_000_000_000:
                return f"{x/1_000_000_000:.{decimals}f}B"
            if abs(x) >= 1_000_000:
                return f"{x/1_000_000:.{decimals}f}M"
            if abs(x) >= 1_000:
                return f"{x/1_000:.{decimals}f}k"
            return f"{x:.{decimals}f}"
        axis.set_major_formatter(mticker.FuncFormatter(short))


# -----------------------------------------------------------------------------
# ChartMonkey class
# -----------------------------------------------------------------------------

class ChartMonkey:
    """Stateful chart helper that centralizes configuration and layout."""

    # ---- Init / state ----------------------------------------------------------
    def __init__(self, palette: Optional[Union[str, Sequence[str]]] = "rainbow") -> None:
        """
        Parameters
        ----------
        palette : str | Sequence[str], optional, default 'rainbow'
            Set the global matplotlib color cycle immediately. Presets:
            'rainbow' (default), 'colorblind' (aka 'colourblind').
            Pass a list of colors to use a custom palette.
        """
        self._sticky: Dict[str, Any] = {}
        self._chart_params: Optional[ChartConfig] = None
        if palette is not None:
            self.set_palette(palette)

    def sticky(self, **kwargs: Dict[str, Any]):
        """
        Adds arguments to the 'sticky' dictionary for reuse.
        These can also be added using sticky=True with individual chart calls.
        Calling with no arguments, e.g., `sticky()`, will clear all sticky arguments.
        """

        # The dictionary of arguments passed from a chart
        sticky_kwargs = kwargs.get('kwargs', {})
        
        # Keyword arguments passed directly as this is a public facing API
        sticky_kwargs.update({k: v for k, v in kwargs.items() if k != 'kwargs'})
        
        if sticky_kwargs:
            # add to the sticky dictionary
            for k, v in sticky_kwargs.items():
                self._sticky[k] = v
        else:
            # reset the sticky dictionary
            self._sticky.clear()

    def _set_params(self, **kwargs: Any) -> ChartConfig:
        """Merge sticky + kwargs into a `ChartConfig` and store on the instance."""

        # use pop to avoid retaining control flags
        sticky = kwargs.pop("sticky", False)
        use_sticky = kwargs.pop("use_sticky", True)

        # save kwargs for future charts
        if sticky:
            self.sticky(kwargs=kwargs)

        # create current chart config using current kwargs and/or sticky values
        base = dict(getattr(self, "_sticky", {})) if use_sticky else {}
        config_values = {**base, **kwargs}
        config = ChartConfig(**config_values)
        self._chart_params = config
        return config

    # ---- Palette API ------------------------------------------------------------
    def set_palette(self, palette: Union[str, Sequence[str]]) -> None:
        """Set the global color cycler (matplotlib rcParam)."""
        colors = _resolve_palette(palette)
        if colors is not None:
            rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    # ---- Sample data ------------------------------------------------------------
    def example_data(self, n: int = 2000, seed: int = 42) -> pd.DataFrame:
        """Return a synthetic wellness dataset for demos/tests."""
        np.random.seed(seed)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=n)
        user = np.random.choice([f'user{i}' for i in range(1, 51)], size=n)
        sleep_hours = np.clip(np.random.normal(7, 1, n), 4, 10)
        steps = np.clip(np.random.normal(6000, 1500, n), 200, 15000)
        alcohol = np.clip(np.random.poisson(1.2, n), 0, 8)
        work_stress = np.clip(np.random.normal(5, 2, n), 0, 10)
        nutrition = np.clip(np.random.normal(6.5, 1.5, n), 0, 10)
        score = (0.4*sleep_hours + 0.001*steps - 0.5*alcohol +
                 0.3*nutrition - 0.2*work_stress + np.random.normal(0, 1, n))
        emotion = pd.cut(score, bins=[-10, -0.5, 0.5, 10], labels=['Engaged', 'Energetic', 'Happy'])
        df = pd.DataFrame(dict(
            date=dates, user=user, sleep_hours=sleep_hours, steps=steps,
            alcohol=alcohol, work_stress=work_stress, nutrition=nutrition,
            score=score, predicted_emotion=emotion))
        df['month'] = df['date'].dt.to_period('M').astype(str)
        df['weekday'] = df['date'].dt.day_name()
        return df

# -----------------------------------------------------------------------------
# Core Data Logic                                                           
# -----------------------------------------------------------------------------

    def _prepare_dataframe(self) -> pd.DataFrame:
        """Check data to chart exists and return a defensive copy to avoid modifying caller's dataframe."""
        cfg = self._chart_params
        if cfg.data is None:
            raise ValueError("`data` must be provided in ChartConfig (or via kwargs/sticky).")
        if not isinstance(cfg.data, pd.DataFrame):
            raise TypeError("`data` must be a pandas DataFrame.")
        return cfg.data.copy()

    def _coerce_x_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        If cfg.x_period is set, coerce the x dimension to the *start* of the period.
        - If x is a column: mutate that column.
        - If x is the index: mutate the index (preserving its name).
        Uses canonical pandas freq codes already normalized in ChartConfig.__post_init__.
        """
        cfg = self._chart_params
        if not cfg or not cfg.x or not cfg.x_period:
            return df

        # Prefer column if it exists; otherwise allow index match
        use_index = False
        if cfg.x in df.columns:
            x_series = df[cfg.x]
        elif df.index.name == cfg.x:
            x_series = pd.Series(df.index, index=df.index)
            use_index = True
        else:
            # x not found; nothing to coerce
            return df

        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(x_series):
            try:
                x_series = pd.to_datetime(x_series, errors="raise")
            except Exception:
                return df

        # Coerce to period start using canonical freq (D, W-XXX, M, Q, Y)
        coerced = x_series.dt.to_period(cfg.x_period).dt.start_time

        # Write back
        if use_index:
            df.index = pd.DatetimeIndex(coerced.values, name=cfg.x)
        else:
            df.loc[:, cfg.x] = coerced

        return df

    def _apply_label_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply label_map to remap cateogrical x or group_by column values"""
        cfg = self._chart_params
        if not cfg.label_map:
            return df
        for col in (cfg.x, cfg.group_by, cfg.names):
            if col and col in df.columns:
                df[col] = df[col].map(cfg.label_map).fillna(df[col])
        return df

    def _group_small_categories(self, df: pd.DataFrame, *, target_col: str, values_col: Optional[str]) -> pd.DataFrame:
        """Groups target_col levels where values_col is less than group_threshold"""
        cfg = self._chart_params
        if cfg.group_threshold is None:
            return df
        if target_col not in df.columns:
            return df
        # Compute size (sum of value column if it exists, or number of rows otherwise) of each category
        if values_col and values_col in df.columns:
            sizes = df.groupby(target_col, dropna=False)[values_col].sum().sort_values(ascending=False)
        else:
            sizes = df[target_col].value_counts(dropna=False)
        # Apply the threshold
        thr = cfg.group_threshold
        if isinstance(thr, int): # keep top N categories
            keep = sizes.head(max(thr, 0)).index
        elif isinstance(thr, float) and 0 < thr <= 1: # keep top %
            total = float(sizes.sum())
            if total == 0:
                return df
            cum = sizes.cumsum() / total
            keep = cum[cum <= thr].index
            if len(keep) == 0 and not sizes.empty:
                keep = pd.Index([sizes.index[0]])
        else:
            return df
        # Name the "other" category
        other_name = cfg.group_other_name or "Other"
        df[target_col] = np.where(df[target_col].isin(keep), df[target_col], other_name)
        return df

    def _aggregate_data(self) -> pd.DataFrame:
        """
        Aggregate ``data`` into a **tidy/long** table suitable for downstream plotting.

        This function *does not* return a wide/pivoted table. Instead, it always returns a
        long-form dataframe with a single numeric column named ``"value"`` and one or more
        key columns (e.g. ``x`` and optionally ``group_by``). Most plots that show grouped or
        stacked series will need to apply an additional pivot step *after* aggregation.        
        """
        cfg = self._chart_params
        df = self._prepare_dataframe() # check data exists, create defensive copy
        df = self._coerce_x_period(df) # x_period mapping
        df = self._apply_label_map(df) # label mapping

        # Place values into a list
        # (note values are often "y" but for pie chart will be "values)
        y_cols: Sequence[str] = []
        if cfg.y is None:
            if cfg.values and cfg.values in df.columns:
                y_cols = [cfg.values]
            else:
                y_cols = []
        else:
            y_cols = [cfg.y] if isinstance(cfg.y, str) else list(cfg.y)

        # Group up smaller categories into an "other" category
        if cfg.names:
            df = self._group_small_categories(df, target_col=cfg.names, values_col=(cfg.values or (y_cols[0] if y_cols else None)))
        elif cfg.x:
            df = self._group_small_categories(df, target_col=cfg.x, values_col=(cfg.values or (y_cols[0] if y_cols else None)))

        # Build group keys
        keys: list[str] = []
        if cfg.x:
            keys.append(cfg.x)
        if cfg.group_by:
            keys.append(cfg.group_by)
        if cfg.names and cfg.names not in keys:
            keys.append(cfg.names)

        # Validate group_by
        if cfg.group_by:
            if pd.api.types.is_numeric_dtype(df[cfg.group_by]) and not isinstance(df[cfg.group_by], pd.CategoricalDtype):
                unique_vals = df[cfg.group_by].nunique(dropna=True)
                if unique_vals <= 20: # for low cardinality data we can convert for the user
                    df[cfg.group_by] = df[cfg.group_by].astype('category')
                else:
                    raise ValueError(
                        f"Column '{cfg.group_by}' is numeric with {unique_vals} unique values. "
                        "For grouping/legend behaviour, convert it to a categorical type, e.g.:\n\n"
                        f"    df['{cfg.group_by}'] = df['{cfg.group_by}'].astype('category')\n\n"
                        "See: https://pandas.pydata.org/docs/reference/api/pandas.Categorical.html"
                    )

        # Pie path
        if cfg.names and (cfg.values is not None):
            if cfg.values not in df.columns:
                raise ValueError(f"`values` column '{cfg.values}' not found in data.")
            grouped = df.groupby(keys, dropna=False, observed=True)[cfg.values].agg(cfg.aggfunc)  # type: ignore[arg-type]
            out = grouped.reset_index().rename(columns={cfg.values: "value"})
            return out

        # Counts
        if not y_cols:
            counted = df.groupby(keys, dropna=False, observed=True).size().reset_index(name="value")
            return counted

        # Aggregate y columns
        grouped = df.groupby(keys, dropna=False, observed=True)[y_cols].agg(cfg.aggfunc)  # type: ignore[arg-type]
        if isinstance(grouped, pd.Series):
            # Series -> name is the original y (or the agg’d series name)
            out = grouped.reset_index().rename(columns={grouped.name: "value"})
        else:
            out = grouped.reset_index()
            if len(y_cols) > 1:
                # Multiple y’s -> melt to long and call it "value"
                out = out.melt(id_vars=keys, value_vars=y_cols, var_name=(cfg.group_by or "variable"), value_name="value")
            else:
                # Single y but came back as a 1-col DataFrame; ensure it's called "value"
                col = y_cols[0]
                if col in out.columns and col != "value":
                    out = out.rename(columns={col: "value"})

        return out
    
    def _pivot_data(self, df_long: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot a long-form table [x, (group_by), value] to wide for grouped/stacked series.

        - When cfg.show_gaps is True and cfg.x_period is set, ensure a continuous DatetimeIndex
        spanning min..max period starts (gaps appear as 0 after fill).
        - If group_by is categorical, reindex columns to the full category set for a stable legend.
        - If stacking == "proportion", normalize each row to sum to 1.0 (treating missing groups as 0).
        """
        import numpy as np
        import pandas as pd

        cfg = self._chart_params

        def _full_x_index(index_like: pd.Index) -> pd.Index | None:
            """Build a continuous DatetimeIndex (period starts) using canonical cfg.x_period."""
            if not cfg or not cfg.x_period or not isinstance(index_like, pd.Index):
                return None
            if not pd.api.types.is_datetime64_any_dtype(index_like):
                return None
            start, end = index_like.min(), index_like.max()
            if pd.isna(start) or pd.isna(end):
                return None
            pr = pd.period_range(start=start, end=end, freq=cfg.x_period)
            return pd.DatetimeIndex(pr.start_time)

        # Grouped
        if cfg.group_by and cfg.group_by in df_long.columns:
            wide = df_long.pivot(index=cfg.x, columns=cfg.group_by, values="value")

            # 1) Stabilize columns for categorical groupers (complete missing categories)
            if cfg.group_by in cfg.data.columns and isinstance(cfg.data[cfg.group_by].dtype, pd.CategoricalDtype):
                full_cols = cfg.data[cfg.group_by].dtype.categories
                wide = wide.reindex(columns=full_cols)

            # 2) Fill time gaps on the index when requested (and x_period is defined)
            if getattr(cfg, "show_gaps", False):
                full_idx = _full_x_index(wide.index)
                if full_idx is not None:
                    wide = wide.reindex(full_idx)

            # For arithmetic and proportion, treat gaps/missing combos as 0
            if getattr(cfg, "show_gaps", False) or cfg.stacking == "proportion":
                wide = wide.fillna(0)

            if cfg.stacking == "proportion":
                rs = wide.sum(axis=1).replace({0: np.nan})
                wide = wide.div(rs, axis=0).fillna(0)

            return wide

        # Single series (no group_by): return one 'value' column indexed by x
        if cfg.x in df_long.columns:
            out = df_long.set_index(cfg.x)["value"].to_frame("value")

            if getattr(cfg, "show_gaps", False):
                full_idx = _full_x_index(out.index)
                if full_idx is not None:
                    out = out.reindex(full_idx).fillna(0)

            return out

        # Fallback
        return df_long

    def _resolve_sort_order(
        self,
        df: pd.DataFrame,
        target: str,
        strategy: Optional[Union[Literal["label", "value", "none"], Sequence[Any]]],
        ascending: Optional[bool],
        value_col: str = "value",
    ) -> pd.Index:
        """Determine most appropriate sort index"""

        if target not in df.columns:
            return pd.Index([])
        if strategy is None or strategy == "none":
            return pd.Index(pd.unique(df[target]))
        if isinstance(strategy, (list, tuple, pd.Index)):  # explicit list (append any missing labels at the end)
            avail_list = list(pd.unique(df[target]))
            requested = list(strategy)
            requested_existing = [r for r in requested if r in avail_list]
            missing_existing = [a for a in avail_list if a not in requested_existing]
            return pd.Index(requested_existing + missing_existing)
        asc = True if ascending is None else bool(ascending)
        if strategy == "label":
            return pd.Index(sorted(pd.unique(df[target]), key=lambda x: (str(x).lower() if pd.notna(x) else ""), reverse=not asc))
        if strategy == "value":
            totals = df.groupby(target, dropna=False)[value_col].sum().sort_values(ascending=asc)
            return totals.index
        return pd.Index(pd.unique(df[target]))

    def _sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort x and group_by columns with time-aware behavior.

        Time-aware behavior:
        - If the x-axis is time-like (either `cfg.x_period` is set OR the column dtype
        is datetime/period), we DO NOT coerce to an ordered Categorical. We sort
        chronologically to preserve time semantics (gap filling, date label formatting).
        - For non-time axes, we convert to an ordered Categorical when an explicit
        sort order is provided, to enforce custom category ordering and a stable legend.

        Returns a new DataFrame; does not mutate the input.
        """

        cfg = self._chart_params
        # x ordering
        if cfg.x and (cfg.x in df.columns):
            order_x = self._resolve_sort_order(df, target=cfg.x, strategy=cfg.sort_x, ascending=cfg.sort_x_ascending)
            if len(order_x):
                if cfg.x_period: # Time-like: keep dtype, sort chronologically
                    asc = True if cfg.sort_x_ascending is None else bool(cfg.sort_x_ascending)
                    df = df.sort_values(cfg.x, ascending=asc)
                else: # Nominal: enforce explicit order via ordered Categorical
                    df[cfg.x] = pd.Categorical(df[cfg.x], categories=list(order_x), ordered=True)
                    df = df.sort_values(cfg.x)
        # group_by ordering
        if cfg.group_by and (cfg.group_by in df.columns):
            order_g = self._resolve_sort_order(df, target=cfg.group_by, strategy=cfg.sort_group_by, ascending=cfg.sort_group_by_ascending)
            if len(order_g):
                df[cfg.group_by] = pd.Categorical(df[cfg.group_by], categories=list(order_g), ordered=True)
                by_cols = [c for c in [cfg.group_by, cfg.x] if c and c in df.columns]
                df = df.sort_values(by_cols)
        # names ordering (pie)
        if cfg.names and (cfg.names in df.columns):
            order_n = self._resolve_sort_order(df, target=cfg.names, strategy=cfg.sort_names, ascending=cfg.sort_names_ascending)
            if len(order_n):
                df[cfg.names] = pd.Categorical(df[cfg.names], categories=list(order_n), ordered=True)
                df = df.sort_values(cfg.names)
        return df
    
# -----------------------------------------------------------------------------
# Axes / Styling 
# -----------------------------------------------------------------------------

    def _ensure_fig_ax(self, ax: Optional[Axes] = None) -> tuple[Figure, Axes]:
        """Create an axes if one doesn't already exist."""
        if ax is not None:
            return ax.figure, ax
        else:
            fig, ax = plt.subplots(figsize=self._chart_params.fig_size)
            return fig, ax

    def _auto_rotate_xticks(self, ax: Axes) -> None:
        """Default to 90° for long labels, but keep short labels horizontal."""
        cfg = self._chart_params
        if cfg.xtick_rotation is None:
            labels = [str(t.get_text()) for t in ax.get_xticklabels()]
            if labels and all(len(lbl) <= 5 for lbl in labels):
                for label in ax.get_xticklabels():
                    label.set_rotation(0)
            elif any(len(lbl) > 5 for lbl in labels):
                for label in ax.get_xticklabels():
                    label.set_rotation(90)
        else:
            for label in ax.get_xticklabels():
                label.set_rotation(cfg.xtick_rotation)
        if cfg.ytick_rotation is not None:
            for label in ax.get_yticklabels():
                label.set_rotation(cfg.ytick_rotation)

    def _format_xperiod_labels(self, idx: pd.DatetimeIndex) -> list[str]:
        """Return human-friendly labels for x_period-driven datetime indexes."""
        cfg = self._chart_params
        if not cfg.x_period or not isinstance(idx, pd.DatetimeIndex):
            return [str(x) for x in idx]

        freq = cfg.x_period
        if freq == "Y":
            return idx.strftime("%Y").tolist()
        if freq == "Q":
            return idx.to_period("Q").astype(str).tolist()        # -> 2021Q1
        if freq == "M":
            labs = idx.strftime("%b %y").tolist()                 # -> Jan 25
            return [l.lstrip("0") for l in labs]                  # -> 1 Jan 25
        if isinstance(freq, str) and freq.startswith("W-"):
            labs = idx.strftime("%d %b %y").tolist()
            return [l.lstrip("0") for l in labs]
        if freq == "D":
            labs = idx.strftime("%d %b %y").tolist()
            return [l.lstrip("0") for l in labs]

        return [str(x) for x in idx]
    
    def _add_benchmark(self, ax: Axes):
        """Adds a benchmark line"""
        cfg = self._chart_params

        # Benchmark horizontal line
        if cfg.target_y is not None:
            ax.axhline(
                y=cfg.target_y,
                linestyle="--",
                color="black",
                linewidth=1,
                label=cfg.target_y_label or None,
            )

        # Benchmark vertical line
        if cfg.target_x is not None:
            ax.axvline(
                x=cfg.target_x,
                linestyle="--",
                color="black",
                linewidth=1,
                label=cfg.target_x_label or None,
            )

    def _apply_common_style(self, fig: Figure, ax: Axes, chart_data: Optional[pd.DataFrame] = None) -> Tuple[Figure, Axes]:
        """Apply axis labels, axis limits, axis formats, tick rotation, grid"""
        cfg = self._chart_params

        # Axis labels
        if cfg.x_label: # use the user passed label with priority
            ax.set_xlabel(cfg.x_label)
        elif cfg.x_label == '': # allow user to pass '' to suppress the label
            pass
        elif cfg.x is not None: # automatically use the x variable as the label
            ax.set_xlabel(str(cfg.x))
        
        if cfg.y_label: # use the user passed label with priority
            ax.set_ylabel(cfg.y_label)
        elif cfg.y_label == '': # allow user to pass '' to suppress the label
            pass
        elif cfg.y is None and cfg.group_by is None: # default label for counts
            ax.set_ylabel("Number of rows in data")
        elif cfg.y is not None: # use y variable if it exists
            ax.set_ylabel(str(cfg.y))

        # Axis label padding
        if ax.get_xlabel():
            ax.xaxis.labelpad = 6
        if ax.get_ylabel():
            ax.yaxis.labelpad = 6
            
        # Axis font sizes
        ax.xaxis.label.set_size(_resolve_fontsize(cfg.x_label_size, 'x_label'))
        ax.yaxis.label.set_size(_resolve_fontsize(cfg.y_label_size, 'y_label'))
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(_resolve_fontsize(cfg.tick_size, 'tick'))

        # Ranges
        if cfg.x_min is not None or cfg.x_max is not None:
            ax.set_xlim(left=cfg.x_min if cfg.x_min is not None else ax.get_xlim()[0],
                        right=cfg.x_max if cfg.x_max is not None else ax.get_xlim()[1])
        if cfg.y_min is not None or cfg.y_max is not None:
            ax.set_ylim(bottom=cfg.y_min if cfg.y_min is not None else ax.get_ylim()[0],
                        top=cfg.y_max if cfg.y_max is not None else ax.get_ylim()[1])

        # Numeric formatting
        _format_axis(ax, "x", cfg.x_axis_format, cfg.decimals)
        _format_axis(ax, "y", cfg.y_axis_format, cfg.decimals)

        # Pretty x-period tick labels
        if cfg.x_period and chart_data is not None and isinstance(chart_data.index, pd.DatetimeIndex):
            new_labels = self._format_xperiod_labels(chart_data.index)
            ax.set_xticks(np.arange(len(new_labels))) # avoid "FixedFormatter without FixedLocator" warning: set tick positions before labels
            ax.set_xticklabels(new_labels)
    
        # Tick rotation + grid
        self._auto_rotate_xticks(ax)
        if cfg.grid_x:
            ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        else:
            ax.grid(False, axis="x")
        if cfg.grid_y:
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        else:
            ax.grid(False, axis="y")

        # Benchmark lines
        self._add_benchmark(ax)

        return fig, ax
    
# -----------------------------------------------------------------------------
# Figure / Legend Layout 
# -----------------------------------------------------------------------------

    def _legend_wrap(self, fig: Figure, labels: Sequence[str], max_ratio: float = 0.3) -> Sequence[str]:
        """
        Automatically wraps legend labels so they fit within max_ratio of figure width.
        max_ratio is the total fraction of figure width allocated to each legend entry 
        (including handle + padding). 
        """

        # determine size of legend in px
        dpi = fig.dpi
        fontsize = FontProperties(size=rcParams.get('legend.fontsize', 10)).get_size_in_points()
        em_to_px = fontsize * dpi / 72.0
        handle_px = rcParams.get('legend.handlelength', 2.0) * em_to_px
        pad_px = rcParams.get('legend.handletextpad', 0.8) * em_to_px
        handle_pad_px = handle_px + pad_px

        # determine maximum width of legend text in px
        fig_width_px = fig.get_size_inches()[0] * fig.dpi
        max_label_px = fig_width_px * max_ratio - handle_pad_px

        # determine maximum character equivalent
        char_px = em_to_px * 0.6
        max_chars = int(max_label_px // char_px)

        # wrap labels according to the maximum characters
        if max_chars <= 0 or not labels:
            return labels
        if max(len(lbl) for lbl in labels) <= max_chars:
            return labels
        return ['\n'.join(wrap(lbl, max_chars)) for lbl in labels]

    def _legend_ncol(self, fig: Figure, ax: Axes, legend_pos: str = "right", max_ratio: float = 0.9) -> int:
        """Determine optimal number of legend columns to ensure legend stays within the 
        height (right legend) / width (bottom legend) of the data axes and uses
        extra cols (right legend) / rows (bottom legend) in order to do so."""

        handles, labels = ax.get_legend_handles_labels()
        if not labels:
            return 1
        
        dpi = fig.dpi
        
        if legend_pos == 'right':
            fig_height_px = fig.get_size_inches()[1] * dpi
            max_height_px = fig_height_px * max_ratio

            # use a temporary 1-col legend to get per label height
            temp_leg = fig.legend(handles, labels, ncol=1, loc='upper right', frameon=False)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bb = temp_leg.get_window_extent(renderer=renderer)
            temp_leg.remove()

            total_height_px = bb.height
            avg_height_px = total_height_px / len(labels)
            max_rows = max(1, int(max_height_px // avg_height_px))
            if max_rows >= len(labels):
                return 1
            else:
                return ceil(len(labels) / max_rows)
        
        elif legend_pos == 'bottom':
            fig_width_px = fig.get_size_inches()[0] * dpi
            max_width_px = fig_width_px * max_ratio

            # measure a temporary 1-row legend to get per-label width
            temp_leg = fig.legend(handles, labels, ncol=len(labels), loc='upper right', frameon=False)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bb = temp_leg.get_window_extent(renderer=renderer)
            temp_leg.remove()

            total_width_px = bb.width
            avg_width_px = total_width_px / len(labels)
            max_cols = max(1, int(max_width_px // avg_width_px)) # e.g. 4
            if max_cols >= len(labels):
                return len(labels)
            else: # distribute labels more evenly across columns e.g. instead of 4 + 1 let's have 3 + 2
                max_rows = ceil(len(labels) / max_cols) # e.g. 5 / 4 -> 2
                return ceil(len(labels) / max_rows) # e.g. 5 / 2 -> 3

        else:
            return 1
        
    def _title_wrap(self, fig: Figure, title: str, fontsize: int, max_ratio: float = 0.9) -> str:
        """Wrap title text so it fits within the target % figure width."""

        fig_w_px = fig.get_size_inches()[0] * fig.dpi # figure width in pixels
        char_px = fontsize * fig.dpi / 72.0 * 0.6 # character width in pixels
        max_chars = int((fig_w_px * max_ratio) // char_px) # maximum characters in % of figure width
        return "\n".join(wrap(title, max_chars))
        
    def _finalise_layout(self, fig: Figure, ax: Axes) -> None:
        """
        Adjust figure to accomodate legend and title
        
        Notes
        - Wraps right legend entries
        - Hides legends with only one label
        - Increase figure size to accomodate legend and title (using a temporary legend and fig.text to estimate required space)
        - Title (figure suptitle) centers over the axes area, not the figure, so it never spans the right legend region
        - Subtitle (axis title) is naturally aligned to the axes.        
        """

        # Config
        cfg = self._chart_params
        legend = cfg.legend or "none"
        title_fontsize = _resolve_fontsize(cfg.title_size, "title")
        subtitle_fontsize = _resolve_fontsize(cfg.subtitle_size, "subtitle")

        # Legend entries
        handles, labels = ax.get_legend_handles_labels()
        labels = [l for l in labels if l and l != "_nolegend_"]
        
        # Hide legend if no useful labels (e.g. single series "value")
        if not labels or len(set(labels)) <= 1:
            legend = "none"

        # Wrap right legend labels
        if labels and legend == "right":
            labels = list(self._legend_wrap(fig, labels))

        # Wrap title/subtitle
        if cfg.title:
            title_wrap = self._title_wrap(fig, cfg.title, title_fontsize)
        if cfg.subtitle:
                subtitle_wrap = self._title_wrap(fig, cfg.subtitle, subtitle_fontsize)

        # If there's no legend, add title/subtitle and return
        if legend == "none" or not labels:
            if cfg.title:
                fig.suptitle(title_wrap, fontsize=title_fontsize)
            if cfg.subtitle:
                ax.set_title(subtitle_wrap, fontsize=subtitle_fontsize)
            fig.tight_layout()
            fig.canvas.draw()
            return
        
        # Default legend title to group_by if not explicitly set
        if cfg.legend_label is None and cfg.group_by is not None:
            cfg.legend_label = cfg.group_by

        # Create a temporary legend and title to measure required space
        ncol = self._legend_ncol(fig, ax, legend_pos=legend)
        temp_legend = fig.legend(handles, labels, ncol=ncol, loc="upper right", title=cfg.legend_label or None)
        if cfg.title:
            temp_title = fig.text(0, 1, title_wrap, fontsize=title_fontsize)

        # Initialise the renderer
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # Check required space
        fig_width, fig_height = fig.get_size_inches() # inches
        dpi = fig.dpi # convert from pixels to inches
        legend_bbox = temp_legend.get_window_extent(renderer=renderer) # pixels
        legend_height = legend_bbox.height / dpi # inches
        legend_width = legend_bbox.width / dpi # inches 
        temp_legend.remove()
        if cfg.title:
            title_bbox = temp_title.get_window_extent(renderer=renderer) # pixels
            title_height = title_bbox.height / dpi # inches
            temp_title.remove()
        else:
            title_height = 0
        
        # Expand the figure to accomodate the legend and title
        if cfg.resize_fig: # TODO: consider removing this and/or test if False and/or consider if any of the later sections should fall into this conditional
            if legend == "right":
                new_fig_width = fig_width + legend_width
                new_fig_height = fig_height + title_height
            elif legend == "bottom":
                new_fig_width = fig_width
                new_fig_height = fig_height + legend_height + title_height
            fig.set_size_inches(new_fig_width, new_fig_height)
            fig.canvas.draw() # TODO: test removing this line

        # Determine the position of the legend (loc, bbox) and reserve legend/title space using tight_layout rect
        # rect : tuple (left, bottom, right, top), default: (0, 0, 1, 1)
        # "A rectangle in normalized figure coordinates into which the whole subplots area (including labels) will fit.""
        if legend == "right": # right legend
            loc = "center right"
            bbox = (1.0, 0.5)
            rect = [0, 0, 1 - legend_width / new_fig_width, 1 - title_height / new_fig_height]
        else: # bottom legend
            loc = "lower center"
            bbox = (0.5, 0.0)
            rect = [0, legend_height / new_fig_height, 1, 1 - title_height / new_fig_height]

        # Add subtitle and use tight_layout to finalise axes position
        if cfg.subtitle:
            ax.set_title(subtitle_wrap, fontsize=subtitle_fontsize)
        fig.tight_layout(rect=rect)
        if cfg.title:
            # Adjust x coordinate of title to center it over the axes
            axes_box = ax.get_position()
            title_x = axes_box.x0 + axes_box.width / 2.0
            fig.suptitle(title_wrap, fontsize=title_fontsize, x=title_x)
        fig.legend(handles, labels, ncol=ncol, loc=loc, bbox_to_anchor=bbox, title=cfg.legend_label or None)
        fig.canvas.draw()

# -----------------------------------------------------------------------------
# Chart Primitives                                                           
# -----------------------------------------------------------------------------

    # TODO: work through the list of options in .plot and wrap them into this framework
    # 'line', 'bar', 'barh', 'hist', 'box', 'kde', 'density', 'area', 'pie', 'scatter', 'hexbin'

    def _bar(self, df: pd.DataFrame, ax: Optional[Axes] = None) -> tuple[Figure, Axes, pd.DataFrame]:
        cfg = self._chart_params
        fig, ax = self._ensure_fig_ax(ax)
        chart_data = self._pivot_data(df)
        chart_data.plot(kind="bar",
                        stacked=(cfg.stacking != "none"),
                        color=_resolve_palette(cfg.palette),
                        ax=ax,
                        legend=False)
        return fig, ax, chart_data

    def _line(self, df: pd.DataFrame, ax: Optional[Axes] = None) -> tuple[Figure, Axes, pd.DataFrame]:
        """
        Line chart using _pivot_data. Returns (fig, ax, chart_data).

        - chart_data is wide for grouped lines (index=cfg.x, columns=cfg.group_by),
        or a single 'value' column (index=cfg.x) for a single series.
        """
        cfg = self._chart_params
        if cfg.x is None:
            raise ValueError("Line charts require `x`.")

        fig, ax = self._ensure_fig_ax(ax)

        # Long -> wide (and handle gaps/percent if configured)
        chart_data = self._pivot_data(df)

        # Draw (DataFrame.plot handles both single and multi-column cases)
        chart_data.plot(ax=ax,
                        marker="o",
                        color=_resolve_palette(cfg.palette),
                        legend=False)

        return fig, ax, chart_data

    def _pie(self, df: pd.DataFrame, ax: Optional[Axes] = None) -> tuple[Figure, Axes, pd.DataFrame]:
        """
        Pie chart that returns (fig, ax, chart_data).

        - chart_data is a two-column table [names, value] (indexed by names for convenience).
        - Honors cfg.show_labels / cfg.show_percent.
        """
        cfg = self._chart_params
        fig, ax = self._ensure_fig_ax(ax)

        # Determine names column
        names_col = cfg.names or (cfg.x if (cfg.x and cfg.x in df.columns) else None)
        if not names_col:
            raise ValueError("Pie charts require `names` (or `x`) plus `value`.")
        if "value" not in df.columns:
            raise ValueError("Pie charts require a numeric 'value' column.")

        # Build chart_data (already aggregated/sorted upstream)
        chart_data = df[[names_col, "value"]].copy().set_index(names_col)

        # Labels / percentages
        labels = chart_data.index.astype(str).values if cfg.show_labels else None
        if cfg.show_percent is True:
            autopct = "%1.0f%%"
        elif isinstance(cfg.show_percent, str):
            autopct = cfg.show_percent
        else:
            autopct = None

        # Draw
        wedges, texts, autotexts = ax.pie(
            chart_data["value"].values,
            labels=labels,
            colors=_resolve_palette(cfg.palette),
            autopct=autopct,
            startangle=90,
            counterclock=False,
        )

        # Use legend instead of wedge labels if requested
        if not cfg.show_labels:
            ax.legend(
                wedges,
                chart_data.index.astype(str).tolist(),
                title=cfg.legend_label or None
            )

        ax.axis("equal")
        return fig, ax, chart_data

# -----------------------------------------------------------------------------
# Public API                                                           
# -----------------------------------------------------------------------------

    def _save_and_return(self, fig, ax, chart_data) -> Optional[tuple[Figure, Axes, pd.DataFrame]]:
        cfg = self._chart_params
        if cfg.save_path:
            fig.savefig(cfg.save_path, bbox_inches="tight", dpi=fig.dpi)
        if cfg.show_fig:
            plt.show()
        if cfg.return_values or not cfg.show_fig:
            return fig, ax, chart_data
        else:
            return None

    @add_docstring(COMMON_DOCSTRING)
    def bar(self, **kwargs: Any) -> Optional[tuple[Figure, Axes, pd.DataFrame]]:
        """Bar chart (counts by default, aggregate `y` when provided)."""

        self._set_params(**kwargs)
        chart_data = self._aggregate_data()
        chart_data = self._sort(chart_data)
        fig, ax, chart_data = self._bar(chart_data)
        self._apply_common_style(fig, ax, chart_data)
        self._finalise_layout(fig, ax)
        return self._save_and_return(fig, ax, chart_data)

    @add_docstring(COMMON_DOCSTRING)
    def line(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Line chart. Requires `x`."""
        self._set_params(**kwargs)
        agg = self._sort(self._aggregate_data())
        fig, ax, chart_data = self._line(agg)
        self._apply_common_style(fig, ax, chart_data)
        self._finalise_layout(fig, ax)
        return self._save_and_return(fig, ax, chart_data)

    @add_docstring(COMMON_DOCSTRING)
    def pie(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Pie chart. Requires `names`+`values` (or `x` as names + counts)."""
        self._set_params(**kwargs)
        agg = self._sort(self._aggregate_data())
        fig, ax, chart_data = self._pie(agg)
        self._apply_common_style(fig, ax)
        # hide axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")
        self._finalise_layout(fig, ax)
        return self._save_and_return(fig, ax, chart_data)
    
# -----------------------------------------------------------------------------
# Advanced charts
# -----------------------------------------------------------------------------

