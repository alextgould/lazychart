from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union, Sequence, Tuple, Dict, Callable, Literal, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
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
    legend_title : str, optional
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
    other_label : str, default 'Other'
        Label for grouped small categories.

    Other
    --------------------
    sticky : bool, default False
        Whether to store arguments passed for the current chart for future charts
    use_sticky : bool, default True
        Whether to apply existing sticky arguments to the current chart
    palette : str | Sequence[str], optional
        List of hex colour codes to use for this chart. Use set_palette to apply this to all charts.
    palette_map : dict, optional
        Mapping of labels to hex colour codes to use for this chart. Use set_palette to apply this to all charts.
    target_x : float, optional
        Adds a vertical dotted line at a given x value
    target_y : float, optional
        Adds a horizontal dotted line at a given y value
    target_x_label : str, optional
    target_y_label : str, optional

    Notes
    -----
    Several aliases from matplotlib/pandas/seaborn/plotly are accepted for
    user convenience, e.g. `xlabel` → `x_label`, `xlim=(min,max)` →
    `x_min`/`x_max`, `legend_title` → `legend_label`. 
    Similarly some plurals may work, e.g. `labels` → `label`
    Only canonical names are documented above.
    """

# -----------------------------------------------------------------------------
# Public-API aliases (ergonomics for users coming from matplotlib/seaborn/pandas/plotly)
# -----------------------------------------------------------------------------

# Simple one-to-one key renames
ALIASES: Dict[str, str] = {
    # Labels / titles
    "xlabel": "x_label",
    "ylabel": "y_label",
    "legend_title": "legend_label",     # canonical stays legend_label internally

    # Sizing / ticks
    "figsize": "fig_size",
    "rot": "xtick_rotation",

    # Grouping / aggregation
    "hue": "group_by",
    "estimator": "aggfunc",

    # Legend synonyms
    "legend_pos": "legend",
    "legend_position": "legend",
    "loc": "legend",                    # common matplotlib muscle memory

    # Sorting synonyms
    "x_sort": "sort_x",
    "x_order": "sort_x",
    "group_sort": "sort_group_by",
    "group_by_sort": "sort_group_by",
    "hue_sort": "sort_group_by",
    "hue_order": "sort_group_by",
    "names_sort": "sort_names",
    "names_order": "sort_names",
    "label_sort": "sort_names",

    # Sorting ascending flags
    "x_sort_ascending": "sort_x_ascending",
    "group_sort_ascending": "sort_group_by_ascending",
    "group_by_sort_ascending": "sort_group_by_ascending",
    "hue_sort_ascending": "sort_group_by_ascending",
    "names_sort_ascending": "sort_names_ascending",

    # Colors / palettes
    "labels_map": "label_map",
    "label_mapping": "label_map",
    "category_map": "label_map",
}

# Composite aliases that expand into multiple canonical keys
COMPOSITE_ALIASES: Dict[str, Tuple[str, str]] = {
    "xlim": ("x_min", "x_max"),
    "ylim": ("y_min", "y_max"),
    # Optional: vertical/horizontal reference lines (friends from mpl)
    "vline": ("target_x", None),            # single float → target_x
    "hline": (None, "target_y"),            # single float → target_y
    "vline_label": ("target_x_label", None),
    "hline_label": (None, "target_y_label"),
}

# Value maps for common synonyms
_STACKING_SYNONYMS = {
    "stack": "stacked",
    "stacked": "stacked",
    "group": "none",          # grouped/dodged bars
    "dodge": "none",
    "none": "none",
    "relative": "proportion", # 100% stacked
    "fill": "proportion",
    "proportion": "proportion",
}

# barmode→stacking (plotly-ism)
_BARMODE_TO_STACKING = {
    "stack": "stacked",
    "group": "none",
    "overlay": "none",
    "relative": "proportion",
}

def _normalize_legend_value(value: Any) -> Optional[Literal["right", "bottom", "none"]]:
    """Coerce a variety of legend inputs into our LegendPosition or None (meaning 'leave default')."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "right" if value else "none"
    v = str(value).strip().lower()
    if v in {"none", "off", "hide", "false", "0"}:
        return "none"
    if v in {"right", "r"}:
        return "right"
    if v in {"bottom", "b"}:
        return "bottom"
    # Common matplotlib locs → a reasonable position
    if "right" in v:
        return "right"
    if "lower" in v or "bottom" in v:
        return "bottom"
    # 'best', 'upper left', 'center', etc. → default to right
    return "right"

def _normalize_kwargs(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize user-facing kwargs into canonical ChartConfig keys/values.
    Precedence: canonical key in `raw` wins over any alias for that key.
    Unknown keys are dropped.
    """
    if not raw:
        return {}

    # Start with a shallow copy to avoid mutating caller data
    kw = dict(raw)

    # Canonical keys taken from dataclass fields
    CANONICAL_KEYS = set(ChartConfig.__dataclass_fields__.keys())  # type: ignore[attr-defined]

    normalized: Dict[str, Any] = {}

    # 1) First, copy through any canonical keys as-is
    for k, v in kw.items():
        if k in CANONICAL_KEYS:
            normalized[k] = v

    # 2) Expand composite aliases like xlim/ylim, vline/hline
    for alias, targets in COMPOSITE_ALIASES.items():
        if alias in kw:
            v = kw[alias]
            left_key, right_key = targets
            # (xlim, ylim): expect a 2-length sequence
            if alias in ("xlim", "ylim"):
                try:
                    a, b = (v[0], v[1]) if isinstance(v, (list, tuple)) and len(v) >= 2 else (None, None)
                except Exception:
                    a, b = (None, None)
                if left_key and left_key not in normalized and a is not None:
                    normalized[left_key] = a
                if right_key and right_key not in normalized and b is not None:
                    normalized[right_key] = b
            else:
                # vline/hline singletons or labels
                if left_key and left_key not in normalized and v is not None:
                    normalized[left_key] = v
                if right_key and right_key not in normalized and v is not None:
                    normalized[right_key] = v

    # 3) Apply one-to-one key aliases (only if canonical missing)
    for alias, canon in ALIASES.items():
        if alias in kw and canon not in normalized:
            normalized[canon] = kw[alias]

    # 4) Value coercion / special cases

    # legend (accept via 'legend' or its aliases handled above)
    if "legend" in normalized or any(a in kw for a in ("legend", "legend_pos", "legend_position", "loc")):
        # Prefer the canonical key value if present, else fall back to the first alias provided
        raw_val = (kw.get("legend", kw.get("legend_pos", kw.get("legend_position", kw.get("loc")))))
        val = _normalize_legend_value(normalized.get("legend", raw_val))
        if val is not None:
            normalized["legend"] = val

    # stacking: accept synonyms
    if "stacking" in kw or "stacking" in normalized:
        val = normalized.get("stacking", kw.get("stacking"))
        if isinstance(val, str):
            normalized["stacking"] = _STACKING_SYNONYMS.get(val.lower(), val)

    # barmode → stacking (only if stacking not explicitly set)
    if "barmode" in kw and "stacking" not in normalized:
        bm = str(kw["barmode"]).lower()
        mapped = _BARMODE_TO_STACKING.get(bm)
        if mapped:
            normalized["stacking"] = mapped

    # If user provided seaborn-style *order= sequences, pass them through to our sort_*
    # (already mapped by ALIASES above). Nothing more to do here.

    # 5) Drop anything not part of ChartConfig (defensive)
    normalized = {k: v for k, v in normalized.items() if k in CANONICAL_KEYS}

    return normalized

# -----------------------------------------------------------------------------
# Palette presets
# -----------------------------------------------------------------------------

FALLBACK_PALETTE = [
    "#0F4C81",  # deep blue (Pantone Classic Blue-ish)
    "#EE6352",  # coral red
    "#59CD90",  # emerald
    "#3FA7D6",  # azure
    "#FAC05E",  # saffron
    "#8F2D56",  # mulberry
    "#5B5F97",  # indigo gray
    "#2EC4B6",  # turquoise
    "#9F86C0",  # lavender
    "#BC4749",  # brick red
    "#6A994E",  # olive green
    "#386641",  # forest green
]

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
    other_label: str = "Other"

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
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    target_x_label: Optional[str] = None
    target_y_label: Optional[str] = None

    # ---- Chart level palette overrides ---
    palette : Optional[Sequence[str]] = None
    palette_map : Dict[Any, str] = field(default_factory=dict)

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
# Compare API helper
# -----------------------------------------------------------------------------

class ChartSpec:
    """
    A proxy for an individual chart which is part of a compare() chart
    e.g. cm.chart.bar (rather than cm.bar for a standalone chart).
     
    The ChartSpec class is used to let us pass functions without messy lambdas i.e.

        cm.compare(
            cm.chart.bar(x="alcohol", subtitle="Alcohol histogram"),
            cm.chart.bar(x="weekday", subtitle="Weekday histogram")
        )

    instead of 
    
        cm.compare(
            lambda ax: cm.bar(ax=ax, x="alcohol", finalize=False),
            lambda ax: cm.bar(ax=ax, x="weekday", finalize=False),
        )
    """

    def __init__(self, cm: "ChartMonkey", kind: Optional[str] = None, kwargs: Optional[Dict[str, Any]] = None):
        self._cm = cm
        self.kind = kind
        self.kwargs = kwargs or {}

    def __getattr__(self, name: str):
        """Builds a concrete spec when used on the root "cm.chart" object.
        A spec is a recipe: it stores *what* to draw (`kind`) and *how* to draw it (`kwargs`).
        It does **not** draw until `compare()` calls `.render(...)`
        """
        def _factory(**kwargs: Any) -> "ChartSpec":
            return ChartSpec(self._cm, name, kwargs)
        return _factory

    def render(self, ax: Axes):
        """API for drawing onto a Matplotlib Axes using the corresponding
        public method (e.g. `cm.bar`) with `finalize=False`
        
        Returns fig, ax, chart_data
        """
        method = getattr(self._cm, self.kind)
        return method(ax=ax, finalize=False, **self.kwargs)

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
    """Apply percent, comma and short formats if requested"""
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
        
        # Config
        self._sticky: Dict[str, Any] = {}
        self._chart_params: Optional[ChartConfig] = None
        
        # Palette
        self.palette: Optional[Sequence[str]] = None
        self.palette_map: Dict[Any, str] = {}
        if palette is not None:
            self.set_palette(palette)

        # Compare API helper
        self.chart = ChartSpec(self)

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
        """Merge sticky + kwargs into a `ChartConfig`, normalizing aliases, and store on the instance."""

        # pop control flags and palette overrides first (do not forward to ChartConfig)
        sticky = kwargs.pop("sticky", False)
        use_sticky = kwargs.pop("use_sticky", True)
        palette = kwargs.pop("palette", None)
        palette_map = kwargs.pop("palette_map", None)

        # normalize the incoming kwargs before doing anything else
        norm_kwargs = _normalize_kwargs(kwargs)

        # If caller asked to make these sticky, store the *normalized* keys
        if sticky:
            self.sticky(kwargs=norm_kwargs)

        # Start from existing sticky (optionally) and normalize it too (for backward safety)
        base = dict(getattr(self, "_sticky", {})) if use_sticky else {}

        # Later kwargs override earlier sticky values
        config_values = {**base, **norm_kwargs}

        # Build and keep the current chart config
        config = ChartConfig(**config_values)

        # chart specific palette overrides
        if palette is not None:
            config.palette = self._coerce_palette_input(palette)
        if palette_map is not None:
            config.palette_map = dict(palette_map)
            
        self._chart_params = config
        return config

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
        other_name = cfg.other_label or "Other"
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

            # Remap labels (group_by → column headers)
            if cfg.label_map:
                wide = wide.rename(columns=cfg.label_map)

            # Remap x index labels (x → index values) for non-time axes
            if cfg.label_map and not (cfg.x_period or isinstance(wide.index, (pd.DatetimeIndex, pd.PeriodIndex))):
                wide.index = wide.index.map(lambda v: cfg.label_map.get(v, v))

            return wide

        # Single series (no group_by): return one 'value' column indexed by x
        if cfg.x in df_long.columns:
            out = df_long.set_index(cfg.x)["value"].to_frame("value")

            # Remap labels
            if cfg.label_map and not (cfg.x_period or isinstance(out.index, pd.DatetimeIndex)):
                out.index = out.index.map(lambda v: cfg.label_map.get(v, v))

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

        # sort target does not exist
        if target not in df.columns:
            return pd.Index([])
        
        # sort strategy does not exist
        if strategy is None or strategy == "none":
            uniq = pd.unique(df[target])

            # If ascending explicitly provided, do a sensible default sort
            if ascending is not None:
                asc = bool(ascending)
                # numeric-friendly sort when possible
                if pd.api.types.is_numeric_dtype(df[target]):
                    return pd.Index(sorted(uniq, key=lambda x: (float(x) if pd.notna(x) else float("inf")), reverse=not asc))
                # otherwise string sort
                return pd.Index(sorted(uniq, key=lambda x: (str(x).lower() if pd.notna(x) else ""), reverse=not asc))
            
            return pd.Index(uniq)
        
        # Explicit list strategy
        if isinstance(strategy, (list, tuple, pd.Index)):
            avail_list = list(pd.unique(df[target]))
            requested = list(strategy)
            requested_existing = [r for r in requested if r in avail_list]

            # also accept mapped labels
            cfg = self._chart_params
            if cfg.label_map:
                rev = {v: k for k, v in cfg.label_map.items()}
                requested = [rev.get(r, r) for r in requested]

            # append missing labels at the end
            missing_existing = [a for a in avail_list if a not in requested_existing]
            return pd.Index(requested_existing + missing_existing)
        asc = True if ascending is None else bool(ascending)

        # normalize strategy strings (accept plural synonyms)
        if isinstance(strategy, str):
            key = strategy.strip().lower()
            if key in {"labels", "names"}:
                strategy = "label"
            elif key in {"values", "totals", "sum"}:
                strategy = "value"
            else:
                strategy = key

        # apply label or value sort strategies
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
            # treat as time if x_period is set OR the dtype is already datetime/period
            is_time = bool(cfg.x_period) or pd.api.types.is_datetime64_any_dtype(df[cfg.x]) or isinstance(df[cfg.x].dtype, pd.PeriodDtype)
            if is_time:
                asc = True if cfg.sort_x_ascending is None else bool(cfg.sort_x_ascending)
                df = df.sort_values(cfg.x, ascending=asc)
            else:
                # nominal - enforce explicit order via ordered Categorical
                order_x = self._resolve_sort_order(df, target=cfg.x, strategy=cfg.sort_x, ascending=cfg.sort_x_ascending)
                if len(order_x):
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
# Axes styling
# -----------------------------------------------------------------------------

    def _ensure_fig_ax(self, ax: Optional[Axes] = None) -> tuple[Figure, Axes]:
        """Create an axes if one doesn't already exist."""
        if ax is not None:
            return ax.figure, ax
        else:
            fig, ax = plt.subplots(figsize=self._chart_params.fig_size)
            return fig, ax

    def _rotate_xticks(self, ax: Axes) -> None:
        """Apply rotation as requested by the user.
        For the x axis, default to 90° for long labels and horizontal for short labels."""
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
        """Return human-friendly labels for x_period-driven datetime indexes with bar charts."""
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
    
    def _add_benchmark(self, ax: Axes, chart_data: pd.DataFrame):
        """Adds a benchmark line"""
        cfg = self._chart_params

        # Horizontal line
        if cfg.target_y is not None:
            ax.axhline(y=cfg.target_y, linestyle="--", color="black", linewidth=1, label=cfg.target_y_label or None)

        # Vertical line
        if cfg.target_x is not None:
            idx = chart_data.index
            tx = None
            
            # datetime - coerce to target_x datetime
            if isinstance(idx, pd.DatetimeIndex):
                try:
                    tx = pd.to_datetime(cfg.target_x)
                except Exception:
                    tx = None

            # categorical/object index - map to ordinal position
            elif cfg.target_x in chart_data.index:
                tx = int(idx.get_loc(cfg.target_x))

            # categorical/object index as str - map to ordinal position
            elif str(cfg.target_x) in chart_data.index:
                tx = int(idx.get_loc(str(cfg.target_x)))
            
            # onus is on the caller to figure this out
            else:
                tx = cfg.target_x
                
            if tx is not None:
                ax.axvline(x=tx, linestyle="--", color="black", linewidth=1, label=cfg.target_x_label or None)

    def _style_axes(self, fig: Figure, ax: Axes, chart_data: Optional[pd.DataFrame] = None):
        """Apply axis labels, axis limits, axis formats, tick rotation, grid"""
        cfg = self._chart_params

        # x axis label
        if cfg.x_label: # use the user passed label with priority
            ax.set_xlabel(cfg.x_label)
        elif cfg.x_label == '': # allow user to pass '' to suppress the label
            pass
        elif cfg.x is not None: # automatically use the x variable as the label
            if cfg.x_period:
                if cfg.x_period == "Q":
                    ax.set_xlabel(f"Quarter ({cfg.x})")
                elif cfg.x_period == "M":
                    ax.set_xlabel(f"Month ({cfg.x})")
                elif cfg.x_period == "Y":
                    ax.set_xlabel(f"Year ({cfg.x})")
                elif isinstance(cfg.x_period, str) and cfg.x_period.startswith("W-"):
                    ax.set_xlabel(f"Week ({cfg.x})")
                else:
                    ax.set_xlabel(f"{cfg.x_period} ({cfg.x})")
            else:
                ax.set_xlabel(str(cfg.x))
        
        # y axis label
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

        # Apply percent, comma and short formats if requested
        _format_axis(ax, "x", cfg.x_axis_format, cfg.decimals)
        _format_axis(ax, "y", cfg.y_axis_format, cfg.decimals)

        # Index min/max range, including individual treatment for date/categorical/numeric x axis 
        idx = chart_data.index
        is_dt = isinstance(idx, pd.DatetimeIndex)
        is_cat = (not is_dt) and (idx.dtype == object or pd.api.types.is_categorical_dtype(idx))

        if is_dt: # datetime x
            def _as_dt(v):
                if v is None:
                    return None
                try:
                    return pd.to_datetime(v)
                except Exception:
                    return v
            x_min, x_max = _as_dt(cfg.x_min), _as_dt(cfg.x_max)

        elif is_cat: # categorical - map to ordinal positions (0..N-1)
            def _cat_pos(v):
                if v is None:
                    return None
                if v in idx:
                    return int(idx.get_loc(v))
                if str(v) in idx:
                    return int(idx.get_loc(str(v)))
                return None
            
            # add a margin on either side
            x_min, x_max = _cat_pos(cfg.x_min), _cat_pos(cfg.x_max)
            if x_min is None:
                x_min = 0
            if x_max is None:
                x_max = len(idx) - 1
            x_min = x_min - 0.5
            x_max = x_max + 0.5

        else:
            # numeric axis
            def _as_num(v):
                if v is None:
                    return None
                try:
                    return float(v)
                except Exception:
                    return v
            x_min, x_max = _as_num(cfg.x_min), _as_num(cfg.x_max)

        if x_min is not None or x_max is not None:
            ax.set_xlim(left=x_min if x_min is not None else ax.get_xlim()[0],
                        right=x_max if x_max is not None else ax.get_xlim()[1])
        if cfg.y_min is not None or cfg.y_max is not None:
            ax.set_ylim(bottom=cfg.y_min if cfg.y_min is not None else ax.get_ylim()[0],
                        top=cfg.y_max if cfg.y_max is not None else ax.get_ylim()[1])

        # x ticks
        # use set_xticks and set_xticklabels rather than locators and formatters
        # as the former feels better with our tentency to pre-summarise data (guaranteed tick for each point)
        # and seems to be more reliable (no odd 1970 epoch formatting issues)
        if is_dt:
            if cfg.x_period:
                if ax.lines: # Line chart → keep datetime positions, use locators + formatters
                    ax.set_xticks(idx)
                    if cfg.x_period == "Q":
                        ax.set_xticklabels([f"{d.year}Q{((d.month-1)//3)+1}" for d in idx])
                    elif cfg.x_period == "M":
                        ax.set_xticklabels([d.strftime("%b %y") for d in idx])
                    elif cfg.x_period == "Y":
                        ax.set_xticklabels([d.strftime("%Y") for d in idx])
                    elif isinstance(cfg.x_period, str) and cfg.x_period.startswith("W-"):
                        ax.set_xticklabels([d.strftime("%d %b %y").lstrip("0") for d in idx])
                    else:  # Day or anything unusual → fall back
                        ax.set_xticklabels([d.strftime("%d %b %y").lstrip("0") for d in idx])

                else: # bar chart → ordinal bins
                    new_labels = self._format_xperiod_labels(chart_data.index)
                    locs = np.arange(len(new_labels))
                    ax.set_xticks(locs)
                    ax.set_xticklabels(new_labels)

            else: # No x_period
                loc = mdates.AutoDateLocator()
                ax.xaxis.set_major_locator(loc)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

        #ax.margins(x=0.02)
        
        # Tick rotation
        self._rotate_xticks(ax)

        # Grid
        if cfg.grid_x:
            ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        else:
            ax.grid(False, axis="x")
        if cfg.grid_y:
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        else:
            ax.grid(False, axis="y")

        # Target/benchmark lines
        self._add_benchmark(ax, chart_data)

        return fig, ax
    
# -----------------------------------------------------------------------------
# Figure styling 
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
        
    def _style_fig(self, fig: Figure, axes: Union[Axes, Sequence[Axes]]):
        """
        Shared legend + title + layout for one or multiple axes.
        
        Notes
        - Wraps right legend entries
        - Hides legends with only one label
        - Increase figure size to accomodate legend and title (using a temporary legend and fig.text to estimate required space)
        - Title (figure suptitle) centers over the axes area, not the figure, so it never spans the right legend region
        - Subtitle (axis title) is naturally aligned to the axes.  
        """
        cfg = self._chart_params
        legend_pos = cfg.legend or "none"
        title_fontsize = _resolve_fontsize(cfg.title_size, "title")
        subtitle_fontsize = _resolve_fontsize(cfg.subtitle_size, "subtitle")

        # Collect handles/labels across all axes; dedupe in order
        handles, labels, seen = [], [], set()
        for a in axes:
            h, l = a.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if not li or li == "_nolegend_":
                    continue
                if li not in seen:
                    seen.add(li); handles.append(hi); labels.append(li)

        # Hide legend if trivial
        if not labels or len(set(labels)) <= 1:
            legend_pos = "none"

        # Wrap title/subtitle                                                              OK
        if cfg.title:
            title_wrap = self._title_wrap(fig, cfg.title, title_fontsize)
        if cfg.subtitle:
            subtitle_wrap = self._title_wrap(fig, cfg.subtitle, subtitle_fontsize)

        # If there's no legend, add title/subtitle and return
        if legend_pos == "none" or not labels:
            if cfg.title:
                fig.suptitle(title_wrap, fontsize=title_fontsize)
            if cfg.subtitle:
                axes[0].set_title(subtitle_wrap, fontsize=subtitle_fontsize)
            fig.tight_layout()
            fig.canvas.draw()
            return

        # Wrap legend labels for right-side legends
        if legend_pos == "right" and labels:
            labels = list(self._legend_wrap(fig, labels))

        # Default legend title to group_by if not explicitly set
        if cfg.legend_label is None and cfg.group_by is not None:
            cfg.legend_label = cfg.group_by
        
        # Create a temporary legend and title to measure required space
        ncol = self._legend_ncol(fig, axes[0], legend_pos=legend_pos)
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
        if cfg.resize_fig:
            if legend_pos == "right":
                new_fig_width = fig_width + legend_width
                new_fig_height = fig_height + title_height
            elif legend_pos == "bottom":
                new_fig_width = fig_width
                new_fig_height = fig_height + legend_height + title_height
            fig.set_size_inches(new_fig_width, new_fig_height)
            fig.canvas.draw()

        # Determine the position of the legend (loc, bbox) and reserve legend/title space using tight_layout rect
        # rect : tuple (left, bottom, right, top), default: (0, 0, 1, 1)
        # "A rectangle in normalized figure coordinates into which the whole subplots area (including labels) will fit.""
        if legend_pos == "right": # right legend
            loc = "center right"
            bbox = (1.0, 0.5)
            rect = [0, 0, 1 - legend_width / new_fig_width, 1 - title_height / new_fig_height]
        else: # bottom legend
            loc = "lower center"
            bbox = (0.5, 0.0)
            rect = [0, legend_height / new_fig_height, 1, 1 - title_height / new_fig_height]

        # Put subtitle on the first axes
        if cfg.subtitle:
            axes[0].set_title(subtitle_wrap, fontsize=subtitle_fontsize)
        fig.tight_layout(rect=rect)

        # Center suptitle over the union of axes
        if cfg.title:
            boxes = [a.get_position() for a in axes]
            x0 = min(b.x0 for b in boxes); x1 = max(b.x1 for b in boxes)
            title_x = x0 + (x1 - x0) / 2.0
            fig.suptitle(title_wrap, fontsize=title_fontsize, x=title_x)

        # Add final legend in the reserved area
        if legend_pos != "none" and labels:
            ncol = self._legend_ncol(fig, axes[0], legend_pos)
            if legend_pos == "right":
                loc, bbox = "center right", (1.0, 0.5)
            else:  # bottom
                loc, bbox = "lower center", (0.5, 0.0)
            fig.legend(handles, labels, ncol=ncol, loc=loc, bbox_to_anchor=bbox, title=cfg.legend_label or None)

        fig.canvas.draw()

# -----------------------------------------------------------------------------
# Palettes
# -----------------------------------------------------------------------------

    def _palette_presets(self, name: str) -> Optional[list[str]]:
        presets = {
            "rainbow": RAINBOW_PALETTE,
            "colorblind": COLORBLIND_PALETTE,
            "default": RAINBOW_PALETTE,
        }
        key = (name or "").lower()
        return list(presets.get(key)) if key in presets else None
    
    def _coerce_palette_input(self, palette: Union[str, Sequence[str]]) -> list[str]:
        """Return a concrete list of hex colors from a preset name or a sequence."""
        if isinstance(palette, str):
            return self._palette_presets(palette) or []
        try:
            return [str(c) for c in list(palette)]
        except Exception:
            return []

    def set_palette(
        self,
        palette: Union[Dict[Any, str], str, Sequence[str]],
        *,
        append: bool = False,
    ) -> None:
        """
        Set instance-level palette defaults:
        - dict -> update/replace self.palette_map
        - str/sequence -> update/replace self.palette (stored as a list of hex strings)
        """
        
        # dict => affects palette_map
        if isinstance(palette, dict):
            if append:
                self.palette_map.update(palette)
            else:
                self.palette_map = dict(palette)
            return

        # str/sequence => affects palette (store as list[str])
        new_list = self._coerce_palette_input(palette)  # list[str]
        if not new_list:
            return

        if append and self.palette:
            self.palette = list(self.palette) + new_list
        else:
            self.palette = new_list

    def _get_palette(self, labels: Sequence[Any]) -> list[str]:
        """Creates a list of hex colour codes with len(labels). Precedence:
        1) cfg.palette_map (chart-specific per-label)
        2) cfg.palette (chart-specific list)
        3) self.palette_map (instance per-label)
        4) self.palette (instance list)
        5) FALLBACK_PALETTE (pads/cycles if necessary)
        """
        cfg = self._chart_params
        labels = list(labels) or ["_dummy_"]
        palette: list[Optional[str]] = [None] * len(labels)

        def fill_from_map(palette: list[Optional[str]], mapping: Dict[Any, str]) -> None:
            """Iterate through a list of labels and place the matching colour if it's found in the dictionary"""
            for i, lab in enumerate(labels):
                if palette[i] is None and lab in mapping:
                    palette[i] = mapping[lab]

        def fill_from_list(palette: list[Optional[str]], colors: Sequence[str]) -> None:
            """Iterate through a list of colours and place them into None slots in the palette"""
            it = iter(colors)
            for i, c in enumerate(palette):
                if c is None:
                    try:
                        palette[i] = next(it)
                    except StopIteration:
                        break

        def cycle_list(palette: list[Optional[str]], colors: Sequence[str]) -> None:
            """Cycle through a list of colours and place them into None slots in the palette"""
            fi = 0
            for i, c in enumerate(palette):
                if c is None:
                    palette[i] = colors[fi % len(colors)]
                    fi += 1

        if cfg.palette_map:
            fill_from_map(palette, cfg.palette_map)
        if cfg.palette:
            fill_from_list(palette, cfg.palette)
        if self.palette_map:
            fill_from_map(palette, self.palette_map)
        if self.palette:
            fill_from_list(palette, self.palette)
        cycle_list(palette, FALLBACK_PALETTE)

        return [str(c) for c in palette]

# -----------------------------------------------------------------------------
# Chart Primitives                                                           
# -----------------------------------------------------------------------------

    def _save_and_return(
        self,
        fig: Figure,
        axes: Union[Axes, Sequence[Axes], Dict[str, Axes]],
        chart_data: Union[pd.DataFrame, List[pd.DataFrame], None],
    ) -> Optional[Tuple[Figure, Union[Axes, Sequence[Axes], Dict[str, Axes]], Union[pd.DataFrame, List[pd.DataFrame], None]]]:
        
        cfg = self._chart_params
        if cfg.save_path:
            fig.savefig(cfg.save_path, bbox_inches="tight", dpi=fig.dpi)
        if cfg.show_fig:
            plt.show()
        if cfg.return_values or not cfg.show_fig:
            return fig, axes, chart_data
        return None

    def _chart_pipeline(
        self,
        plot_fn: Callable[[pd.DataFrame, Optional[Axes]], tuple[Figure, Axes, pd.DataFrame]],
        *,
        ax: Optional[Axes] = None,
        finalize: bool = True,
        hide_axis_labels: bool = False,
        **kwargs: Any,
    ) -> Optional[tuple[Figure, Axes, pd.DataFrame]]:
        """
        Common logic for all charts: set params -> aggregate -> sort -> plot using plot_fn chart primitive -> style axes -> (optional) style fig -> save/return.
        When finalize=False, skips figure-level styling and save/show for parent containers like compare().
        """
        self._set_params(**kwargs)                                  # ChartConfig on instance
        df_long = self._sort(self._aggregate_data())                # tidy long-form (x, group_by, value)
        fig, ax, chart_data = plot_fn(df_long, ax=ax)               # delegate to primitive (handles pivot & draw)

        # Axis-only styling (labels, limits, tick format, grid, benchmarks, etc.)
        self._style_axes(fig, ax, chart_data)

        if hide_axis_labels:
            ax.set_xlabel(""); ax.set_ylabel("")

        if finalize:
            self._style_fig(fig, [ax])                              # shared legend/title, tight layout
            return self._save_and_return(fig, ax, chart_data)       # obeys save_path, show_fig, return_values
        else:
            # For parent flow (e.g., compare), just return the pieces without saving/showing.
            return fig, ax, chart_data

    # TODO: work through the list of options in .plot and wrap them into this framework
    # 'line', 'bar', 'barh', 'hist', 'box', 'kde', 'density', 'area', 'pie', 'scatter', 'hexbin'

    def _bar(self, df: pd.DataFrame, ax: Optional[Axes] = None) -> tuple[Figure, Axes, pd.DataFrame]:
        cfg = self._chart_params
        fig, ax = self._ensure_fig_ax(ax)
        chart_data = self._pivot_data(df)
        labels_for_color = list(chart_data.columns) if cfg.group_by else [cfg.y or "value"]
        chart_data.plot(kind="bar",
                        stacked=(cfg.stacking != "none"),
                        color=self._get_palette(labels_for_color),
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
        chart_data = chart_data.mask(chart_data.eq(0)) # prefer gaps in lines where no data exists

        labels_for_color = list(chart_data.columns) if chart_data.shape[1] > 1 else ["value"]
        chart_data.plot(ax=ax, marker="o", color=self._get_palette(labels_for_color), legend=False)

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

        # Remap labels
        if cfg.label_map:
            chart_data.index = chart_data.index.map(lambda v: cfg.label_map.get(v, v))

        # Labels / percentages
        labels = chart_data.index.astype(str).values if cfg.show_labels else None
        if cfg.show_percent is True:
            autopct = "%1.0f%%"
        elif isinstance(cfg.show_percent, str):
            autopct = cfg.show_percent
        else:
            autopct = None

        # Draw
        labels_for_color = list(chart_data.index)
        wedges, texts, autotexts = ax.pie(
            chart_data["value"].values,
            labels=labels,
            colors=self._get_palette(labels_for_color),
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
# Public APIs                                                           
# -----------------------------------------------------------------------------

    @add_docstring(COMMON_DOCSTRING)
    def bar(self, ax: Optional[Axes] = None, finalize: bool = True, **kwargs: Any):
        """Bar chart (counts by default, aggregate `y` when provided)."""
        return self._chart_pipeline(self._bar, ax=ax, finalize=finalize, **kwargs)

    @add_docstring(COMMON_DOCSTRING)
    def line(self, ax: Optional[Axes] = None, finalize: bool = True, **kwargs: Any):
        """Line chart. Requires `x`."""
        return self._chart_pipeline(self._line, ax=ax, finalize=finalize, **kwargs)

    @add_docstring(COMMON_DOCSTRING)
    def pie(self, ax: Optional[Axes] = None, finalize: bool = True, **kwargs: Any):
        """Pie chart. Requires `names`+`values` (or `x` as names + counts)."""
        # Pie uses axis-only labels; we hide axes labels explicitly as before.
        return self._chart_pipeline(self._pie, ax=ax, finalize=finalize, hide_axis_labels=True, **kwargs)

    @add_docstring(COMMON_DOCSTRING)
    def compare(
        self,
        chart1: ChartSpec,
        chart2: ChartSpec,
        chart3: Optional[ChartSpec] = None,
        *,
        sharey: bool = False,
        **kwargs: Any,
    ) -> tuple[Figure, Dict[str, Axes]]:
        """Create a side-by-side figure (2 or 3 axes) of individual lazychart charts,
        using the chart proxy to create a ChartSpec for each chart.

        Usage example:
        
        cm.compare(
            cm.chart.bar(...),
            cm.chart.line(...),
            title = "..."
        )

        Figure-level config (title, legend position, fig_size, save/show) is taken from compare() kwargs.
        Axis-level config comes from each chart's own call.
        """
        # Compare() function config drives figure/title/legend/save/show
        fig_cfg = self._set_params(**kwargs)

        # Build a figure to place the individual charts into
        n = 3 if chart3 else 2
        base_w, base_h = fig_cfg.fig_size
        fig, axes = plt.subplots(1, n, figsize=(base_w * n, base_h), sharey=sharey)

        # Draw child charts
        _, _, chart_data1 = chart1.render(axes[0])
        _, _, chart_data2 = chart2.render(axes[1])
        chart_data = [chart_data1, chart_data2]
        if chart3:
            _, _, chart_data3 = chart3.render(axes[2])
            chart_data.append(chart_data3)

        # Figure styling
        self._chart_params = fig_cfg
        self._style_fig(fig, axes)
        self._save_and_return(fig, axes, chart_data)