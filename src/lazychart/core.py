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
from math import ceil, floor, log10
from difflib import get_close_matches
import colorsys
import math

# Logging
#from lazychart.log import create_logger
#put it here instead of separate module for ease of copying between environments
import logging
def create_logger(name: str = __name__, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers twice if create_logger is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)

    # Silence noisy third-party packages
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    return logger
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
        Will auto-detect if left as None
    decimals : int, optional
        Will auto-detect if left as None
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
    footer : str, optional
        Figure-level footnote shown at the bottom-left. Intended for filters,
        date ranges, caveats—kept with the chart so it travels with copies.
    footer_size : int | {'small','medium','large','x-large'}, optional
        Defaults to smaller than tick labels.
    footer_color : str, optional
        Matplotlib color spec for the footnote text. Defaults to a soft grey.

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
    palette_gradient : str, optional
        Single hex colour code from which shades of the same colour are used to fill the palette.
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

    # Targets
    "x_target": "target_x",
    "y_target": "target_y",
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

    # 6) Warn if something unexpected is passed
    allowed = set(CANONICAL_KEYS) | set(ALIASES.keys()) | set(COMPOSITE_ALIASES.keys()) | {
        # also allow these “entry” names that normalize elsewhere:
        "legend", "legend_pos", "legend_position", "loc", "barmode"
    }
    unknown = [k for k in kw.keys() if k not in allowed]
    if unknown:
        suggestions = []
        universe = list(CANONICAL_KEYS | set(ALIASES.values()) | set(ALIASES.keys()) | set(COMPOSITE_ALIASES.keys()))
        for bad in unknown:
            guess = get_close_matches(bad, universe, n=1)
            if guess:
                suggestions.append(f"`{bad}` (did you mean `{guess[0]}`?)")
            else:
                suggestions.append(f"`{bad}`")
        msg = "Unsupported argument(s): " + ", ".join(suggestions)
        logger.warning(msg)

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
    """Having all parameters stored here allows them to be used with sticky()"""

    # ---- Core selection / aggregation ----
    data: Optional[pd.DataFrame] = None
    x: Optional[str] = None
    y: Optional[Union[str, Sequence[str]]] = None
    group_by: Optional[str] = None
    aggfunc: Union[str, Callable] = "sum"
    x_period: Optional[Literal["month", "quarter", "year", "week", "day"]] = None
    show_gaps: bool = True

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

    # ---- Bar ----
    stacking: Literal["stacked", "proportion", "none"] = "stacked"

    # ---- Line ----
    show_total: bool = False # when True and group_by is used, add an overall Total line
    total_name: str = "Average" # label for the aggregate line (used for legend & palette_map lookup)
    
    # ---- Pie ----
    values: Optional[str] = None
    names: Optional[str] = None
    show_labels: bool = False

    # ---- Delay ----
    x2: Optional[Union[str, pd.Timestamp]] = None
    cumulative: bool = False
    proportion: bool = False
    bin_size: int = 1

    # ---- Mix / Compare ----
    mix_kind: Literal['bar','pie','line'] = 'bar'
    mix_by: Optional[str] = None
    allow_x_fallback: bool = True
    mix_max_categories: int = 30
    mix_max_ratio: float = 0.2
    mix_subtitle: Optional[str] = 'Mix'
    sharey: Optional[bool] = None  # figure-level, used by compare()
    
    # ---- Axis formats ----
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    x_axis_format: AxisFormat = None
    y_axis_format: AxisFormat = None
    decimals: Optional[int] = None
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

    # ---- Footer ----
    footer: Optional[str] = None
    footer_size: Optional[Union[int, str]] = None
    footer_color: Optional[str] = None

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
    palette_gradient : Optional[str] = None

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
        # TODO: consider allowing negative decimals to e.g. round to nearest 10 with -1 or 100 with -2
        if self.decimals and self.decimals < 0:
            self.decimals = None

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

    def render(self, ax: Axes, **overrides):
        """API for drawing onto a Matplotlib Axes using the corresponding
        public method (e.g. `cm.bar`) with `finalize=False`
        
        Returns fig, ax, chart_data
        """
        method = getattr(self._cm, self.kind)
        merged = {**overrides, **self.kwargs}  # child kwargs override compare-level
        return method(ax=ax, finalize=False, **merged)

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
        "small":    {"title": 13, "subtitle": 11, "x_label": 10, "y_label": 10, "tick": 9,  "footnote": 8},
        "medium":   {"title": 15, "subtitle": 13, "x_label": 12, "y_label": 12, "tick": 11, "footnote": 9},
        "large":    {"title": 17, "subtitle": 15, "x_label": 14, "y_label": 14, "tick": 13, "footnote": 11},
        "x-large":  {"title": 19, "subtitle": 17, "x_label": 16, "y_label": 16, "tick": 15, "footnote": 13},
    }

    if isinstance(size, int):
        return size
    elif isinstance(size, str):
        preset = FONT_PRESETS.get(size.lower())
        if preset:
            return preset[label]
    return FONT_PRESETS["medium"][label] # use medium by default

def _format_axis(ax: Axes, which: str, fmt: str | None, decimals: int | None) -> None:
    """
    Apply percent, comma, short, or auto-detected format to an axis.

    - If `fmt` is provided ("percent", "comma", "short") it is respected.
    - If `fmt` is None, formatter is auto-selected from axis range:
        * values in [0,1] → percent
        * max abs value >= 1e6 → short (M/B)
        * max abs value >= 1e3 → comma
        * else → plain decimals
    - Decimal places are auto-detected from tick step if `decimals` is None,
      using the rule: needed_dp = -floor(log10(step)), clamped to [0,2].
    """
    axis = ax.yaxis if which == "y" else ax.xaxis

    # Range of axis
    vmin, vmax = axis.get_view_interval()
    vmax_abs = max(abs(vmin), abs(vmax))

    # Choose format
    if fmt in ("percent", "comma", "short"):
        chosen_fmt = fmt
    else:
        if vmax <= 1.0 and vmin >= 0.0:
            chosen_fmt = "percent"
        elif vmax_abs >= 1_000_000:
            chosen_fmt = "short"
        elif vmax_abs >= 1_000:
            chosen_fmt = "comma"
        else:
            chosen_fmt = None

    # Estimate tick step
    ticks = axis.get_ticklocs()
    finite_ticks = [t for t in ticks if np.isfinite(t)]
    if len(finite_ticks) >= 2:
        step = max(1e-12, finite_ticks[1] - finite_ticks[0])
    else:
        step = max(1e-12, vmax - vmin)

    # Pick divisor & suffix based on vmax_abs
    if chosen_fmt == "short":
        if vmax_abs >= 1_000_000_000:
            divisor, suffix = 1_000_000_000, "B"
        elif vmax_abs >= 1_000_000:
            divisor, suffix = 1_000_000, "M"
        elif vmax_abs >= 1_000:
            divisor, suffix = 1_000, "k"
        else:
            divisor, suffix = 1, ""
    elif chosen_fmt == "percent":
        divisor, suffix = 0.01, "%"
    else:
        divisor, suffix = 1, ""

    # Auto-decimal detection
    if decimals is None:
        step = step / divisor if divisor else step
        decimals = max(0, -floor(log10(step)))

    # Apply formatter
    if chosen_fmt == "percent":
        axis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=decimals))
    elif chosen_fmt == "comma":
        axis.set_major_formatter(mticker.StrMethodFormatter(f"{{x:,.{decimals}f}}"))
    elif chosen_fmt == "short":
        def short(x, pos=None):
            return f"{x/divisor:.{decimals}f}{suffix}"
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

        # NEW: if x is numeric (e.g., delay bins), do NOT coerce to datetime
        if pd.api.types.is_numeric_dtype(x_series):
            return df
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(x_series):
            try:
                # pandas >= 2.0: avoids the "Could not infer format..." UserWarning
                x_series = pd.to_datetime(x_series, errors="raise", format="mixed")
            except TypeError:
                # pandas < 2.0 fallback
                x_series = pd.to_datetime(x_series, errors="raise")
            except Exception:
                return df

        # Coerce to period start using canonical freq (D, W-XXX, M, Q, Y)
        coerced = x_series.dt.to_period(cfg.x_period).dt.start_time

        # Write back
        if use_index:
            df.index = pd.DatetimeIndex(coerced.values, name=cfg.x)
        else:
            # Avoid dtype-incompatible .loc setitem
            df = df.copy()
            df[cfg.x] = pd.DatetimeIndex(coerced.values)

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

        # de-duplicate while preserving order (protects reset_index) in case e.g. x=group_by
        seen = set()
        keys = [k for k in keys if not (k in seen or seen.add(k))]

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

    def _ensure_fig_ax(self, ax: Optional[Axes] = None, prefer_square: bool = False) -> tuple[Figure, Axes]:
        """Create an axes if one doesn't already exist."""
        if ax is not None:
            return ax.figure, ax
        
        cfg = self._chart_params
        base_w, base_h = cfg.fig_size
        if prefer_square:
            s = min(base_w, base_h)
            figsize = (s, s)
        else:
            figsize = (base_w, base_h)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
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
            agg = cfg.aggfunc
            if agg is None:
                ax.set_ylabel(str(cfg.y))
            elif isinstance(agg, str):
                ax.set_ylabel(f"{agg}({cfg.y})")
            else:
                name = getattr(agg, "__name__", "agg")
                ax.set_ylabel(f"{name}({cfg.y})")

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

    def _legend_ncol(self, fig: Figure, handles: Sequence[Any], labels: Sequence[str], legend_pos: str = "right", max_ratio: float = 0.9) -> int:
        """Determine optimal number of legend columns to ensure legend stays within the 
        height (right legend) / width (bottom legend) of the data axes and uses
        extra cols (right legend) / rows (bottom legend) in order to do so."""

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
        footer_fontsize = _resolve_fontsize(cfg.footer_size, "footnote")
        footer_color = cfg.footer_color or "#6B7280"  # soft grey

        # Collect handles/labels across all axes; dedupe in order
        axes_seq = list(axes)  # axes is already a list for compare(); safe for single charts too
        is_compare = len(axes_seq) > 1 # Treat as 'compare' only if there are multiple axes
        handles, labels, seen = [], [], set()
        for a in axes:
            h, l = a.get_legend_handles_labels()
            # valid, user-facing labels on this axis
            valid = [li for li in l if li and li != "_nolegend_"]
            # In compare/mix: only include entries from axes that *individually* have >1 unique labels
            if is_compare and len(set(valid)) <= 1:
                continue

            # Otherwise, add valid labels from this axis, de-duplicated globally
            for hi, li in zip(h, l):
                if not li or li == "_nolegend_":
                    continue
                if li not in seen:
                    seen.add(li)
                    handles.append(hi)
                    labels.append(li)

        # Hide legend if trivial
        if not labels or len(set(labels)) <= 1:
            legend_pos = "none"

        # Wrap title/subtitle
        if cfg.title:
            title_wrap = self._title_wrap(fig, cfg.title, title_fontsize)
        if cfg.subtitle:
            subtitle_wrap = self._title_wrap(fig, cfg.subtitle, subtitle_fontsize)

        # Wrap legend labels for right-side legends
        if legend_pos == "right" and labels:
            labels = list(self._legend_wrap(fig, labels))

        # Default legend title to group_by if not explicitly set
        if cfg.legend_label is None and cfg.group_by is not None:
            cfg.legend_label = cfg.group_by
        
        # Create temporary legend/title (and compute a tentative footer height) to measure required space
        ncol = self._legend_ncol(fig, handles, labels, legend_pos=legend_pos)
        temp_legend = None
        if legend_pos != "none" and labels:
            temp_legend = fig.legend(handles, labels, ncol=ncol, loc="upper right", title=cfg.legend_label or None)
        if cfg.title:
            temp_title = fig.text(0, 1, title_wrap, fontsize=title_fontsize)
        else:
            temp_title = None

        # Initialise the renderer
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # Measure spaces
        fig_width, fig_height = fig.get_size_inches()  # inches
        dpi = fig.dpi
        legend_height = 0.0
        legend_width = 0.0
        if temp_legend is not None:
            lb = temp_legend.get_window_extent(renderer=renderer)
            legend_height = lb.height / dpi
            legend_width = lb.width / dpi
        title_height = 0.0
        if temp_title is not None:
            tb = temp_title.get_window_extent(renderer=renderer)
            title_height = tb.height / dpi

        # Estimate footer wrapping and height (reserve even when legend='none')
        footer_height = 0.0
        footer_wrap = None
        if cfg.footer:
            # Available width fraction for footnote text (shrink if right legend)
            if legend_pos == "right" and legend_width > 0:
                avail_w_frac = max(0.05, 1.0 - (legend_width / fig_width))
            else:
                avail_w_frac = 1.0
            fig_w_px = fig_width * dpi
            approx_char_px = (footer_fontsize * dpi / 72.0) * 0.6
            max_chars = max(10, int((fig_w_px * avail_w_frac * 0.95) // approx_char_px))
            from textwrap import wrap as _wrap
            footer_wrap = "\n".join(_wrap(cfg.footer, max_chars))
            temp_footer = fig.text(0, 0, footer_wrap, fontsize=footer_fontsize)
            fig.canvas.draw()
            fb = temp_footer.get_window_extent(renderer=renderer)
            footer_height = fb.height / dpi
            footer_height *= 1.3 # add a margin between the footer and the axis label / legend
            temp_footer.remove()

        # Cleanup temporary legend/title
        if temp_legend is not None:
            temp_legend.remove()
        if temp_title is not None:
            temp_title.remove()

        # Expand the figure to accomodate the legend and title
        new_fig_width = fig_width
        new_fig_height = fig_height
        if legend_pos == "right":
            new_fig_width = fig_width + legend_width
            new_fig_height = fig_height + title_height + footer_height
        elif legend_pos == "bottom":
            new_fig_width = fig_width
            new_fig_height = fig_height + legend_height + title_height + footer_height
        else:  # no legend
            new_fig_width = fig_width
            new_fig_height = fig_height + title_height + footer_height
        fig.set_size_inches(new_fig_width, new_fig_height)
        fig.canvas.draw()

        # Determine subplot rect (reserve space for title at top and footer + legend at bottom)
        # rect : (left, bottom, right, top) in figure coords for the subplots area
        if legend_pos == "right": # right legend
            loc = "center right"
            bbox = (1.0, 0.5)
            rect = [0, footer_height / new_fig_height, 1 - legend_width / new_fig_width, 1 - title_height / new_fig_height]
        elif legend_pos == "bottom":  # bottom legend
            loc = "lower center"
            bbox = (0.5, footer_height / new_fig_height)
            rect = [0, (legend_height + footer_height) / new_fig_height, 1, 1 - title_height / new_fig_height]
        else: # no legend
            loc = None
            bbox = None
            rect = [0, footer_height / new_fig_height, 1, 1 - title_height / new_fig_height]

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

        # Add footer
        if cfg.footer and footer_wrap:
            #x_left = fig.subplotpars.left # anchor to left margin of subplot area
            x_left = 0.05 # anchor to left of the figure (with optional tiny margin)
            # margin = (footer_height * 0.3) / new_fig_height
            # y_pos = (footer_height * 0.5) / new_fig_height + margin # centered in reserved footer band
            y_pos = 0.0 + (footer_height * 0.1) / new_fig_height # position at the bottom of the reserved footer band with a slight margin
            fig.text(x_left, y_pos, footer_wrap, fontsize=footer_fontsize, color=footer_color, ha="left", va="center")

        # Add final legend in the reserved area
        if legend_pos != "none" and labels:
            ncol = self._legend_ncol(fig, handles, labels, legend_pos)
            if legend_pos == "right":
                loc, bbox = "center right", (1.0, 0.5)
            else:  # bottom
                loc, bbox = "lower center", (0.5, footer_height / new_fig_height)
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

        def generate_shades(palette: list[Optional[str]], color: str) -> list[str]:
            """Converts one hex colour code into shades of that colour
            to fill any remaining None slots in the palette
            """
            n = sum(c is None for c in palette)
            base_hex = color.lstrip("#")
            r = int(base_hex[0:2], 16) / 255.0
            g = int(base_hex[2:4], 16) / 255.0
            b = int(base_hex[4:6], 16) / 255.0
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            # spread lightness between ~0.25..0.75 for contrast
            lows, highs = 0.25, 0.75
            vals = [lows + (highs - lows) * (i / max(n - 1, 1)) for i in range(n)]
            shades = []
            for ll in vals:
                rr, gg, bb = colorsys.hls_to_rgb(h, ll, s)
                shades.append("#{:02x}{:02x}{:02x}".format(int(rr*255), int(gg*255), int(bb*255)))
            return shades

        if cfg.palette_map:
            fill_from_map(palette, cfg.palette_map)
        if cfg.palette_gradient:
            shades = generate_shades(palette, cfg.palette_gradient)
            fill_from_list(palette, shades)
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
        style_axes: bool = True, 
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
        if style_axes:
            self._style_axes(fig, ax, chart_data)

        if hide_axis_labels:
            ax.set_xlabel("")
            ax.set_ylabel("")

        if finalize:
            self._style_fig(fig, [ax])                              # shared legend/title, tight layout
            return self._save_and_return(fig, ax, chart_data)       # obeys save_path, show_fig, return_values
        else:
            # axes level subtitles
            cfg = self._chart_params
            if cfg.subtitle:
                sub_fs = _resolve_fontsize(cfg.subtitle_size, "subtitle")
                ax.set_title(self._title_wrap(fig, cfg.subtitle, sub_fs), fontsize=sub_fs)

            # For parent flow (e.g., compare), just return the pieces without saving/showing.
            return fig, ax, chart_data

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

        # Optional total average series
        if cfg.group_by and cfg.show_total:
            avg_label = (cfg.total_name or "Average")
            # Treat structural zeros as NaN so gaps don't depress the average
            avg = chart_data.mask(chart_data.eq(0)).mean(axis=1, skipna=True)
            chart_data[avg_label] = avg
            # Default a distinct color if the user didn't map it
            if avg_label not in cfg.palette_map and avg_label not in self.palette_map:
                cfg.palette_map[avg_label] = "#000000"  # default black
        
        # Prefer gaps in lines where no data exists
        chart_data = chart_data.mask(chart_data.eq(0))

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
        fig, ax = self._ensure_fig_ax(ax, prefer_square=True)

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

        # set legend labels on the wedge artists, *do not* create an axes legend here.
        # This allows _style_fig to collect handles/labels and place a single, figure-level legend
        # at the requested location without triggering aspect/limits gymnastics.
        for w, lab in zip(wedges, chart_data.index.astype(str).tolist()):
            w.set_label(lab)

        # Keep the pie circular and centered without messing with x/y limits
        ax.set_aspect('equal', adjustable='box')

        return fig, ax, chart_data

# -----------------------------------------------------------------------------
# Public APIs                                                           
# -----------------------------------------------------------------------------

    @add_docstring(COMMON_DOCSTRING)
    def bar(self, ax: Optional[Axes] = None, finalize: bool = True, **kwargs: Any):
        """Bar chart.

        By default shows **counts** per category (``x``). If you provide ``y``, the
        values are aggregated per category with ``aggfunc`` (default ``'sum'``).

        Supports grouped bars via ``group_by`` and stacking via ``stacking``
        (``'stacked'`` | ``'proportion'`` | ``'none'``).

        Examples
        --------
        >>> cm.bar(data=df, x='department')
        >>> cm.bar(data=df, x='month', y='revenue', aggfunc='mean', group_by='region', stacking='none')
        """
        return self._chart_pipeline(self._bar, ax=ax, finalize=finalize, **kwargs)

    @add_docstring(COMMON_DOCSTRING)
    def line(self, ax: Optional[Axes] = None, finalize: bool = True, **kwargs: Any):
        """Line chart.

        Requires ``x``. If ``y`` is omitted, draws counts per ``x``. With
        ``group_by``, draws one line per group. If ``x`` is datetime-like, you can
        bin with ``x_period`` (e.g., ``'month'``/``'quarter'``) and optionally fill
        gaps (``show_gaps=True``, default when ``x_period`` is set).

        Examples
        --------
        >>> cm.line(data=df, x='date', y='sales', group_by='region', x_period='month')
        >>> cm.line(data=df, x='delay_days') # counts per delay day
        """
        return self._chart_pipeline(self._line, ax=ax, finalize=finalize, **kwargs)

    @add_docstring(COMMON_DOCSTRING)
    def pie(self, ax: Optional[Axes] = None, finalize: bool = True, **kwargs: Any):
        """Pie chart.

        Parameters
        ----------
        data : pd.DataFrame
        Source data.
        names : str, optional
        Category column for slices. If omitted, you may pass ``x`` (alias).
        values : str, optional
        Numeric column with slice sizes. If omitted and only ``names``/``x``
        are supplied, counts are used upstream.
        show_labels : bool, default False
        If ``True``, draw labels directly on wedges.
        show_percent : bool | str, default True
        If ``True``, add percentage text to wedges (``"%1.0f%%"``). If a format
        string is provided, it is passed to Matplotlib's ``autopct``.

        Notes
        -----
        Legends are composed at the **figure level** (right/bottom) so the pie itself
        stays large and centered. Use ``legend='right'`` (default), ``'bottom'``, or ``'none'``.

        Examples
        --------
        >>> cm.pie(data=df, names='species', values='count', legend='right', title='Share by species')
        >>> cm.pie(data=df, x='species') # counts rows per category
        """

        # aliasing just for pies
        if "values" not in kwargs and "x" in kwargs:
            kwargs["values"] = kwargs.pop("x")
        if "names" not in kwargs and "group_by" in kwargs:
            kwargs["names"] = kwargs.pop("group_by")

        # Pie uses axis-only labels; we hide axes labels explicitly as before.
        return self._chart_pipeline(self._pie, ax=ax, finalize=finalize, hide_axis_labels=True, style_axes=False, **kwargs)

    @add_docstring(COMMON_DOCSTRING)
    def compare(self, chart1: ChartSpec, chart2: ChartSpec, chart3: Optional[ChartSpec] = None, **kwargs: Any) -> tuple[Figure, Dict[str, Axes]]:
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

        charts = [chart1, chart2] + ([chart3] if chart3 else [])

        # ensure individual charts are passed using cm.chart.x syntax and not cm.x syntax
        def _check_spec(obj, name):
            if not isinstance(obj, ChartSpec):
                raise TypeError(
                    f"compare(...): `{name}` must be built with cm.chart.<kind>(...). "
                    f"You passed {type(obj).__name__}. Use cm.chart.bar(...) not cm.bar(...)."
                )
        
        for i, c in enumerate(charts, 1):
            _check_spec(c, f"chart{i}")

        # Compare() function config drives figure/title/legend/save/show
        fig_cfg = self._set_params(**kwargs)

        # Build a figure to place the individual charts into
        n = len(charts)
        base_w, base_h = fig_cfg.fig_size
        min_wh = min(base_w, base_h)
        sizes = [(min_wh, min_wh) if c.kind == 'pie' else (base_w, base_h) for c in charts]
        fig_w = sum(w for w, _ in sizes) # side by side charts have additive widths 
        fig_h = max(h for _, h in sizes) # but not additive heights
        width_ratios = [w for w, _ in sizes]
        fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), sharey=fig_cfg.sharey, gridspec_kw={"width_ratios": width_ratios})
         
        # Draw child charts
        chart_data = []
        for ax, chart in zip(axes, charts):
            _, _, data = chart.render(ax, **kwargs)
            chart_data.append(data)

        # Figure styling
        self._chart_params = fig_cfg
        self._style_fig(fig, axes)
        return self._save_and_return(fig, axes, chart_data)
    
    def mix(self, chart1: 'ChartSpec', **kwargs: Any):
        """
        Compare a primary chart with a simple **class mix** panel.

        Dimension selection (in order):
        1) `mix_by` if provided,
        2) `group_by` from the primary chart; if absent, sticky `group_by` from cfg,
        3) fallback to `x` **iff** it's plausibly categorical (object/category/string/bool,
            or low cardinality by absolute/relative thresholds).

        Parameters
        ----------
        chart1 : ChartSpec
            Chart spec created with `cm.chart.<kind>(...)`.
        mix_kind : {'bar','pie','line'}, default 'bar'
            Chart type for the mix panel.
        mix_by : str, optional
            Column to show the mix over (overrides heuristics).
        allow_x_fallback : bool, default True
            Allow falling back to `x` when no `group_by` is found and `x` is categorical.
        mix_max_categories : int, default 30
            Absolute unique-count threshold for categorical fallback.
        mix_max_ratio : float, default 0.2
            Relative uniqueness threshold (nunique / rows) for numeric fallback.
        sharey : bool, default False
            Share y-axis when sensible.
        mix_subtitle : str | None, default 'Mix'
            Subtitle for the mix panel.

        Examples
        --------
        >>> cm.mix(cm.chart.bar(data=df, x='species', y='weight', aggfunc='mean'))
        >>> cm.mix(cm.chart.line(data=df, x='feature'), mix_kind='bar')
        """
        if not isinstance(chart1, ChartSpec):
            raise TypeError("mix(...): pass a chart created with cm.chart.<kind>(...).")

        # 0) Pull sticky params & figure-level kwargs (like compare() does)
        fig_cfg = self._set_params(**kwargs)

        # 1) Read first-chart kwargs
        ckwargs = dict(chart1.kwargs)
        grp_chart = ckwargs.get('group_by', None)
        x_chart   = ckwargs.get('x', None)

        # 2) Resolve the dataframe once (respects sticky cfg.data, filters, etc.)
        #    _prepare_dataframe uses the cfg set by _set_params above.
        df = self._prepare_dataframe()

        # 3) Choose mix dimension
        mix_var = fig_cfg.mix_by  # explicit wins
        if mix_var is None:
            mix_var = grp_chart or fig_cfg.group_by  # prefer group_by (chart → sticky)

        # helper: decide if a df column is plausibly categorical
        def _is_plausibly_categorical(df: pd.DataFrame, col: str) -> bool:
            if col not in df.columns:
                return False
            s = df[col]
            import pandas.api.types as ptypes
            n = len(s)
            nunique = s.nunique(dropna=True)

            # obvious categorical types
            if ptypes.is_categorical_dtype(s) or ptypes.is_bool_dtype(s) or ptypes.is_string_dtype(s) or s.dtype == object:
                return True

            # datetime: treat as categorical only if low-card
            if ptypes.is_datetime64_any_dtype(s):
                return nunique <= fig_cfg.mix_max_categories

            # numeric: accept small discrete domains
            if ptypes.is_integer_dtype(s) or ptypes.is_numeric_dtype(s):
                return (nunique <= fig_cfg.mix_max_categories) or ((nunique / max(n, 1)) <= fig_cfg.mix_max_ratio)

            # default: conservative
            return nunique <= fig_cfg.mix_max_categories

        if mix_var is None and fig_cfg.allow_x_fallback and isinstance(x_chart, str) and _is_plausibly_categorical(df, x_chart):
            mix_var = x_chart

        if not mix_var:
            raise ValueError(
                "mix(...): could not infer a categorical field for the mix panel. "
                "Prefer charts with `group_by`, pass mix_by='column', or ensure x is categorical."
            )

        if fig_cfg.mix_kind not in {'bar', 'pie', 'line'}:
            raise ValueError("mix_kind must be one of: 'bar', 'pie', or 'line'.")

        # 4) Build the second spec: simple counts by mix_var (no y)
        second_kwargs: Dict[str, Any] = {'x': mix_var, 'group_by': None, 'x_period': None, 'show_gaps': False, 'names': None, 'values': None}
        if fig_cfg.mix_subtitle is not None:
            second_kwargs['subtitle'] = fig_cfg.mix_subtitle
        if fig_cfg.mix_kind == 'pie':
            second_kwargs.pop('x', None)
            second_kwargs['names'] = mix_var
            second_kwargs['values'] = None  # counts

        chart2 = ChartSpec(self, fig_cfg.mix_kind, second_kwargs)

        # 5) Compare (figure-level kwargs already merged via _set_params; still pass through)
        return self.compare(chart1, chart2, sharey=fig_cfg.sharey, **kwargs)

    def delay(self, *, ax: Optional[Axes] = None, finalize: bool = True, **kwargs: Any):
        """
        Delay distribution chart.

        Compute delays between two date/time values (**x** and **x2**) and plot either
        the **incremental** (per-bin) or **cumulative** distribution as a line chart.

        - If you pass a numeric **y** in kwargs, the chart aggregates **sums of y**.
        - If you omit **y**, it defaults to **counts**.
        - If **group_by** is provided, the above is done per group.

        Parameters
        ----------
        x, x2 : str | pd.Timestamp
            Either column names in ``data`` or fixed date/times (parsed with ``pd.to_datetime``).
            Order does **not** matter; delays are absolute differences.
        x_period : {'month','quarter','year','week','day'}, default 'day'
            Calendar-aware unit used to measure the delay (mapped to pandas freq codes).
        cumulative : bool, default False
            If True, plot per-bin values (PDF). If False, plot cumulative values (CDF).
        proportion : bool, default False
            If True, scale each series to proportions:
              - incremental: values per series sum to 1.0
              - cumulative: the rightmost value reaches 1.0
        bin_size : int, default 1
            Combine consecutive delay bins (e.g., 7-day buckets when x_period='day' and bin_size=7).
        """

        # Pull chart config (y/group_by/title/etc.)
        cfg = self._set_params(**kwargs)
        if cfg.x is None or cfg.x2 is None:
            raise TypeError("delay(): required arguments missing: x and x2")

        df0 = self._prepare_dataframe()  # defensive copy

        # Resolve two datetime-like series (column or scalar)
        def _resolve_datetime(val):
            if isinstance(val, str) and val in df0.columns:
                return pd.to_datetime(df0[val], errors='coerce'), True
            try:
                ts = pd.to_datetime(val)
                return pd.Series(ts, index=df0.index), False  # broadcast scalar
            except Exception as exc:
                raise ValueError(f"delay(...): could not parse {val!r} as datetime or column.") from exc

        s, _ = _resolve_datetime(cfg.x)
        e, _ = _resolve_datetime(cfg.x2)

        # Normalize x_period using existing helper (calendar-aware)
        freq = None
        if cfg.x_period is not None:
            freq = self._chart_params._normalize_x_period(cfg.x_period)  # e.g., 'D','W-MON','M','Q','Y'

        # Compute absolute delay in "number of periods"
        if freq:
            s_per = s.dt.to_period(freq)
            e_per = e.dt.to_period(freq)
            diffs = e_per - s_per
            delays = pd.Series([d.n for d in diffs], index=s.index).abs()
        else:
            # default to days if somehow None
            vals = (e - s).abs() / np.timedelta64(1, 'D')
            delays = np.floor(pd.to_numeric(vals, errors='coerce') / max(int(cfg.bin_size), 1)) * max(int(cfg.bin_size), 1)

        # Optional integer binning across calendar units (e.g., 3 months, 2 quarters)
        if freq:
            b = max(int(cfg.bin_size), 1)
            if b != 1:
                delays = (delays // b) * b

        df1 = df0.copy()
        df1['delay'] = delays
        df1 = df1.dropna(subset=['delay'])

        # Determine grouping and y (sum column) behavior
        grp = cfg.group_by  # may be None
        y_col: Optional[str] = None
        if cfg.y is not None:
            if isinstance(cfg.y, str):
                y_col = cfg.y
                if y_col not in df1.columns:
                    raise ValueError(f"y={y_col!r} not found in data.")
            else:
                raise ValueError("delay(...): y must be a single column (str), not a sequence.")

        # Default friendly y-axis label (can be overridden by kwargs)
        if cfg.proportion:
            default_ylabel = 'Proportion'
        elif y_col:
            agg = cfg.aggfunc
            if agg is None:
                default_ylabel = str(y_col)
            elif isinstance(agg, str):
                default_ylabel = f"{agg}({y_col})"
            else:
                default_ylabel = f"{getattr(agg, '__name__', 'agg')}({y_col})"
        else:
            default_ylabel = 'Count'

        # --- Build incremental base (per-bin) ---
        if grp:
            if y_col:
                base = (df1.groupby([grp, 'delay'], observed=True)[y_col].agg(cfg.aggfunc).reset_index(name='value'))
            else:
                base = (df1.groupby([grp, 'delay'], observed=True).size().reset_index(name='value'))
            base = base.sort_values(['delay', grp])
        else:
            if y_col:
                base = (df1.groupby('delay', observed=True)[y_col].agg(cfg.aggfunc).reset_index(name='value'))
            else:
                base = (df1.groupby('delay', observed=True).size().reset_index(name='value'))
            base = base.sort_values('delay')

        # Scale to proportions if requested (incremental)
        data_inc = base.copy()
        if cfg.proportion and not data_inc.empty:
            if grp:
                totals = data_inc.groupby(grp, observed=True)['value'].transform('sum')
                data_inc['value'] = np.where(totals.gt(0), data_inc['value'] / totals, 0.0)
            else:
                total = float(data_inc['value'].sum())
                data_inc['value'] = data_inc['value'] / total if total > 0 else 0.0

        # Prepare passthrough kwargs (do not leak our local control args)
        def _passthrough(extras: Dict[str, Any]) -> Dict[str, Any]:
            merged = {k: v for k, v in kwargs.items()
                    if k not in {'data','x','x2','y','group_by','aggfunc','x_period',
                                'incremental','proportion','bin_size','start','end','y_label'}}
            merged['y_label'] = kwargs.get('y_label', default_ylabel) # user label has priority
            if cfg.proportion and 'y_axis_format' not in kwargs: # show proportions as %
                merged['y_axis_format'] = 'percent'
            merged.update(extras)
            return merged

        # Cumulative view
        if cfg.cumulative:
            data_cum = data_inc if cfg.proportion else base
            if grp:
                data_cum = data_cum.sort_values(['delay', grp])
                data_cum['value'] = data_cum.groupby(grp, observed=True)['value'].cumsum()
                if cfg.proportion and not data_cum.empty:
                    finals = data_cum.groupby(grp, observed=True)['value'].transform('max')
                    data_cum['value'] = np.where(finals.gt(0), data_cum['value'] / finals, 0.0)
                return self.line(data=data_cum, x='delay', y='value', group_by=grp,
                                 aggfunc='sum', x_period=None, ax=ax, finalize=finalize, **_passthrough({}))
            else:
                data_cum = data_cum.sort_values('delay')
                data_cum['value'] = data_cum['value'].cumsum()
                if cfg.proportion and not data_cum.empty:
                    final = float(data_cum['value'].iloc[-1])
                    data_cum['value'] = data_cum['value'] / final if final > 0 else 0.0
                return self.line(data=data_cum, x='delay', y='value',
                                 aggfunc='sum', x_period=None, ax=ax, finalize=finalize, **_passthrough({}))

        # Incremental view (default)
        if grp:
            return self.line(data=data_inc if cfg.proportion else base,
                             x='delay', y='value', group_by=grp,
                             aggfunc='sum', x_period=None, ax=ax, finalize=finalize, **_passthrough({}))
        else:
            return self.line(data=data_inc if cfg.proportion else base,
                             x='delay', y='value',
                             aggfunc='sum', x_period=None, ax=ax, finalize=finalize, **_passthrough({}))