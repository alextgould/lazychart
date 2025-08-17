"""lazychart is a lightweight Python helper to rapidly generate static plots from pandas DataFrames.
It features sensible defaults and the option to use *sticky* parameters — it can remember previous arguments
(like data, variables and axis labels) so you can churn out similar charts quickly without repeating boilerplate."""
from typing import TypedDict, Optional, Union, Iterable, Dict, Any, Callable, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from .colors import DEFAULT_PALETTE, COLORBLIND_PALETTE
from functools import wraps
import collections.abc
from collections.abc import Sequence
from textwrap import wrap
from math import ceil

# Logging
from lazychart.log import create_logger
logger = create_logger(__name__)

# Decorators (apply to all chart functions)

def show_chart(func: Callable) -> Callable:
    """
    Decorator to handle showing and/or saving matplotlib-based charts.
    
    Parameters:
        save_path (str, optional): If provided, saves the chart to this path
        show_fig (bool, default: True): Whether to display the chart (and return None to suppress notebook output, rather than fig, ax)
    """
    @wraps(func) # preserves the original function's name, docstring, module etc
    def wrapper(self, *args, save_path: Optional[str] = None, show_fig: bool = True, **kwargs):

        fig, ax = func(self, *args, **kwargs)
        
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        if show_fig:
            plt.show()
            return None
        return fig, ax
    return wrapper

def sticky_args(func):
    """
    Decorator for charting methods to enable 'sticky' argument reuse.

    Parameters:
        - sticky (Bool, default: False): save provided arguments to self._sticky for future use.
        - use_sticky (Boolean, default: True): missing keyword arguments are filled from self._sticky.
    """
    @wraps(func) 
    def wrapper(self, *args, sticky=False, use_sticky=True, **kwargs):

        # save kwargs for future charts
        if sticky:
            self._save_sticky(kwargs)

        # load arguments from past charts unless in current kwargs
        if use_sticky:
            kwargs = self._get_sticky(kwargs)

        # all charts need data
        if kwargs.get('data', None) is None:
            raise ValueError('No data to chart')
        
        return func(self, *args, **kwargs)
    return wrapper

class CommonArgsArgs(TypedDict, total=False):
    """
    Logic for common arguments is handled in the common_args decorator function
    This class allows us to pass these arguments to various chart functions
    via kwargs without having to repeat them while retaining type hints.
    """

    # group smaller categories
    data: pd.DataFrame
    group_by: Optional[str] = None
    group_threshold: Optional[Union[int, float]] = None
    group_other_name: Optional[str] = None
    
    # labels
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None

    # label font size
    title_size: Optional[Union[str, int]] = None
    x_label_size: Optional[Union[str, int]] = None
    y_label_size: Optional[Union[str, int]] = None
    tick_size: Optional[Union[str, int]] = None
    
    # axis ranges
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None

    # axis formats
    x_axis_format: Optional[str] = None
    y_axis_format: Optional[str] = None
    decimals: int = 0
    xtick_rotation: Optional[int] = None
    ytick_rotation: Optional[int] = None

    # legend
    legend: Optional[str] = None

    # grid
    grid_x: bool = False
    grid_y: bool = True

def common_args(func: Callable) -> Callable:
    """
    Decorator to handle common charting arguments.

    Parameters (optional):

        - group up small categories
          - group_by (str): column name of categorical variable
          - group_threshold: number (int) or proportion (float, default: 1.0) of categories to retain
          - group_other_name (str, default: 'Other'): name of new category containing smaller categories
           
        - labels
          - title, subtitle, x_label, y_label (str): Various labels
            for x_label and y_label pass "" to have no label, otherwise variable names are used by default
          - title_size, subtitle, x_label, y_label, tick_size: Font size of labels (int or str preset)
            presets: 'small', 'medium', 'large', 'x-large'
          
        - axes
            - x_min, x_max, y_min, y_max (float): axis limits
            - x_axis_format, y_axis_format (str, optional): None (default), 'percent', 'comma', 'short'
            - decimals (int, default: 0): number of decimal places to display on axes
            - xtick_rotation, ytick_rotation (int): degrees rotation of tick labels

        - other
          - legend (str, default: right): where to place the legend
            presets: right, bottom, overlap, hide
          - grid_x, grid_y (bool): whether to show gridlines (default: hide x, show y)
    """
    
    @wraps(func)
    def wrapper(self, *args, **kwargs):

        # Preprocess data for small category grouping
        data = kwargs.get('data')
        group_by = kwargs.get('group_by')
        group_threshold = kwargs.get('group_threshold')
        if group_by and group_threshold:
            total = data[group_by].value_counts(normalize=True)
            if isinstance(group_threshold, float):    
                # Keep categories covering up to group_threshold proportion
                sorted_cats = total.sort_values(ascending=False)
                cum_props = sorted_cats.cumsum()
                keep_cats = cum_props[cum_props <= group_threshold].index.tolist()
                # Always keep the first category above threshold
                if len(keep_cats) < len(sorted_cats):
                    keep_cats.append(sorted_cats.index[len(keep_cats)])
                small_cats = [cat for cat in sorted_cats.index if cat not in keep_cats]
            else:
                # Keep top N categories, group the rest
                sorted_cats = total.sort_values(ascending=False)
                keep_cats = sorted_cats.index[:group_threshold].tolist()
                small_cats = [cat for cat in sorted_cats.index if cat not in keep_cats]
            group_other_name = kwargs.get('group_other_name')
            
            # avoid modifying caller's dataframe
            data = data.copy()
            data[group_by] = data[group_by].replace(small_cats, group_other_name if group_other_name else "Other")
            kwargs['data'] = data

        # call the charting function
        fig, ax = func(self, *args, **kwargs)

        ## Customise appearance

        # Labels

        # TODO: rethink how we're doing these presets?
        FONT_PRESETS = {
            "small":    {"title": 12, "subtitle": 11, "x_label": 10, "y_label": 10, "tick": 9},
            "medium":   {"title": 16, "subtitle": 14, "x_label": 12, "y_label": 12, "tick": 10},
            "large":    {"title": 20, "subtitle": 18, "x_label": 16, "y_label": 16, "tick": 12},
            "x-large":  {"title": 24, "subtitle": 20, "x_label": 18, "y_label": 18, "tick": 14},
        }

        def resolve_fontsize(val, category):
            if isinstance(val, int):
                return val
            if isinstance(val, str):
                preset = FONT_PRESETS.get(val.lower())
                if preset:
                    return preset[category]
            return FONT_PRESETS["medium"][category]

        # TODO: consider adding subtitle functionality e.g. using suptitle for main title and title for subtitle
        if (x_label := kwargs.get('x_label')):
            ax.set_xlabel(x_label, fontsize=resolve_fontsize(kwargs.get('x_label_size'), "x_label"))
        if not (y_label := kwargs.get('y_label')):
            y_label = kwargs.get('y') if kwargs.get('y') else "Number of rows in dataset"
        if y_label:
            ax.set_ylabel(y_label, fontsize=resolve_fontsize(kwargs.get('y_label_size'), "y_label"))

        tick_size = resolve_fontsize(kwargs.get('tick_size'), "tick")
        ax.tick_params(axis='both', labelsize=tick_size)

        # Axis limits
        if kwargs.get('x_min') is not None or kwargs.get('x_max') is not None:
            ax.set_xlim(kwargs.get('x_min'), kwargs.get('x_max'))
        if kwargs.get('y_min') is not None or kwargs.get('y_max') is not None:
            ax.set_ylim(kwargs.get('y_min'), kwargs.get('y_max'))
            
        # Axis formatting
        def format_axis(ax, which, fmt, decimals):
            axis = ax.xaxis if which == 'x' else ax.yaxis
            if fmt == 'percent':
                axis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=decimals))
            elif fmt == 'comma':
                axis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):,}"))
            elif fmt == 'short':
                def short(x, pos):
                    if abs(x) >= 1_000_000:
                        return f"{x/1_000_000:.{decimals}f}M"
                    if abs(x) >= 1_000:
                        return f"{x/1_000:.{decimals}f}k"
                    return f"{x:.{decimals}f}"
                axis.set_major_formatter(mticker.FuncFormatter(short))

        if kwargs.get('x_axis_format'):
            format_axis(ax, 'x', kwargs['x_axis_format'], kwargs.get('decimals', 0))
        if kwargs.get('y_axis_format'):
            format_axis(ax, 'y', kwargs['y_axis_format'], kwargs.get('decimals', 0))

        if kwargs.get('xtick_rotation') is not None:
            for label in ax.get_xticklabels():
                label.set_rotation(kwargs['xtick_rotation'])
        if kwargs.get('ytick_rotation') is not None:
            for label in ax.get_yticklabels():
                label.set_rotation(kwargs['ytick_rotation'])
        
        # Grid
        grid_x = kwargs.get('grid_x', False)
        grid_y = kwargs.get('grid_y', True)
        if grid_x:
            ax.grid(True, axis='x', linestyle='--', alpha=0.4)
        else:
            ax.grid(False, axis='x')
        if grid_y:
            ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        else:
            ax.grid(False, axis='y')

        return fig, ax
    return wrapper

# Main class

class ChartMonkey:
    def __init__(self, palette: Optional[Union[str, Iterable[str]]] = 'default'):
        """
        Parameters:
        - palette (str, iterable(str), default 'default'): sets the matplotlib prop_cycle colours
          - presets: 'default' (rainbow), 'colourblind'
          - can also pass a list of hex codes
        """
        # keep a dict of sticky arguments
        self._sticky: Dict[str, Any] = {}

        # placeholder for better and more in-depth colour palette treatment
        # (e.g. add to core.py decorator so it only impacts charts rather than using rcParams)
        # (e.g. have a config file where HEX codes can be added for quick brand colour charts)
        self.palette = self._resolve_palette(palette)
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=self.palette)       

    # Sticky parameters

    def _save_sticky(self, kwargs: Dict[str, Any]):
        """Update sticky kwargs with new values"""
        for k, v in kwargs.items():
            self._sticky[k] = v
            
    def _get_sticky(self, kwargs: Dict[str, Any]):
        """Combine sticky kwargs with new values"""
        for key in self._sticky:
            if key not in kwargs.keys():
                kwargs[key] = self._sticky.get(key, None)
        return kwargs

    def clear_sticky(self):
        self._sticky = {}

    # Sample data

    def example_data(self, n: int = 2000, seed: int = 42) -> pd.DataFrame:
        np.random.seed(seed)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=n)
        user = np.random.choice([f'user{i}' for i in range(1, 51)], size=n)
        sleep_hours = np.clip(np.random.normal(7, 1, n), 4, 10)
        steps = np.clip(np.random.normal(6000, 1500, n), 200, 15000)
        alcohol = np.clip(np.random.poisson(1.2, n), 0, 8)
        work_stress = np.clip(np.random.normal(5, 2, n), 0, 10)
        nutrition = np.clip(np.random.normal(6.5, 1.5, n), 0, 10)
        score = 0.4*sleep_hours + 0.001*steps - 0.5*alcohol + 0.3*nutrition - 0.2*work_stress + np.random.normal(0,1,n)
        emotion = pd.cut(score, bins=[-10, -0.5, 0.5, 10], labels=['Engaged', 'Energetic', 'Happy'])
        df = pd.DataFrame(dict(date=dates, user=user, sleep_hours=sleep_hours, steps=steps, alcohol=alcohol, work_stress=work_stress, nutrition=nutrition, score=score, predicted_emotion=emotion))
        df['month'] = df['date'].dt.to_period('M').astype(str)
        df['weekday'] = df['date'].dt.day_name()
        return df
    
    # Palette
    
    def _resolve_palette(self, palette):
        if isinstance(palette, str):
            if palette.lower() == 'default':
                return DEFAULT_PALETTE
            elif palette.lower() == 'colorblind':
                return COLORBLIND_PALETTE
            else:
                raise ValueError(f"Unknown palette preset: {palette}")
        elif isinstance(palette, collections.abc.Iterable) and not isinstance(palette, str):
            # Optionally validate that all elements are strings (hex codes or color names)
            palette_list = list(palette)
            if not all(isinstance(c, str) for c in palette_list):
                raise ValueError("Custom palette must be an iterable of strings (hex codes or color names)")
            return palette_list
        else:
            raise ValueError("Palette must be a string preset or an iterable of color strings")
    
    # Data aggregation
    
    def _aggregate_data(
        self,
        data: pd.DataFrame,
        x: str,
        y: Optional[str] = None,
        group_by: Optional[str] = None,
        aggfunc: Union[str, Callable] = "sum",
        x_period: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Aggregate data for charting, mimicking behaviour of tools like Excel or Power BI.

        Parameters:
            data (pd.DataFrame): Input dataframe.
            x (str): Column name for categories on the x-axis.
            y (str, optional): Column name for values to aggregate. If None, counts will be used.
            group_by (str, optional): Column name for subgrouping within each x category.
            aggfunc (str or callable, default 'sum'): Aggregation function to apply.
                Examples: 'sum', 'mean', 'count', 'median', np.sum, np.mean, etc.
            x_period (str, optional). One of: 'month', 'quarter', 'year', 'week', 'day'.
                  If provided, the function will create a temporary period column
                  from the datetime column named by `x` and aggregate by that period.

        Returns:
            pd.DataFrame:
                - If group_by is provided: pivot table with x as index, group_by categories as columns.
                - If group_by is None: aggregated dataframe with x and aggregated y.
        """

        #logger.debug(f"Aggregating data with x='{x}', y='{y}', group_by='{group_by}', aggfunc={aggfunc}")
        #logger.debug(f"Data shape before aggregation: {data.shape}")
        
        # Defensive copy to avoid modifying caller's dataframe
        df = data.copy()

        # If x_period requested, ensure x is parsed as datetime and create temp column
        if x_period:
            if x_period.lower() not in {'month', 'quarter', 'year', 'week', 'day'}:
                raise ValueError("x_period must be one of: 'month', 'quarter', 'year', 'week', 'day'")
            # try to convert x to datetime if it isn't already
            if not pd.api.types.is_datetime64_any_dtype(df[x]):
                df[x] = pd.to_datetime(df[x], errors='coerce')
            if df[x].isna().any():
                raise ValueError(f"Column '{x}' contains values that cannot be parsed as datetimes.")
            # map to period string
            period_map = {'month': 'M', 'quarter': 'Q', 'year': 'Y', 'week': 'W', 'day': 'D'}
            period_freq = period_map[x_period.lower()]
            tmp_x = f"{x} ({x_period})"
            # use Period -> string to get consistent labels (e.g. '2023-05')
            df[tmp_x] = df[x].dt.to_period(period_freq).astype(str)
            x = tmp_x  # use the temp column from here on

        # Validate group_by
        if group_by:
            if pd.api.types.is_numeric_dtype(df[group_by]) and not isinstance(df[group_by], pd.CategoricalDtype):
                unique_vals = df[group_by].nunique(dropna=True)
                if unique_vals <= 20: # for low cardinality data we can convert for the user
                    df[group_by] = df[group_by].astype('category')
                else:
                    raise ValueError(
                        f"Column '{group_by}' is numeric with {unique_vals} unique values. "
                        "For grouping/legend behaviour, convert it to a categorical type, e.g.:\n\n"
                        f"    df['{group_by}'] = df['{group_by}'].astype('category')\n\n"
                        "See: https://pandas.pydata.org/docs/reference/api/pandas.Categorical.html"
                    )

        # If no y column, use counts
        if y is None:
            if group_by:
                grouped = df.groupby([x, group_by], observed=True).size().unstack(fill_value=0)
            else:
                grouped = df.groupby(x, observed=True).size().reset_index(name="count")
            return grouped

        # If group_by is provided: create pivot table
        if group_by:
            grouped = df.groupby([x, group_by], observed=True)[y].agg(aggfunc).unstack(fill_value=0)
        else:
            grouped = df.groupby(x, as_index=False, observed=True)[y].agg(aggfunc)

        #logger.debug(f"Grouped data shape: {grouped.shape}")
        #logger.debug(f"Grouped head:\n{grouped.head()}")

        return grouped
    
    def _resolve_sort_order(
        self,
        available,
        grouped: Optional[pd.DataFrame] = None,
        sort_hint: Optional[Union[str, Sequence]] = None,
        is_index: bool = False,
        ascending: Optional[bool] = None
    ) -> list:
        """
        Resolve ordering for a single axis (x or group_by).

        Parameters
        - available: iterable of labels currently present (e.g. list(grouped.index) or list(grouped.columns) or unique raw values)
        - grouped: the aggregated/pivot DataFrame (optional). Required when sort_hint == 'value' to compute sums.
        - sort_hint: None | 'label' | 'value' | 'none' | list_of_labels
        - is_index: True if available corresponds to the DataFrame index (use sums axis=1 for 'value'), else columns (axis=0)
        - ascending: optional boolean; if None we fall back to intuitive defaults:
            - label => True for numeric-like (keep chronological), False for categorical (desc)
            - value => False for categorical (desc), True for numeric-like (asc)
        Returns:
        - list: resolved ordering (subset/reorder of available)
        """
        # Normalize available to list of strings (preserves original labels but cast to str for matching)
        avail_list = [str(a) for a in list(available)]

        # If explicit list provided, respect it (append any missing existing labels at the end)
        if isinstance(sort_hint, Sequence) and not isinstance(sort_hint, (str, bytes)):
            requested = [str(x) for x in list(sort_hint)]
            requested_existing = [r for r in requested if r in avail_list]
            missing_existing = [a for a in avail_list if a not in requested_existing]
            return requested_existing + missing_existing

        # If user asked for no sorting, preserve natural available order
        if sort_hint == 'none':
            return avail_list

        # Decide numeric-like hint for default ascending behaviour (only if grouped provided, else conservative False)
        numeric_like = False
        if grouped is not None and len(avail_list) > 0:
            try:
                # Peek first element mapping back to grouped to check dtype only if possible
                numeric_like = False
                if is_index:
                    # try to infer from grouped.index if it has dtype information
                    try:
                        idx = grouped.index
                        numeric_like = pd.api.types.is_numeric_dtype(idx.dtype) or pd.api.types.is_datetime64_any_dtype(idx.dtype)
                    except Exception:
                        numeric_like = False
                else:
                    # infer from grouped columns or its first column values
                    try:
                        # if columns are strings, not helpful — fallback to values in grouped
                        numeric_like = any(pd.api.types.is_numeric_dtype(grouped[c].dtype) for c in grouped.columns[:1])  # rough test
                    except Exception:
                        numeric_like = False
            except Exception:
                numeric_like = False

        # Default ascending if not provided
        if ascending is None:
            if sort_hint == 'label':
                ascending = True if numeric_like else False
            elif sort_hint == 'value':
                ascending = False if not numeric_like else True
            else:
                # fallback to label-like preference
                ascending = True if numeric_like else False

        # Sorting by label (alphabetical / chronological)
        if sort_hint == 'label':
            return sorted(avail_list, reverse=not ascending)

        # Sorting by value: requires grouped DataFrame to compute sums
        if sort_hint == 'value':
            if grouped is None:
                # no information to sort by value; fall back to available order
                return avail_list
            if is_index:
                # grouped: rows are the x categories -> sum across columns to compare row totals
                sums = grouped.sum(axis=1)
                ordered = [str(i) for i in sums.sort_values(ascending=ascending).index.tolist()]
                # ensure only available ones are returned (safety)
                return [o for o in ordered if o in avail_list]
            else:
                # columns case: sum across rows for each column
                try:
                    col_sums = grouped.sum(axis=0)
                    ordered = [str(c) for c in col_sums.sort_values(ascending=ascending).index.tolist()]
                    return [o for o in ordered if o in avail_list]
                except Exception:
                    return avail_list

        # Default case: preserve available order
        return avail_list
    
    # Layout - Title, Legends

    @staticmethod
    def _legend_handle_and_padding_width(fig):
        """Derive figures to help split legend between text and non-text components for wrapping purposes"""
        fontsize = FontProperties(size=rcParams['legend.fontsize']).get_size_in_points()
        dpi = fig.dpi
        # Convert "em" units to points → pixels
        em_to_px = fontsize * dpi / 72.0

        handle_px = rcParams['legend.handlelength'] * em_to_px
        pad_px = rcParams['legend.handletextpad'] * em_to_px
        return handle_px + pad_px
    
    def _legend_wrap(self, fig, labels, max_ratio=0.3):
        """
        Automatically wraps legend labels so they fit within max_ratio of figure width.
        max_ratio is the total fraction of figure width allocated to each legend entry 
        (including handle + padding). 
        """
        # Figure and font settings
        fig_width_px = fig.get_size_inches()[0] * fig.dpi
        fontsize = FontProperties(size=rcParams['legend.fontsize']).get_size_in_points()
        dpi = fig.dpi

        # Approximate character width in pixels
        char_px = fontsize * dpi / 72.0 * 0.6

        # Available text width after handle + padding
        handle_pad_px = self._legend_handle_and_padding_width(fig)
        max_label_px = fig_width_px * max_ratio - handle_pad_px

        # Convert to max characters allowed
        max_chars = int(max_label_px // char_px)

        # If all labels already fit, return unchanged
        if max(len(lbl) for lbl in labels) <= max_chars:
            return labels

        # Otherwise wrap
        return ['\n'.join(wrap(lbl, max_chars)) for lbl in labels]

    def _legend_ncol(self, fig, ax, legend_pos, max_ratio=0.9):
        """Determine optimal number of legend columns"""

        handles, labels = ax.get_legend_handles_labels()
        if not labels:
            return 1  # nothing to do

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

    @staticmethod
    def _unique_by_label(handles, labels):
        """Remove duplicate legend entries by label."""
        seen = set()
        uh, ul = [], []
        for h, l in zip(handles, labels):
            if l and not l.startswith("_") and l not in seen:
                seen.add(l)
                uh.append(h)
                ul.append(l)
        return uh, ul

    @staticmethod
    def _wrap_title(fig, title: str, max_ratio: float = 0.9) -> str:
        """Wrap the title text so it fits within the figure width."""
        fig_w_px = fig.get_size_inches()[0] * fig.dpi
        fontsize = FontProperties(size=rcParams['axes.titlesize']).get_size_in_points()
        char_px = fontsize * fig.dpi / 72.0 * 0.6
        max_chars = int((fig_w_px * max_ratio) // char_px)
        return "\n".join(wrap(title, max_chars))
    
    @staticmethod
    def _make_figure(
        chart2: bool = False,
        default_size: Tuple[float, float] = (6.4, 4.8),
    ) -> Tuple[Figure, Union[Axes, Dict[str, Axes]]]:
        """
        Create a blank figure and axes.
        For single charts -> (fig, ax).
        For combo charts  -> (fig, {"D1": ax1, "D2": ax2}).
        """

        if chart2:
            figsize = (default_size[0] * 2, default_size[1])
            fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=False)
            axes = {"D1": axs[0], "D2": axs[1]}
        else:
            figsize = default_size
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
            axes = ax

        return fig, axes
    
    def _finalize_layout(
        self, fig: Figure, axes: Union[Axes, Dict[str, Axes]],
        legend: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """Adjust legend, title, and figure size after plotting."""

        #logger.debug(f"_finalize_layout: initial fig size: {fig.get_size_inches()}")
        
        if not isinstance(axes, dict):
            axes = {"D1": axes}

        handles, labels = [], []
        for ax in axes.values():
            h, l = ax.get_legend_handles_labels()
            handles += h
            labels += l

        if not handles or not legend:
            if title:
                fig.suptitle(self._wrap_title(fig, title))
            fig.tight_layout()
            return

        # Dedup + wrap + ncol
        handles, labels = self._unique_by_label(handles, labels)
        if legend == 'right':
            labels = self._legend_wrap(fig, labels)

        # Decide number of columns dynamically
        ncol = self._legend_ncol(fig, list(axes.values())[0], legend)

        # Temporary legend for sizing
        temp_leg = fig.legend(handles, labels, ncol=ncol, loc='upper right')  # fully inside
        fig.canvas.draw()
        bb = temp_leg.get_window_extent().transformed(fig.transFigure.inverted())
        temp_leg.remove()

        fig_width, fig_height = fig.get_size_inches()
        leg_width, leg_height = fig_width * bb.width, fig_height * bb.height

        #logger.debug(f"_finalize_layout: temp_leg bb {bb}")
        #logger.debug(f"_finalize_layout: fig_width {fig_width} leg_width {leg_width}, fig_height {fig_height}, leg_height {leg_height}")

        # # Expand figure to acccomodate the legend
        if True: # figure resizing
            if legend == "right":
                fig.set_size_inches(fig_width + leg_width, fig_height, forward=True)
            elif legend == "bottom":
                fig.set_size_inches(fig_width, fig_height + leg_height, forward=True)
            fig.canvas.draw()
        
        #logger.debug(f"_finalize_layout: fig size after expansion: {fig.get_size_inches()}")
        
        # Adjust layout
        if True: # rect scaling
            if legend == "right":
                rect = [0, 0, 1 / (1 + bb.width), 1]
            else:  # bottom
                rect = [0, bb.height / (1 + bb.height), 1, 1]
        else:
            rect = [0, 0, 1, 1]

        if title:
            fig.suptitle(self._wrap_title(fig, title))

        #logger.debug(f"_finalize_layout: calling tight_layout with rect: {rect}")
        
        fig.tight_layout(rect=rect) # rect is [left, bottom, right, top]

        #for name, ax in axes.items():
        #    logger.debug(f"_finalize_layout: ax {name} pos: {ax.get_position().bounds}")

        loc = "center right" if legend == "right" else "lower center"
        bbox = (1.0, 0.5) if legend == "right" else (0.5, 0)
        fig.legend(handles, labels, ncol=ncol, loc=loc, bbox_to_anchor=bbox)

        final_leg = fig.legend(handles, labels, ncol=ncol, loc=loc, bbox_to_anchor=bbox)
        fig.canvas.draw()
        bb_final = final_leg.get_window_extent().transformed(fig.transFigure.inverted())
        #logger.debug(f"_finalize_layout: final legend bb {bb_final}")

        #logger.debug(f"_finalize_layout: final fig size: {fig.get_size_inches()}")

    # Charts

    def _bar(
        self,
        ax: Axes,
        data: pd.DataFrame,
        x: str,
        y: Optional[str] = None,
        group_by: Optional[str] = None,
        aggfunc: Union[str, Callable] = "sum",
        stacking: str = "none",
        sort_x: Optional[str] = None,
        sort_group_by: Optional[str] = None,
        sort_x_ascending: Optional[bool] = None,
        sort_group_by_ascending: Optional[bool] = None,
        x_period: Optional[str] = None,
    ):
        
        if stacking not in {"none", "standard", "proportion"}:
            raise ValueError("stacking must be one of: 'none', 'standard', 'proportion'")

        grouped = self._aggregate_data(
            data, x, y, group_by, aggfunc,
            x_period=x_period if "x_period" in self._aggregate_data.__code__.co_varnames else None
        )

        if group_by:
            grouped.columns = grouped.columns.astype(str)
            grouped.index = grouped.index.astype(str)

            if stacking == "proportion":
                row_sums = grouped.sum(axis=1).replace({0: np.nan})
                grouped = grouped.div(row_sums, axis=0).fillna(0)

            cols_order = self._resolve_sort_order(
                available=list(grouped.columns),
                grouped=grouped,
                sort_hint=sort_group_by,
                is_index=False,
                ascending=sort_group_by_ascending,
            )
            if cols_order: grouped = grouped[cols_order]

            idx_order = self._resolve_sort_order(
                available=list(grouped.index),
                grouped=grouped,
                sort_hint=sort_x,
                is_index=True,
                ascending=sort_x_ascending,
            )
            if idx_order: grouped = grouped.loc[idx_order]

            grouped.plot(kind="bar", stacked=(stacking != "none"), ax=ax, legend=False)

        else:
            value_col = "count" if y is None else y
            if isinstance(grouped, pd.Series):
                grouped = grouped.reset_index(name=value_col)
            grouped[x] = grouped[x].astype(str)

            if isinstance(sort_x, Sequence) and not isinstance(sort_x, (str, bytes)):
                requested = [str(s) for s in sort_x]
                requested_existing = [r for r in requested if r in grouped[x].unique()]
                missing_existing = [v for v in grouped[x].unique() if v not in requested_existing]
                grouped = grouped.set_index(x).reindex(requested_existing + missing_existing).reset_index()
            else:
                order = self._resolve_sort_order(
                    available=list(grouped[x].unique()),
                    grouped=grouped.set_index(x) if value_col in grouped.columns else None,
                    sort_hint=sort_x,
                    is_index=False,
                    ascending=sort_x_ascending,
                )
                if order:
                    order_existing = [o for o in order if o in grouped[x].unique()]
                    leftovers = [v for v in grouped[x].unique() if v not in order_existing]
                    grouped = grouped.set_index(x).reindex(order_existing + leftovers).reset_index()

            grouped.plot(kind="bar", x=x, y=value_col, ax=ax, legend=False)

    @show_chart
    @sticky_args
    @common_args
    def bar(
        self,
        data: pd.DataFrame,
        x: str,
        y: Optional[str] = None,
        group_by: Optional[str] = None,
        aggfunc: Union[str, Callable] = "sum",
        stacking: str = "none",
        sort_x: Optional[str] = None,
        sort_group_by: Optional[str] = None,
        sort_x_ascending: Optional[bool] = None,
        sort_group_by_ascending: Optional[bool] = None,
        x_period: Optional[str] = None,
        **kwargs: CommonArgsArgs
    ):
        """
        Create a bar chart with optional grouping, stacking, and automatic aggregation.

        Parameters:
            data (pd.DataFrame): Input dataframe.
            x (str): Column name for x-axis categories.
            y (str, optional): Column name for y-axis values. If None, counts will be plotted.
            group_by (str, optional): Column name for grouping within x-axis categories.
            aggfunc (str or callable, default 'sum'): Aggregation function for y-values.
                Examples: 'sum', 'mean', 'count', 'median', np.sum, np.mean, etc.
            stacking (str, default 'none'): One of:
                - 'none' (side-by-side bars)
                - 'standard' (stacked bars)
                - 'proportion' (100% stacked bars)
            sort_x (str, optional): How to sort x-axis categories. Attempts to sort intuitively by default.
                One of: 'label', 'value', 'none'.
            sort_group_by (str, optional): How to sort group_by categories. Attempts to sort intuitively by default
                One of: 'label', 'value', 'none'.
            sort_x_ascending (bool, optional): Whether to sort x-axis categories in ascending order. Attempts to sort intuitively by default.
            sort_group_by_ascending (bool, optional): Whether to sort group_by categories in descending order. Attempts to sort intuitively by default.
            **kwargs: Additional common_args customisation (title, label, axis limit, etc.)

        Returns:
            (fig, ax): Matplotlib Figure and Axes objects (unless show_fig=True, then returns None).
        """

        fig, ax = self._make_figure()

        # Draw chart
        self._bar(
            ax=ax,
            data=data, x=x, y=y,
            group_by=group_by, aggfunc=aggfunc, stacking=stacking,
            sort_x=sort_x, sort_group_by=sort_group_by,
            sort_x_ascending=sort_x_ascending,
            sort_group_by_ascending=sort_group_by_ascending,
            x_period=x_period,
        )
        self._finalize_layout(fig, ax, legend=kwargs.get("legend"), title=kwargs.get("title"))

        return fig, ax

    def _line(
        self,
        ax: Axes,
        data: pd.DataFrame,
        x: str,
        y: Union[str, Sequence[str]],
        group_by: Optional[str] = None,
        aggfunc: Union[str, Callable] = "sum",
        sort_x: Optional[str] = None,
        sort_group_by: Optional[str] = None,
        sort_x_ascending: Optional[bool] = None,
        sort_group_by_ascending: Optional[bool] = None,
        x_period: Optional[str] = None,
    ):
        
        # Ensure y is a list
        if isinstance(y, str):
            y = [y]

        grouped = self._aggregate_data(
            data, x, y, group_by, aggfunc,
            x_period=x_period if "x_period" in self._aggregate_data.__code__.co_varnames else None
        )

        # If grouped is multi-column (multiple y or group_by), pandas plots multiple lines
        if group_by:
            grouped.columns = grouped.columns.astype(str)
            grouped.index = grouped.index.astype(str)

            cols_order = self._resolve_sort_order(
                available=list(grouped.columns),
                grouped=grouped,
                sort_hint=sort_group_by,
                is_index=False,
                ascending=sort_group_by_ascending,
            )
            if cols_order:
                grouped = grouped[cols_order]

            idx_order = self._resolve_sort_order(
                available=list(grouped.index),
                grouped=grouped,
                sort_hint=sort_x,
                is_index=True,
                ascending=sort_x_ascending,
            )
            if idx_order:
                grouped = grouped.loc[idx_order]

            grouped.plot(kind="line", ax=ax, legend=False)

        else:
            # Simple case: x vs y (one or more series)
            data[x] = data[x].astype(str)

            order = self._resolve_sort_order(
                available=list(data[x].unique()),
                grouped=data.set_index(x)[y] if y else None,
                sort_hint=sort_x,
                is_index=False,
                ascending=sort_x_ascending,
            )
            if order:
                ordered_idx = pd.Categorical(data[x], categories=order, ordered=True)
                data = data.assign(_ordered=ordered_idx).sort_values("_ordered").drop(columns="_ordered")

            data.plot(x=x, y=y, kind="line", ax=ax, legend=False)

    @show_chart
    @sticky_args
    @common_args
    def line(
        self,
        data: pd.DataFrame,
        x: str,
        y: Union[str, Sequence[str]],
        group_by: Optional[str] = None,
        aggfunc: Union[str, Callable] = "sum",
        sort_x: Optional[str] = None,
        sort_group_by: Optional[str] = None,
        sort_x_ascending: Optional[bool] = None,
        sort_group_by_ascending: Optional[bool] = None,
        x_period: Optional[str] = None,
        **kwargs: CommonArgsArgs
    ):
        """TODO: fix docstrings"""
        
        fig, ax = self._make_figure()

        # Draw chart
        self._line(
            ax=ax,
            data=data, x=x, y=y,
            group_by=group_by, aggfunc=aggfunc,
            sort_x=sort_x, sort_group_by=sort_group_by,
            sort_x_ascending=sort_x_ascending,
            sort_group_by_ascending=sort_group_by_ascending,
            x_period=x_period,
        )

        return fig, ax

    def _pie(
        self,
        ax: Axes,
        data: pd.DataFrame,
        values: str,
        names: Optional[str] = None,
        aggfunc: Union[str, Callable] = "sum",
        sort_names: Optional[str] = None,
        sort_names_ascending: Optional[bool] = None,
    ):
        
        # Aggregate if category column provided
        if names:
            grouped = data.groupby(names, as_index=True)[values].agg(aggfunc)
        else:
            grouped = data[values]

        # Optional sorting
        order = self._resolve_sort_order(
            available=list(grouped.index.astype(str)),
            grouped=grouped,
            sort_hint=sort_names,
            is_index=True,
            ascending=sort_names_ascending,
        )
        if order:
            grouped = grouped.loc[order]

        # Draw pie — no legend (we’ll handle that outside)
        wedges, texts, autotexts = ax.pie(
            grouped,
            labels=None,  # suppress labels inside slices
            autopct=None,
            startangle=90
        )

        # Attach handles+labels to Axes for external legend use
        ax._pie_handles = wedges
        ax._pie_labels = [str(l) for l in grouped.index]

    @show_chart
    @sticky_args
    @common_args
    def pie(
        self,
        data: pd.DataFrame,
        values: str,
        names: Optional[str] = None,
        aggfunc: Union[str, Callable] = "sum",
        sort_names: Optional[str] = None,
        sort_names_ascending: Optional[bool] = None,
        **kwargs: CommonArgsArgs
    ):
        """TODO: fix docstrings"""

        fig, ax = self._make_figure()

        # Draw chart
        self._pie(
            ax=ax,
            data=data,
            values=values,
            names=names,
            aggfunc=aggfunc,
            sort_names=sort_names,
            sort_names_ascending=sort_names_ascending,
        )
        
        return fig, ax
    
    # Combo charts

    @show_chart
    @sticky_args
    # @common_args # needs a rewrite to handle separate axes
    # consider replacing the decorator approach with an explicit call similar to _finalize_layout e.g. _style_data_axes which is called in charts but not combo chart
    # also consider building sticky_args into show_chart... and/or consider how sticky_args will work with combo charts...
    def combo(
        self,
        chart1: Callable[[Axes], None],
        chart2: Callable[[Axes], None],
        legend: Optional[str] = "right",
        title: Optional[str] = None
    ):
        """
        Create a combo chart with two subcharts side by side.

        Parameters
        ----------
        chart1 : Callable[[Axes], None]
            A plotting function that draws into the left axes (D1).
        chart2 : Callable[[Axes], None]
            A plotting function that draws into the right axes (D2).
        legend : {"right", "bottom", None}, optional
            Placement of a shared figure-level legend spanning both charts.
        """

        fig, axes = self._make_figure(chart2=True, legend=legend)
        
        # Draw charts
        chart1(axes["D1"])
        chart2(axes["D2"])

        self._finalize_layout(fig, axes, legend=legend, title=title)

        return fig, axes