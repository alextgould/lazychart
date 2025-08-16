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
from matplotlib.gridspec import GridSpec
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
    subtitle: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None

    # label font size
    title_size: Optional[Union[str, int]] = None
    subtitle_size: Optional[Union[str, int]] = None
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
    
    def legend_handle_and_padding_width(fig):
        """Derive figures to help split legend between text and non-text components for wrapping purposes"""
        fontsize = FontProperties(size=rcParams['legend.fontsize']).get_size_in_points()
        dpi = fig.dpi
        # Convert "em" units to points → pixels
        em_to_px = fontsize * dpi / 72.0

        handle_px = rcParams['legend.handlelength'] * em_to_px
        pad_px = rcParams['legend.handletextpad'] * em_to_px
        return handle_px + pad_px

    def legend_ncol(fig, ax, legend, max_ratio=0.9):
        fontsize = FontProperties(size=rcParams['legend.fontsize']).get_size_in_points()
        dpi = fig.dpi
        char_px = fontsize * dpi / 72.0 * 0.6
        _, labels = ax.get_legend_handles_labels()
        handle_pad_px = legend_handle_and_padding_width(fig)
        text_heights = [fontsize * dpi / 72.0 * 1.2 for _ in labels]  # approx height per label in px

        if legend == 'bottom':
            fig_width_px = fig.get_size_inches()[0] * fig.dpi # use width
            max_width_px = fig_width_px * max_ratio
            text_widths = [len(lbl) * char_px + handle_pad_px for lbl in labels]
            avg_width = sum(text_widths) / len(text_widths)
            ideal_ncol = max(1, int(max_width_px // avg_width))
            rows = ceil(len(labels) / ideal_ncol)
            ncol = ceil(len(labels) / rows)
            return max(1, min(len(labels), ncol))
        else:
            fig_height_px = fig.get_size_inches()[1] * fig.dpi # use height
            max_height_px = fig_height_px * max_ratio
            avg_label_h = sum(text_heights) / len(text_heights)
            max_rows = max(1, int(max_height_px // avg_label_h))
            if max_rows >= len(labels):
                return 1
            else:
                ncol = ceil(len(labels) / max_rows)
                return max(1, min(len(labels), ncol))
    
    def legend_wrap(fig, labels, max_ratio=0.3):
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
        handle_pad_px = legend_handle_and_padding_width(fig)
        max_label_px = fig_width_px * max_ratio - handle_pad_px

        # Convert to max characters allowed
        max_chars = int(max_label_px // char_px)

        # If all labels already fit, return unchanged
        if max(len(lbl) for lbl in labels) <= max_chars:
            return labels

        # Otherwise wrap
        return ['\n'.join(wrap(lbl, max_chars)) for lbl in labels]
    
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

        if (title := kwargs.get('title')):
            #ax.set_title(title, fontsize=resolve_fontsize(kwargs.get('title_size'), "title"))
            fig.suptitle(title, fontsize=resolve_fontsize(kwargs.get('title_size'), "title")) # doesn't seem to work
        if (subtitle := kwargs.get('subtitle')) and hasattr(ax, "set_subtitle"):
            ax.set_subtitle(subtitle, fontsize=resolve_fontsize(kwargs.get('subtitle_size'), "subtitle"))
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

        # Legend handling - designed for single axes, may need to adjust for subplot_mosaic combo charts
        if kwargs.get('group_by'):
            ax.legend_.remove()  # Removes auto-added legend if it exists
            if kwargs.get('legend', None) == 'bottom':
                ncol = legend_ncol(fig, ax, 'bottom')
                fig.legend(loc='outside lower center', ncol=ncol, title=kwargs.get('group_by'))
            elif kwargs.get('legend', None) == 'right':
                handles, labels = ax.get_legend_handles_labels()
                wrapped_labels = legend_wrap(fig, labels)
                ncol = legend_ncol(fig, ax, 'right')
                fig.legend(handles, wrapped_labels, ncol=ncol, loc='outside right upper', title=kwargs.get('group_by'))
                # TODO(test): fig.legend(handles, wrapped_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title=kwargs.get('group_by'))
            else:
                fig.legend(title=kwargs.get('group_by'))
        
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

    @show_chart
    @sticky_args
    @common_args
    def bar_original(
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

        fig, ax = plt.subplots(constrained_layout=True)

        if stacking not in {"none", "standard", "proportion"}:
            raise ValueError("stacking must be one of: 'none', 'standard', 'proportion'")

        # Aggregate (this respects x_period if your _aggregate_data accepts it)
        grouped = self._aggregate_data(data, x, y, group_by, aggfunc, x_period=x_period if 'x_period' in self._aggregate_data.__code__.co_varnames else None)

        # If group_by provided, grouped is pivot-like (index = x, columns = group_by)
        if group_by:
            # Ensure columns/index are strings for consistent comparisons (we keep underlying values in grouped)
            grouped.columns = grouped.columns.astype(str)
            grouped.index = grouped.index.astype(str)

            # 100% stacked proportions -> convert rows to proportions first (before sorting by value if needed)
            if stacking == "proportion":
                # avoid division by zero
                row_sums = grouped.sum(axis=1).replace({0: np.nan})
                grouped = grouped.div(row_sums, axis=0).fillna(0)

            # Resolve ordering for group_by categories (columns)
            cols_order = self._resolve_sort_order(
                available=list(grouped.columns.astype(str)),
                grouped=grouped,
                sort_hint=sort_group_by,
                is_index=False,
                ascending=sort_group_by_ascending
            )
            # Reorder columns preserving only those that exist
            cols_order = [c for c in cols_order if c in grouped.columns]
            if cols_order:
                grouped = grouped[cols_order]

            # Resolve ordering for x categories (index)
            idx_order = self._resolve_sort_order(
                available=list(grouped.index.astype(str)),
                grouped=grouped,
                sort_hint=sort_x,
                is_index=True,
                ascending=sort_x_ascending
            )
            idx_order = [i for i in idx_order if i in grouped.index]
            if idx_order:
                grouped = grouped.loc[idx_order]

            # Finally plot: stacked if stacking != 'none'
            grouped.plot(kind="bar", stacked=(stacking != "none"), ax=ax)

        else:
            # Non-grouped case: grouped is a DataFrame either with columns [x, 'count'] or aggregated with as_index=False
            # Normalize column names: ensure x exists and values are present
            if y is None:
                value_col = "count"
            else:
                value_col = y

            # If grouped came back as a Series-like (size grouped), ensure it's a DataFrame with columns
            if isinstance(grouped, pd.Series):
                grouped = grouped.reset_index(name=value_col)

            # Cast x column to string for explicit-list matching
            grouped[x] = grouped[x].astype(str)

            # Sorting when explicit list provided for sort_x
            if isinstance(sort_x, Sequence) and not isinstance(sort_x, (str, bytes)):
                requested = [str(s) for s in list(sort_x)]
                requested_existing = [r for r in requested if r in grouped[x].unique()]
                missing_existing = [a for a in grouped[x].unique() if a not in requested_existing]
                explicit_order = requested_existing + missing_existing
                # set index to x, reindex according to explicit order
                grouped = grouped.set_index(x).reindex(explicit_order).reset_index()
            else:
                # Use helper to compute an ordering (value-based or label)
                order = self._resolve_sort_order(
                    available=list(grouped[x].astype(str).unique()),
                    grouped=grouped.set_index(x) if value_col in grouped.columns else None,
                    sort_hint=sort_x,
                    is_index=False,
                    ascending=sort_x_ascending
                )
                # If we got an ordering, reindex accordingly
                if order:
                    # make sure order contains only present values
                    order_existing = [o for o in order if o in grouped[x].astype(str).unique()]
                    if set(order_existing) != set(grouped[x].astype(str).unique()):
                        # include any leftovers at end
                        leftovers = [v for v in grouped[x].astype(str).unique() if v not in order_existing]
                        order_existing += leftovers
                    grouped = grouped.set_index(x).reindex(order_existing).reset_index()

            # Now plot: basic bar with no legend (common_args handles legend)
            grouped.plot(kind="bar", x=x, y=value_col, ax=ax, legend=False)

        return fig, ax
    
    def _make_figure(
        chart1: Callable[[Axes], None],
        chart2: Optional[Callable[[Axes], None]] = None,
        legend1: Optional[str] = None, # left chart
        legend2: Optional[str] = None, # right chart
        legend: Optional[str] = None, # figure
    ) -> Tuple[Figure, Dict[str, Axes]]:
        """
        Create a flexible figure layout for one or two charts (combo).
        
        This helper sets up a `subplot_mosaic` with slots for:
        - TS (title space)
        - D1 (left chart), RL1 (right legend for left chart)
        - D2 (right chart, optional), RL2 (right legend for right chart, or combo with figure legend 'right')
        - BL1, BL2 (bottom legend areas)
        - FL (bottom figure legend that spans both BL1 and BL2)

        Parameters
        ----------
        chart1 : Callable[[Axes], None]
            A plotting function (e.g. self._bar) that draws into an Axes.
        chart2 : Callable[[Axes], None], optional
            A second plotting function for side-by-side combo charts.
        legend1 : {"right", "bottom", None}, optional
            Placement of legend for chart1.
        legend2 : {"right", "bottom", None}, optional
            Placement of legend for chart2 (only if chart2 is provided).
        legend : {"right", "bottom", None}, optional
            Placement of shared legend for combo charts ('right' will place it in the chart2 'right' region).

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : dict[str, matplotlib.axes.Axes]
            Dictionary of axes keyed by ["TS", "D1", "RL1", "D2", "RL2", "BL1", "BL2", "FL"].
        """

        LAYOUT = [
        #     w[0]   w[1]    w[2]   w[3]
            ["TS",  "TS",  "TS",  "TS"],  # h[0]
            ["D1",  "RL1", "D2",  "RL2"], # h[1]
            ["BL1", "BL1", "BL2", "BL2"], # h[2]
            ["FL",  "FL",  "FL",  "FL"]   # h[3]
        ]   

        # Figure segment width and height values in inches
        # standard figure size is [6.4, 4.8] and standard figure would have D1 and possibly TS above and RL on right or BL on bottom
        # may need to tweak these to get a decent "average" starting point
        w = [5.2, 1.2, 5.2, 1.2]  # Base: D1, RL1, D2, RL2 - assume right legends take up about 20% of a standard chart width
        h = [0.6, 3.9, 0.9, 0.9]  # Base: TS, data row, bottom legend row - assume bottom legends take up about 20% of a standard chart height

        # Collapse D2 & RL2 columns if not a combo chart
        if not chart2:
            w[2] = 0  # D2
            w[3] = 0  # RL2

        # Collapse unused legends
        if legend == 'right': # figure legend in RL2
            w[1] = 0  # RL1
            h[2] = 0  # BL1, BL2
            h[3] = 0 # FL
        elif legend == 'bottom': # figure legend in FL
            w[1] = 0  # RL1
            w[3] = 0  # RL2
            h[2] = 0 # BL1, BL2
        else: # individual chart1/chart2 legends
            h[3] = 0 # FL
            if (legend1 != 'bottom' and legend2 != 'bottom'):
                h[2] = 0  # BL1, BL2
            if legend1 != 'right':
                w[1] = 0  # RL1
            if legend2 != 'right':
                w[3] = 0  # RL2

        # TODO: consider making additional adjustments based on legend row/ncol calcs (for larger legends), but may need a temp chart to resolve this?
        
        # Compute final fig size
        fig_w = sum(w)
        fig_h = sum(h)

        fig, axes = plt.subplot_mosaic(LAYOUT, figsize=(fig_w, fig_h), gridspec_kw={"width_ratios": w, "height_ratios": h}, constrained_layout=True)

        # Disable axes for legends/title areas
        for key in ["TS", "RL1", "RL2", "BL1", "BL2", "FL"]:
            axes[key].set_axis_off()

        # Create charts
        chart1(axes["D1"])
        if chart2:
            chart2(axes["D2"])

        return fig, axes
    
    
    



    # TODO: I'm attempting to split bar_original into bar() and _bar() in line with the new approach
    # the bar() has all the docstrings and is user facing, but actually passes most of the heavy lifting to the _bar function
    # need to check/refine this attempt
    def bar_rework_original(
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

        # determine legend location
        legend = kwargs.get('legend', None) # TODO: confirm default if no 'legend' is 'right' vs 'none' vs None
        legend = legend if kwargs.get('group_by') else 'none' # TODO: confirm legends are only appropriate when you have a group_by variable

        # prepare the figure and axes
        fig, axes = self._make_figure(chart1='bar', legend1=legend)

        # add the chart and legend to the appropriate axes
        ax = axes["D1"]
        # TODO: confirm if we need to add kwarg values into the kwargs dict e.g. data=... is passed, but we only pass kwargs through to _bar
        self._bar(ax=ax, **kwargs) # TODO: confirm _bar doesn't need a return value? it just modifies the ax in place? also how to pass this through?
        if legend == "right":
            axes["RL1"].legend(*ax.get_legend_handles_labels(), loc="center left")
        elif legend == "bottom":
            axes["BL1"].legend(*ax.get_legend_handles_labels(), loc="center")
        return fig, axes # TODO: this was original return fig. think about decorators and how return values will propogate up the decorator chain
    
    def _bar_rework_original(self, ax, *args, **kwargs): # TODO: give me a refresher on the unpack list * and why we have * for args and ** for kwargs. ELI15 :p I don't want ELI5 metaphors but I need a simple worked example to help it stick in my brain

        # unpack values
        # TODO: confirm if there's a quicker way to do this which isn't hugely anti-pythonic
        data = kwargs.get('data')
        x = kwargs.get('x')
        y = kwargs.get('y')
        group_by = kwargs.get('group_by')
        aggfunc = kwargs.get('aggfunc')
        stacking = kwargs.get('stacking')
        sort_x = kwargs.get('sort_x')
        sort_group_by = kwargs.get('sort_group_by')
        sort_x_ascending = kwargs.get('sort_x_ascending')
        sort_group_by_ascending = kwargs.get('sort_group_by_ascending')
        x_period = kwargs.get('x_period')

        # check values which are specific to this chart type so can't be checked earlier
        if stacking not in {"none", "standard", "proportion"}:
            raise ValueError("stacking must be one of: 'none', 'standard', 'proportion'")
        
        # Aggregate (this respects x_period if your _aggregate_data accepts it)
        # TODO: understand what self._aggregate_data.__code__.co_varnames does
        grouped = self._aggregate_data(data, x, y, group_by, aggfunc, x_period=x_period if 'x_period' in self._aggregate_data.__code__.co_varnames else None)

        # If group_by provided, grouped is pivot-like (index = x, columns = group_by)
        if group_by:
            # Ensure columns/index are strings for consistent comparisons (we keep underlying values in grouped)
            grouped.columns = grouped.columns.astype(str)
            grouped.index = grouped.index.astype(str)

            # 100% stacked proportions -> convert rows to proportions first (before sorting by value if needed)
            if stacking == "proportion":
                # avoid division by zero
                row_sums = grouped.sum(axis=1).replace({0: np.nan})
                grouped = grouped.div(row_sums, axis=0).fillna(0)

            # Resolve ordering for group_by categories (columns)
            cols_order = self._resolve_sort_order(
                available=list(grouped.columns.astype(str)),
                grouped=grouped,
                sort_hint=sort_group_by,
                is_index=False,
                ascending=sort_group_by_ascending
            )
            # Reorder columns preserving only those that exist
            cols_order = [c for c in cols_order if c in grouped.columns]
            if cols_order:
                grouped = grouped[cols_order]

            # Resolve ordering for x categories (index)
            idx_order = self._resolve_sort_order(
                available=list(grouped.index.astype(str)),
                grouped=grouped,
                sort_hint=sort_x,
                is_index=True,
                ascending=sort_x_ascending
            )
            idx_order = [i for i in idx_order if i in grouped.index]
            if idx_order:
                grouped = grouped.loc[idx_order]

            # Finally plot: stacked if stacking != 'none'
            grouped.plot(kind="bar", stacked=(stacking != "none"), ax=ax)

        else:
            # Non-grouped case: grouped is a DataFrame either with columns [x, 'count'] or aggregated with as_index=False
            # Normalize column names: ensure x exists and values are present
            if y is None:
                value_col = "count"
            else:
                value_col = y

            # If grouped came back as a Series-like (size grouped), ensure it's a DataFrame with columns
            if isinstance(grouped, pd.Series):
                grouped = grouped.reset_index(name=value_col)

            # Cast x column to string for explicit-list matching
            grouped[x] = grouped[x].astype(str)

            # Sorting when explicit list provided for sort_x
            if isinstance(sort_x, Sequence) and not isinstance(sort_x, (str, bytes)):
                requested = [str(s) for s in list(sort_x)]
                requested_existing = [r for r in requested if r in grouped[x].unique()]
                missing_existing = [a for a in grouped[x].unique() if a not in requested_existing]
                explicit_order = requested_existing + missing_existing
                # set index to x, reindex according to explicit order
                grouped = grouped.set_index(x).reindex(explicit_order).reset_index()
            else:
                # Use helper to compute an ordering (value-based or label)
                order = self._resolve_sort_order(
                    available=list(grouped[x].astype(str).unique()),
                    grouped=grouped.set_index(x) if value_col in grouped.columns else None,
                    sort_hint=sort_x,
                    is_index=False,
                    ascending=sort_x_ascending
                )
                # If we got an ordering, reindex accordingly
                if order:
                    # make sure order contains only present values
                    order_existing = [o for o in order if o in grouped[x].astype(str).unique()]
                    if set(order_existing) != set(grouped[x].astype(str).unique()):
                        # include any leftovers at end
                        leftovers = [v for v in grouped[x].astype(str).unique() if v not in order_existing]
                        order_existing += leftovers
                    grouped = grouped.set_index(x).reindex(order_existing).reset_index()

            # Now plot: basic bar with no legend (common_args handles legend)
            grouped.plot(kind="bar", x=x, y=value_col, ax=ax, legend=False)

    @show_chart
    @sticky_args # TODO: consider how this will function if we pass different kwargs to the two charts e.g. x=col1 y=col2 and then x=col1 y=col3
    @common_args
    def combo(chart1, chart2, legend="right", sticky=False): # TODO: consider if we need other args here e.g. title, subtitle here vs taking it from chart1
        """Create a combo chart with two charts side by side and optionally a shared legend.

        TODO: tidy up this docstring and ideally make other docstrings throughout this script consistent. pick a base style that is best practice (e.g. Google docstrings) but balances being informative with being concise.
        # I'm particularly keen to see how we resolve the common_args which are defined above but which need to be evident when one looks at the docstring for e.g. bar() combo()
        
        Arguments:
        chart1 - a lazychart chart call e.g. bar(data=... x=..., y=..., groupby=...)
        chart2 (optional) - a lazychart chart call e.g. bar(y=... groupby=...)
          - note if sticky=True, arguments from chart1 will apply immediately to chart2

        sticky (default: False)
        """

        # TODO - set this up e.g. pass two calls to _bar for chart1 and chart2, handle legend being passed through vs keeping it here and using axes
        fig, axes = self._make_figure(chart1, chart2, legend)
        return fig # TODO: this was original return fig. think about decorators and how return values will propogate up the decorator chain