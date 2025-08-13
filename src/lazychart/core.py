from typing import TypedDict, Optional, Union, Iterable, Dict, Any, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from .colors import DEFAULT_PALETTE, COLORBLIND_PALETTE
from functools import wraps
import collections.abc
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

    FONT_PRESETS = {
        "small":    {"title": 12, "subtitle": 11, "x_label": 10, "y_label": 10, "tick": 9},
        "medium":   {"title": 16, "subtitle": 14, "x_label": 12, "y_label": 12, "tick": 10},
        "large":    {"title": 20, "subtitle": 18, "x_label": 16, "y_label": 16, "tick": 12},
        "x-large":  {"title": 24, "subtitle": 20, "x_label": 18, "y_label": 18, "tick": 14},
    }

    def legend_handle_and_padding_width(fig):
        """Derive figures to help split legend between text and non-text components for wrapping purposes"""
        fontsize = FontProperties(size=rcParams['legend.fontsize']).get_size_in_points()
        dpi = fig.dpi
        # Convert "em" units to points → pixels
        em_to_px = fontsize * dpi / 72.0

        handle_px = rcParams['legend.handlelength'] * em_to_px
        pad_px = rcParams['legend.handletextpad'] * em_to_px
        return handle_px + pad_px

    def legend_ncol(fig, ax, max_ratio=0.9):
        """Rough calculation of number of categories to display on each line of a legend placed at the bottom."""

        fontsize = FontProperties(size=rcParams['legend.fontsize']).get_size_in_points()
        dpi = fig.dpi
        char_px = fontsize * dpi / 72.0 * 0.6  # 0.6 is approx width/height ratio of a character

        _, labels = ax.get_legend_handles_labels()
        handle_pad_px = legend_handle_and_padding_width(fig)
        
        text_widths = [len(lbl) * char_px + handle_pad_px for lbl in labels]
        avg_width = sum(text_widths) / len(text_widths)

        fig_width_px = fig.get_size_inches()[0] * fig.dpi
        max_width_px = fig_width_px * max_ratio

        ideal_ncol = int(max_width_px // avg_width) # maximum columns that could fit on one line
        rows = ceil(len(labels) / ideal_ncol) # minimum rows required to acocmodate all labels
        ncol = ceil(len(labels) / rows) # balance labels evenly between rows
        ncol = max(1, min(len(labels), ncol)) # ensure >0 and <= number of labels
        return ncol
    
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
        def resolve_fontsize(val, category):
            if isinstance(val, int):
                return val
            if isinstance(val, str):
                preset = FONT_PRESETS.get(val.lower())
                if preset:
                    return preset[category]
            return FONT_PRESETS["medium"][category]

        if (title := kwargs.get('title')):
            ax.set_title(title, fontsize=resolve_fontsize(kwargs.get('title_size'), "title"))
        if (subtitle := kwargs.get('subtitle')) and hasattr(ax, "set_subtitle"):
            ax.set_subtitle(subtitle, fontsize=resolve_fontsize(kwargs.get('subtitle_size'), "subtitle"))
        if (x_label := kwargs.get('x_label')):
            ax.set_xlabel(x_label, fontsize=resolve_fontsize(kwargs.get('x_label_size'), "x_label"))
        if not (y_label := kwargs.get('y_label')):
            y_label = kwargs.get('y') if kwargs.get('y') else "count"
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
                ncol = legend_ncol(fig, ax)
                fig.legend(loc='outside lower center', ncol=ncol, title=kwargs.get('group_by'))
            elif kwargs.get('legend', None) == 'right':
                handles, labels = ax.get_legend_handles_labels()
                wrapped_labels = legend_wrap(fig, labels)
                fig.legend(handles, wrapped_labels, loc='outside right upper', title=kwargs.get('group_by'))
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
        aggfunc: Union[str, Callable] = "sum"
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

        Returns:
            pd.DataFrame:
                - If group_by is provided: pivot table with x as index, group_by categories as columns.
                - If group_by is None: aggregated dataframe with x and aggregated y.
        """

        #logger.debug(f"Aggregating data with x='{x}', y='{y}', group_by='{group_by}', aggfunc={aggfunc}")
        #logger.debug(f"Data shape before aggregation: {data.shape}")
        
        # Defensive copy to avoid modifying caller's dataframe
        df = data.copy()

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
    
    def _intuitive_sorting(self, data, x, group_by, sort_x, sort_group_by, sort_x_ascending, sort_group_by_ascending):
        """
        Decide default sorting columns for plotting if none are given, using data types. Intuitively
         - numeric variables are sorted in ascending order by label (e.g. successive finanical years)
         - categorical variables are sorted in descending order by values (e.g. count of records by status)

        Returns:
            sort_x (str), sort_group_by (str), sort_x_ascending (bool), sort_group_by_ascending (bool)
        """

        #logger.debug(f"Sorting decision for x='{x}' group_by='{group_by}'")
        #logger.debug(f"Initial sort_x={sort_x}, sort_x_ascending={sort_x_ascending}, "f"sort_group_by={sort_group_by}, sort_group_by_ascending={sort_group_by_ascending}")
        
        numeric_x = (pd.api.types.is_numeric_dtype(data[x]) or pd.api.types.is_datetime64_any_dtype(data[x]))
        if not sort_x:
            sort_x = 'label' if numeric_x else 'values'
        if not sort_x_ascending:
            sort_x_ascending = True if numeric_x else False
            
        numeric_group_by = (pd.api.types.is_numeric_dtype(data[group_by]) or pd.api.types.is_datetime64_any_dtype(data[group_by]))
        if not sort_group_by:
            sort_group_by = 'label' if numeric_group_by else 'values'
        if not sort_group_by_ascending:
            sort_group_by_ascending = True if numeric_group_by else False

        #logger.debug(f"Final sort_x={sort_x}, sort_x_ascending={sort_x_ascending}, "f"sort_group_by={sort_group_by}, sort_group_by_ascending={sort_group_by_ascending}")

        return sort_x, sort_group_by, sort_x_ascending, sort_group_by_ascending

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

        # Aggregate data
        grouped = self._aggregate_data(data, x, y, group_by, aggfunc)

        # Determine sorting
        sort_x, sort_group_by, sort_x_ascending, sort_group_by_ascending = self._intuitive_sorting(data, x, group_by, sort_x, sort_group_by, sort_x_ascending, sort_group_by_ascending)

        if group_by:
            # Sort columns (group_by categories)
            if sort_group_by == "label":
                grouped = grouped[sorted(grouped.columns, reverse=not sort_group_by_ascending)]
            elif sort_group_by == "value":
                col_sums = grouped.sum(axis=0)
                sorted_cols = col_sums.sort_values(ascending=sort_group_by_ascending).index.tolist()
                grouped = grouped[sorted_cols]

            # Sort index (x categories)
            if sort_x == "label":
                grouped = grouped.sort_index(ascending=sort_x_ascending)
            elif sort_x == "value":
                idx_sums = grouped.sum(axis=1)
                logger.debug(f"Index sums used for sorting: {idx_sums.to_dict()}")
                grouped = grouped.loc[idx_sums.sort_values(ascending=sort_x_ascending).index]

            # 100% stacked proportions
            if stacking == "proportion":
                grouped = grouped.div(grouped.sum(axis=1), axis=0)

            grouped.plot(kind="bar", stacked=stacking != "none", ax=ax)

        else:
            if sort_x == "label":
                grouped = grouped.sort_values(by=x, ascending=sort_x_ascending)
            elif sort_x == "value":
                grouped = grouped.sort_values(by=y if y else "count", ascending=sort_x_ascending)

            grouped.plot(kind="bar", x=x, y=(y if y else "count"), ax=ax, legend=False) # handle legend at the common_args figure level

        return fig, ax