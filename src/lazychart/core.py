from typing import TypedDict, Optional, Union, Iterable, Dict, Any, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from .colors import DEFAULT_PALETTE, COLORBLIND_PALETTE
from functools import wraps
import collections.abc

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
            plt.tight_layout()
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
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None

    # label font size
    title_size: Optional[Union[str, int]] = None
    subtitle_size: Optional[Union[str, int]] = None
    xlabel_size: Optional[Union[str, int]] = None
    ylabel_size: Optional[Union[str, int]] = None
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
    show_legend: bool = True
    legend_loc: str = "best" # TODO: presets e.g. right, bottom 

    # other
    show_grid: bool = True

def common_args(func: Callable) -> Callable:
    """
    Decorator to handle common charting arguments.

    Parameters (optional):

        - group up small categories
          - group_by (str): column name of categorical variable
          - group_threshold: number (int) or proportion (float, default: 1.0) of categories to retain
          - group_other_name (str, default: 'Other'): name of new category containing smaller categories
           
        - labels
          - title, subtitle, xlabel, ylabel (str): Various labels
          - title_size, subtitle, xlabel, ylabel, tick_size: Font size of labels (int or str preset)
            presets: 'small', 'medium', 'large', 'x-large'
          
        - axes
            - x_min, x_max, y_min, y_max (float): axis limits
            - x_axis_format, y_axis_format (str, optional): None (default), 'percent', 'comma', 'short'
            - decimals (int, default: 0): number of decimal places to display on axes
            - xtick_rotation, ytick_rotation (int): degrees rotation of tick labels

        - other
          - show_legend (bool, default: True): whether to show the legend
          - legend_loc (str, default: best): legend position (TODO presets)
          - show_grid (bool, True): whether to show gridlines
    """

    FONT_PRESETS = {
        "small":    {"title": 12, "subtitle": 11, "xlabel": 10, "ylabel": 10, "tick": 9},
        "medium":   {"title": 16, "subtitle": 14, "xlabel": 12, "ylabel": 12, "tick": 10},
        "large":    {"title": 20, "subtitle": 18, "xlabel": 16, "ylabel": 16, "tick": 12},
        "x-large":  {"title": 24, "subtitle": 20, "xlabel": 18, "ylabel": 18, "tick": 14},
    }
    
    @wraps(func)
    def wrapper(self, *args, **kwargs):

        ## Create the plot

        # combine smaller categories
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
            data[group_by] = data[group_by].replace(small_cats, group_other_name if group_other_name else "Other")

        # call the charting function
        fig, ax = func(self, *args, **kwargs)

        ## Customise appearance

        # labels
        title = kwargs.get('title')
        if title is not None:
            ax.set_title(title)
        subtitle = kwargs.get('subtitle')
        if subtitle is not None:
            ax.set_title(title)
        xlabel = kwargs.get('xlabel')
        if (xlabel is not None) and hasattr(ax, "xlabel"):
            ax.set_xlabel(xlabel)
        ylabel = kwargs.get('ylabel')
        if (ylabel is not None) and hasattr(ax, "ylabel"):
            ax.set_ylabel(ylabel)
        
        # label font size
        def resolve_fontsize(val, category):
            if isinstance(val, int):
                return val
            if isinstance(val, str):
                preset = FONT_PRESETS.get(val.lower())
                if preset:
                    return preset[category]
            return FONT_PRESETS.get('medium')[category] # fall back to medium
        title_size = kwargs.get('title_size')
        title_size = resolve_fontsize(title_size, "title")
        if title is not None and hasattr(ax, "set_title"):
            ax.set_title(title, fontsize=title_size)
        subtitle_size = kwargs.get('subtitle_size')
        subtitle_size = resolve_fontsize(subtitle_size, "subtitle")
        if subtitle is not None and hasattr(ax, "set_subtitle"):
            ax.set_subtitle(title, fontsize=title_size)
        xlabel_size = kwargs.get('xlabel_size')
        xlabel_size = resolve_fontsize(xlabel_size, "xlabel")
        if xlabel is not None and hasattr(ax, "set_xlabel"):
            ax.set_xlabel(xlabel, fontsize=xlabel_size)
        ylabel_size = kwargs.get('ylabel_size')
        ylabel_size = resolve_fontsize(ylabel_size, "ylabel")
        if ylabel is not None and hasattr(ax, "set_ylabel"):
            ax.set_ylabel(ylabel, fontsize=ylabel_size)
        tick_size = kwargs.get('tick_size')
        tick_size = resolve_fontsize(tick_size, "tick")
        if tick_size is not None:
            ax.tick_params(axis='both', labelsize=tick_size)

        # axis ranges
        x_min = kwargs.get('x_min')
        x_max = kwargs.get('x_max')
        if (x_min is not None or x_max is not None) and hasattr(ax, "set_xlim"):
            ax.set_xlim(x_min, x_max)
        y_min = kwargs.get('y_min')
        y_max = kwargs.get('y_max')
        if (y_min is not None or y_max is not None) and hasattr(ax, "set_ylim"):
            ax.set_ylim(y_min, y_max)

        # axis formats
        def format_axis(ax, which: str = 'y', fmt: Optional[str] = None, decimals: int = 0):
            """Format a given axis according to the specified style."""
            if fmt is None:
                return

            axis = ax.xaxis if which == 'x' else ax.yaxis

            if fmt == 'percent':
                axis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=decimals))
            elif fmt == 'comma':
                axis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):,}"))
            elif fmt == 'short':
                def short(x, pos):
                    x = float(x)
                    if abs(x) >= 1_000_000:
                        return f"{x/1_000_000:.{decimals}f}M"
                    if abs(x) >= 1_000:
                        return f"{x/1_000:.{decimals}f}k"
                    return f"{x:.{decimals}f}"
                axis.set_major_formatter(mticker.FuncFormatter(short))

        x_axis_format = kwargs.get('x_axis_format')
        decimals = kwargs.get('decimals')
        if x_axis_format:
            format_axis(ax, which='x', fmt=x_axis_format, decimals=decimals)
        y_axis_format = kwargs.get('y_axis_format')
        if y_axis_format:
            format_axis(ax, which='y', fmt=y_axis_format, decimals=decimals)
        if hasattr(ax, "set_rotation"):
            xtick_rotation = kwargs.get('xtick_rotation')
            if xtick_rotation is not None:
                for label in ax.get_xticklabels():
                    label.set_rotation(xtick_rotation)
            ytick_rotation = kwargs.get('ytick_rotation')
            if ytick_rotation is not None:
                for label in ax.get_yticklabels():
                    label.set_rotation(ytick_rotation)

        # legend
        
        show_legend = kwargs.get('show_legend')
        if show_legend and hasattr(ax, "legend"):
            legend_loc = kwargs.get('legend_loc')
            ax.legend(loc=legend_loc if legend_loc else "best")

        # grid
        
        show_grid = kwargs.get('show_grid')
        if show_grid and hasattr(ax, "grid"):
            ax.grid(True, linestyle='--', alpha=0.4)

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

        return grouped

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
        sort_x_by: str = "value",
        sort_group_by: str = "label",
        sort_x_desc: bool = True,
        sort_group_desc: bool = True,
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
            sort_x_by (str, default 'value'): How to sort x-axis categories.
                One of: 'label', 'value', 'none'.
            sort_group_by (str, default 'label'): How to sort group_by categories.
                One of: 'label', 'value', 'none'.
            sort_x_desc (bool, default True): Whether to sort x-axis categories in descending order.
            sort_group_desc (bool, default True): Whether to sort group_by categories in descending order.
            **kwargs: Additional common_args customisation (title, label, axis limit, etc.)

        Returns:
            (fig, ax): Matplotlib Figure and Axes objects (unless show_fig=True, then returns None).
        """

        fig, ax = plt.subplots()

        if stacking not in {"none", "standard", "proportion"}:
            raise ValueError("stacking must be one of: 'none', 'standard', 'proportion'")
        if sort_x_by not in {"none", "label", "value"}:
            raise ValueError("sort_x_by must be one of: 'none', 'label', 'value'")
        if sort_group_by not in {"none", "label", "value"}:
            raise ValueError("sort_group_by must be one of: 'none', 'label', 'value'")

        # Aggregate data

        grouped = self._aggregate_data(data, x, y, group_by, aggfunc)

        # Sorting
        if group_by:
            # Sort columns (group_by categories)
            if sort_group_by == "label":
                grouped = grouped[sorted(grouped.columns, reverse=sort_group_desc)]
            elif sort_group_by == "value":
                col_sums = grouped.sum(axis=0)
                sorted_cols = col_sums.sort_values(ascending=not sort_group_desc).index.tolist()
                grouped = grouped[sorted_cols]

            # Sort index (x categories)
            if sort_x_by == "label":
                grouped = grouped.sort_index(ascending=not sort_x_desc)
            elif sort_x_by == "value":
                idx_sums = grouped.sum(axis=1)
                grouped = grouped.loc[idx_sums.sort_values(ascending=not sort_x_desc).index]

            # 100% stacked proportions
            if stacking == "proportion":
                grouped = grouped.div(grouped.sum(axis=1), axis=0)

            grouped.plot(kind="bar", stacked=stacking != "none", ax=ax)

        else:
            if sort_x_by == "label":
                grouped = grouped.sort_values(by=x, ascending=not sort_x_desc)
            elif sort_x_by == "value":
                grouped = grouped.sort_values(by=y if y else "count", ascending=not sort_x_desc)

            grouped.plot(kind="bar", x=x, y=(y if y else "count"), ax=ax)

        return fig, ax