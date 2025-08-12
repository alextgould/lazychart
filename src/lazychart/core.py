from typing import TypedDict, Optional, Union, Iterable, Dict, Any, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from .colors import DEFAULT_PALETTE, COLORBLIND_PALETTE
from functools import wraps
import collections.abc

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

    LEGEND_PRESETS = {
        "right":   {"loc": "center left", "bbox_to_anchor": (1.0, 0.5)},
        "bottom":  {"loc": "upper center", "bbox_to_anchor": (0.5, -0)},
        #"overlap": {"loc": "best", "bbox_to_anchor": None},
    }

    # def _auto_adjust_for_legend(fig, ax, legend, padding=0.02):
    #     """Expand figure size to accomodate legends placed right/bottom (outside the axes).
    #     matplotlib's tight_layout() and plt.subplots(constrained_layout=True) options seem to struggle with this.
    #     """

    #     logger.debug(f'attempt to _auto_adjust_for_legend {legend}')
    #     if legend == 'bottom': # horizontal list
    #         ncol = len(ax.get_legend_handles_labels()[1])
    #     else:
    #         ncol = 1 # vertical list

    #     width, height = fig.get_size_inches()
    #     ax_pos = ax.get_position().bounds  # returns (x0, y0, width, height) in fig coords
    #     logger.debug("Before initial legend: | "
    #         f"Figure size (in): [{width:.3f}, {height:.3f}] | "
    #         f"Axes pos: ({ax_pos[0]:.3f}, {ax_pos[1]:.3f}, {ax_pos[2]:.3f}, {ax_pos[3]:.3f}) | "
    #     )

    #     # add legend at random location
    #     leg = ax.legend(loc='lower left', ncol=ncol)

    #     # compute layout
    #     fig.canvas.draw()

    #     # get legend bbox in figure coordinates
    #     width, height = fig.get_size_inches()
    #     ax_pos = ax.get_position().bounds  # returns (x0, y0, width, height) in fig coords
    #     bbox = leg.get_window_extent().transformed(fig.transFigure.inverted())
    #     logger.debug("After initial legend: | "
    #         f"Figure size (in): [{width:.3f}, {height:.3f}] | "
    #         f"Axes pos: ({ax_pos[0]:.3f}, {ax_pos[1]:.3f}, {ax_pos[2]:.3f}, {ax_pos[3]:.3f}) | "
    #         f"Legend bbox: ({bbox.x0:.3f}, {bbox.y0:.3f}, {bbox.x1:.3f}, {bbox.y1:.3f}) | "
    #     )

    #     # remove legend and add another with updated bbox_to_anchor value
    #     leg.remove()
    #     if legend == 'bottom':
    #         leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0 - (bbox.height + padding)), ncol=ncol)
    #     elif legend == 'right':
    #         leg = ax.legend(loc='center left', bbox_to_anchor=(1 + (bbox.width + padding), 0.5), ncol=ncol)

    #     width, height = fig.get_size_inches()
    #     ax_pos = ax.get_position().bounds  # returns (x0, y0, width, height) in fig coords
    #     bbox = leg.get_window_extent().transformed(fig.transFigure.inverted())
    #     logger.debug("After revised legend: | "
    #         f"Figure size (in): [{width:.3f}, {height:.3f}] | "
    #         f"Axes pos: ({ax_pos[0]:.3f}, {ax_pos[1]:.3f}, {ax_pos[2]:.3f}, {ax_pos[3]:.3f}) | "
    #         f"Legend bbox: ({bbox.x0:.3f}, {bbox.y0:.3f}, {bbox.x1:.3f}, {bbox.y1:.3f}) | "
    #     )
        
    #     # compute layout - is this necessary?
    #     fig.canvas.draw()
        
    #     width, height = fig.get_size_inches()
        
    #     # If legend extends past right side
    #     if bbox.x1 > 1:
    #         extra_width = bbox.x1 - 1
    #         fig.set_size_inches(width * (1 + extra_width), height)

    #     # If legend extends below the bottom
    #     if bbox.y0 < 0:
    #         extra_height = abs(bbox.y0)
    #         fig.set_size_inches(width, height * (1 + extra_height))

    #         pos = ax.get_position()
    #         new_pos = [pos.x0, pos.y0 + extra_height, pos.width, pos.height - extra_height]
    #         ax.set_position(new_pos)

    #         logger.debug(f"adjusting size due to extra_height {extra_height} on bottom")

    #     # If legend extends above the top (e.g., large right-side legend)
    #     if bbox.y1 > 1:
    #         extra_height = bbox.y1 - 1
    #         fig.set_size_inches(width, height * (1 + extra_height))

    #     ax_pos = ax.get_position().bounds  # returns (x0, y0, width, height) in fig coords
    #     logger.debug("After resizing: | "
    #         f"Figure size (in): [{width:.3f}, {height:.3f}] | "
    #         f"Axes pos: ({ax_pos[0]:.3f}, {ax_pos[1]:.3f}, {ax_pos[2]:.3f}, {ax_pos[3]:.3f}) | "
    #         f"Legend bbox: ({bbox.x0:.3f}, {bbox.y0:.3f}, {bbox.x1:.3f}, {bbox.y1:.3f}) | "
    #     )

        
        # leg = ax.legend(loc=legend_pos, bbox_to_anchor=preset["bbox_to_anchor"], ncol=ncol
        #         )
        
        
        # width, height = fig.get_size_inches()
        # ax_pos = ax.get_position().bounds  # returns (x0, y0, width, height) in fig coords
        # bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())


        # if legend_pos == 'bottom':

        #     # increase height by the legend's bbox height plus a padding fraction
        #     extra_height = (bbox.y1 - bbox.y0) * (1 + padding)
        #     fig.set_size_inches(width, height * (1 + extra_height))

        #     # Move axes up by extra_height fraction and shrink height accordingly
        #     pos = ax.get_position()
        #     new_y0 = pos.y0 + extra_height
        #     new_height = pos.height - extra_height
        #     ax.set_position([pos.x0, new_y0, pos.width, new_height])
        #     logger.debug(f"Adjusted for bottom legend: extra_height={extra_height}")

        # elif legend_pos == 'right':
            
        #     # increase width by legend's bbox width plus padding
        #     extra_width = (bbox.x1 - bbox.x0) * (1 + padding)
        #     fig.set_size_inches(width * (1 + extra_width), height)

        #     # Optionally move/shrink axes horizontally if needed
        #     pos = ax.get_position()
        #     new_width = pos.width - extra_width
        #     ax.set_position([pos.x0, pos.y0, new_width, pos.height])
        #     logger.debug(f"Adjusted for right legend: extra_width={extra_width}")

        # resized = False

        # # Legend extends past right side
        # if bbox.x1 > 1:
        #     extra_width = bbox.x1 - 1
        #     fig.set_size_inches(width * (1 + extra_width), height)
        #     resized = True
        #     logger.debug(f"adjusting size due to extra_width {extra_width} on right side")

        # # Legend extends below bottom
        # if bbox.y0 < 0:
        #     extra_height = abs(bbox.y0)
        #     fig.set_size_inches(width, height * (1 + extra_height))
        #     logger.debug(f"adjusting size due to extra_height {extra_height} on bottom")
        #     # Shift axes upward by this fraction to avoid overlap
        #     pos = ax.get_position()
        #     new_y0 = pos.y0 + extra_height
        #     new_height = pos.height - extra_height
        #     if new_height <= 0:
        #         new_height = pos.height * 0.8  # fallback: shrink a bit
        #     logger.debug(f"about to set using ax.set_position ax.get_position().bounds is {ax.get_position().bounds}")
        #     ax.set_position([pos.x0, new_y0, pos.width, new_height])
        #     logger.debug(f"finished setting using ax.set_position ax.get_position().bounds is {ax.get_position().bounds}")

        #     resized = True

        # # Legend extends above top (rare case)
        # if bbox.y1 > 1:
        #     extra_height = bbox.y1 - 1
        #     fig.set_size_inches(width, height * (1 + extra_height))
        #     resized = True

        # #if resized:
        #     #fig.tight_layout()

        # width, height = fig.get_size_inches()
        # ax_pos = ax.get_position().bounds
        # bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
        # logger.debug("After making changes: | "
        #     f"Figure size (in): [{width:.3f}, {height:.3f}] | "
        #     f"Axes pos: ({ax_pos[0]:.3f}, {ax_pos[1]:.3f}, {ax_pos[2]:.3f}, {ax_pos[3]:.3f}) | "
        #     f"Legend bbox: ({bbox.x0:.3f}, {bbox.y0:.3f}, {bbox.x1:.3f}, {bbox.y1:.3f}) | "
        # )

        #logger.debug(f"After: fig size=({width:.3f},{height:.3f}), ax_pos={ax_pos}, legend bbox=({bbox.x0:.3f}, {bbox.y0:.3f}, {bbox.x1:.3f}, {bbox.y1:.3f})")

    # def _auto_adjust_for_legend(fig, ax, legend, pad_fraction=0):
    #     """
    #     Expands figure size if legend is outside the plot area.
    #     Uses tight_layout after resizing to avoid squished axes.
    #     """

    #     fig.canvas.draw()  # ensure layout is computed
    #     width, height = fig.get_size_inches()
    #     ax_pos = ax.get_position().bounds  # returns (x0, y0, width, height) in fig coords
    #     bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
    #     logger.debug("Before making changes: | "
    #         f"Figure size (in): [{width:.3f}, {height:.3f}] | "
    #         f"Axes pos: ({ax_pos[0]:.3f}, {ax_pos[1]:.3f}, {ax_pos[2]:.3f}, {ax_pos[3]:.3f}) | "
    #         f"Legend bbox: ({bbox.x0:.3f}, {bbox.y0:.3f}, {bbox.x1:.3f}, {bbox.y1:.3f}) | "
    #     )

    #     # If legend extends past right side
    #     if bbox.x1 > 1:
    #         extra_width = bbox.x1 - 1
    #         fig.set_size_inches(width * (1 + extra_width), height)

    #     # If legend extends below the bottom
    #     if bbox.y0 < 0:
    #         extra_height = abs(bbox.y0)
    #         fig.set_size_inches(width, height * (1 + extra_height))

    #         pos = ax.get_position()
    #         new_pos = [pos.x0, pos.y0 + extra_height, pos.width, pos.height - extra_height]
    #         ax.set_position(new_pos)

    #     # If legend extends above the top (e.g., large right-side legend)
    #     if bbox.y1 > 1:
    #         extra_height = bbox.y1 - 1
    #         fig.set_size_inches(width, height * (1 + extra_height))

    #     width, height = fig.get_size_inches()
    #     ax_pos = ax.get_position().bounds  # returns (x0, y0, width, height) in fig coords
    #     bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
    #     logger.debug("After making changes: | "
    #         f"Figure size (in): [{width:.3f}, {height:.3f}] | "
    #         f"Axes pos: ({ax_pos[0]:.3f}, {ax_pos[1]:.3f}, {ax_pos[2]:.3f}, {ax_pos[3]:.3f}) | "
    #         f"Legend bbox: ({bbox.x0:.3f}, {bbox.y0:.3f}, {bbox.x1:.3f}, {bbox.y1:.3f}) | "
    #     )

        # fig.canvas.draw()  # ensure layout is computed
        # bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
        # ax_pos = ax.get_position().bounds  # returns (x0, y0, width, height) in fig coords

        # width, height = fig.get_size_inches()
        # extra_width = max(0, bbox.x1 - 1)
        # extra_height_bottom = max(0, -bbox.y0)
        # extra_height_top = max(0, bbox.y1 - 1)
        # logger.debug("Before making changes: | "
        #     f"Figure size (in): [{width:.3f}, {height:.3f}] | "
        #     f"Axes pos: ({ax_pos[0]:.3f}, {ax_pos[1]:.3f}, {ax_pos[2]:.3f}, {ax_pos[3]:.3f}) | "
        #     f"Legend bbox: ({bbox.x0:.3f}, {bbox.y0:.3f}, {bbox.x1:.3f}, {bbox.y1:.3f}) | "
        # )

        # # Amount the legend sticks out beyond figure bounds
        # extra_left = max(0, -bbox.x0)
        # extra_right = max(0, bbox.x1 - 1)
        # extra_bottom = max(0, -bbox.y0)
        # extra_top = max(0, bbox.y1 - 1)

        # logger.debug(
        #     f"Extra L/R: {extra_left:.3f}/{extra_right:.3f}, Extra T/B: {extra_top:.3f}/{extra_bottom:.3f}"
        # )

        # # Additional padding
        # extra_left += pad_fraction
        # extra_right += pad_fraction
        # extra_bottom += pad_fraction
        # extra_top += pad_fraction

        # # Expand figure size
        # new_width = width * (1 + extra_left + extra_right)
        # new_height = height * (1 + extra_top + extra_bottom)

        # if new_width > width or new_height > height:
        #     fig.set_size_inches(new_width, new_height)
        #     #fig.tight_layout()  # let Matplotlib handle repositioning

        # logger.debug("After making changes: | "
        #     f"Figure size (in): [{width:.3f}, {height:.3f}] | "
        #     f"Axes pos: ({ax_pos[0]:.3f}, {ax_pos[1]:.3f}, {ax_pos[2]:.3f}, {ax_pos[3]:.3f}) | "
        #     f"Legend bbox: ({bbox.x0:.3f}, {bbox.y0:.3f}, {bbox.x1:.3f}, {bbox.y1:.3f}) | "
        # )

        # if new_width > width or new_height > height:
        #     #fig.set_size_inches(new_width, new_height)
        #     fig.tight_layout()  # let Matplotlib handle repositioning

        # logger.debug("After making changes and after tight_layout(): | "
        #     f"Figure size (in): [{width:.3f}, {height:.3f}] | "
        #     f"Axes pos: ({ax_pos[0]:.3f}, {ax_pos[1]:.3f}, {ax_pos[2]:.3f}, {ax_pos[3]:.3f}) | "
        #     f"Legend bbox: ({bbox.x0:.3f}, {bbox.y0:.3f}, {bbox.x1:.3f}, {bbox.y1:.3f}) | "
        # )

    # def _auto_adjust_for_legend(fig, ax, legend):
    #     """Expand figure size if legend is outside the visible area."""
    #     fig.canvas.draw()  # ensure layout is computed
    #     bbox = legend.get_window_extent()
    #     inv = fig.transFigure.inverted()
    #     legend_bbox = bbox.transformed(inv)

    #     width, height = fig.get_size_inches()

    #     # If legend extends past right side
    #     if legend_bbox.x1 > 1:
    #         extra_width = legend_bbox.x1 - 1
    #         fig.set_size_inches(width * (1 + extra_width), height)

    #     # If legend extends below the bottom
    #     if legend_bbox.y0 < 0:
    #         extra_height = abs(legend_bbox.y0)
    #         fig.set_size_inches(width, height * (1 + extra_height))

    #     # If legend extends above the top (e.g., large right-side legend)
    #     if legend_bbox.y1 > 1:
    #         extra_height = legend_bbox.y1 - 1
    #         fig.set_size_inches(width, height * (1 + extra_height))

    #     # Use logging to see what is going on
    #     ax_pos = ax.get_position().bounds  # returns (x0, y0, width, height) in fig coords
    #     extra_width = max(0, bbox.x1 - 1)
    #     extra_height_bottom = max(0, -bbox.y0)
    #     extra_height_top = max(0, bbox.y1 - 1)
    #     logger.debug(
    #         f"Figure size (in): [{width:.3f}, {height:.3f}] | "
    #         f"Axes pos: ({ax_pos[0]:.3f}, {ax_pos[1]:.3f}, {ax_pos[2]:.3f}, {ax_pos[3]:.3f}) | "
    #         f"Legend bbox: ({bbox.x0:.3f}, {bbox.y0:.3f}, {bbox.x1:.3f}, {bbox.y1:.3f}) | "
    #         f"Extra W: {extra_width:.3f}, Extra H bottom: {extra_height_bottom:.3f}, Extra H top: {extra_height_top:.3f}"
    #     )

    def _auto_adjust_for_legend(fig, ax, legend_pos, padding=0.02):
        """
        Adjust figure size and axes position to accommodate legends placed
        outside the axes on the right or bottom.

        Parameters:
            fig: matplotlib.figure.Figure
            ax: matplotlib.axes.Axes
            legend_pos: str, either 'bottom' or 'right'
            padding: float, extra padding in figure fraction units
        
        This function places a temporary legend, measures its bbox,
        then resizes the figure and repositions the axes to make room.
        """

        # Determine number of columns for legend
        handles, labels = ax.get_legend_handles_labels()
        ncol = len(labels) if legend_pos == 'bottom' else 1

        # Step 1: Add a temporary legend at initial "outside" position
        if legend_pos == 'bottom':
            leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=ncol)
        elif legend_pos == 'right':
            leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=ncol)
        else:
            raise ValueError("legend_pos must be 'bottom' or 'right'")

        fig.canvas.draw()  # Needed to compute bbox

        # Step 2: Get legend bounding box in figure coords
        bbox = leg.get_window_extent().transformed(fig.transFigure.inverted())
        width, height = fig.get_size_inches()
        pos = ax.get_position()

        # Remove the temporary legend before adding adjusted one
        leg.remove()

        # Step 3: Calculate needed expansion and adjust figure & axes

        if legend_pos == 'bottom':
            # Legend extends below the axes, check if y0 < 0
            extra_height = max(0, -bbox.y0 + padding)

            if extra_height > 0:
                # Increase figure height
                new_height = height * (1 + extra_height)
                fig.set_size_inches(width, new_height)

                # Shift axes up and reduce height accordingly
                new_pos = [pos.x0, pos.y0 + extra_height, pos.width, pos.height - extra_height]
                ax.set_position(new_pos)

                # Adjust subplot bottom margin to not cut off axes labels
                fig.subplots_adjust(bottom=pos.y0 + extra_height + padding)

            # Re-add legend with adjusted bbox_to_anchor to sit below axes
            leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0 - extra_height), ncol=ncol)

        elif legend_pos == 'right':
            # Legend extends beyond right, check if x1 > 1
            extra_width = max(0, bbox.x1 - 1 + padding)

            if extra_width > 0:
                # Increase figure width
                new_width = width * (1 + extra_width)
                fig.set_size_inches(new_width, height)

                # Shrink axes width to make room for legend
                new_pos = [pos.x0, pos.y0, pos.width - extra_width, pos.height]
                ax.set_position(new_pos)

                # Adjust subplot right margin to prevent clipping
                fig.subplots_adjust(right=pos.x0 + pos.width - extra_width - padding)

            # Re-add legend with adjusted bbox_to_anchor to sit right of axes
            leg = ax.legend(loc='center left', bbox_to_anchor=(1 + extra_width, 0.5), ncol=ncol)

        fig.canvas.draw()  # Redraw with new layout

        # Debug info (optional)
        # bbox = leg.get_window_extent().transformed(fig.transFigure.inverted())
        # ax_pos = ax.get_position().bounds
        # print(f"Figure size: {fig.get_size_inches()}, Axes pos: {ax_pos}, Legend bbox: {bbox}")
    
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

        # Legend handling
        if kwargs.get('legend') is not None:
            preset = LEGEND_PRESETS[kwargs.get('legend', None)] 
            if preset:
                # if kwargs['legend'] == 'bottom':
                #     ncol = len(ax.get_legend_handles_labels()[1])
                # else:
                #     ncol = 1
                # leg = ax.legend(
                #     loc=preset["loc"],
                #     bbox_to_anchor=preset["bbox_to_anchor"],
                #     ncol=ncol
                # )
                #_auto_adjust_for_legend(fig, ax, leg, kwargs['legend'])
                _auto_adjust_for_legend(fig, ax, kwargs.get('legend')) #leg, kwargs['legend'])
        
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

        fig, ax = plt.subplots() #constrained_layout=True)

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

            grouped.plot(kind="bar", x=x, y=(y if y else "count"), ax=ax)

        return fig, ax