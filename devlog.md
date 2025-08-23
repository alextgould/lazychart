# Prompt history

## First "impossible prompt" for GPT-5

I would like to develop a Python script/class/repo/package (exact details to be discussed - it will probably start as one or more classes/scripts in a repo and will eventually become a package, but if there's anything that should be done in the early stages to make the package transition easy it may be worth doing so). My vision is to be able to rapidly generate data visualisations using Python. My aim is to make this process as clean and simple for the user as possible. For example:
* I want to have sensible defaults and/or arguments to easily control things like x/y axis limits, x/y axis labels, title, subtitle, legends, presence/absence of gridlines etc. I often find myself having to look up and add many lines of code to control these things and would prefer to just pass arguments when needed
* A key feature that I think will differentiate my package/class from other packages is that it will have the capacity (and likely default behaviour) to remember things such as x/y axis titles/formats, source datasets etc, so when you want to produce several similar images (e.g. a series of images that use different variables or different categories from a categorical variable), you don't have to keep calling all the parameters and can just call the one that is changing each time. For this reason I'm contemplating calling the class/package something like "sticky" so you might have e.g. from sticky import bar and then after calling bar(data=..., y_min=0) then next call to bar() would remember the data being used as e.g. self.data self.y_min (unless an argument such as sticky = False is passed) and apply these to subsequent charts (unless an argument such as use_sticky = False is passed)

I want to be able to generate most of the standard sorts of plots that someone using Excel for example would be able to generate - for example:
* line charts, with different lines (in Excel this would be "series") for each value of a categorical variable
* bar charts (vertical and horizontal, values and proportions (in Excel this would "100% stacked chart")), with stacking/clustering by a categorical variable (or 2 with multiple levels)
  * make it easy to create histograms as a special class of bar chart, with count of rows or % rows
* scatter/bubble chart (x/y with optional additional variables for color/size of points)
* pie chart (ideally just need to specify the category variable and then % rows is done automatically, with % value done if another numeric variable is specified)

Some additional features that are optional but could be valuable:
* ability to add trend lines (line and scatter/bar charts) - at a minimum, a linear one, but potentially some more fits, noting that I want to balance the complexity of the code base with the potential usefulness of the features being added added and prefer simple but flexible code; also being able to "forecast" by extending the trendline beyond the range of the data
* ability to add one or more horizontal/vertical dotted lines at specific values (e.g. a target x value or a y value that indicates a change in environment) (at a minimum, to line/scatter charts, but ideally also between bars in a categorical chart if e.g. monthly bars were being shown)
* where categories are used (e.g. line charts with multiple series, or bar charts with categorical x/y values), sorting data in intelligent ways by default and making it easy to change between common sorting schemes (e.g. sorting by descending value vs alphabetical x axis variable for categories)
* formatting axes values in intelligent ways by default and making it easy to change between common formats (e.g. % (e.g. values are 0.45), % but with automatic /100 (e.g. values are 45 but relate to percentages), comma delimited (e.g. values are 5000 so format as 5,000), shorter values (e.g. values are 5000000 but show as 5M or for 5000 show as 5k)), also ability to easily adjust the number of decimal places
* ability to show/hide legend and some intelligent code to place it on the right/bottom with the figure size being sensible and not having to manually adjust the figsize to get this to happen (this may be complex, but could involve e.g. aspect ratios, or figuring out the default size of the plot without the legend / the legend without the plot, and then adjusting the size to include the legend)
* ability to automatically group smaller categories with less than a given % of the data (e.g. 10% by default) into an "Other" category
* ability to adjust the "font size" of the title, axes labels and legend, possibly having some better defaults or intelliegent way to adjust this dynamically in response to what's being plotted
* ability to easily "remove" or "turn off" the title/subtitle and x/y axes labels
* scatter plots
  * having an optional parameter to add "jitter" so the density of values in the same/similar location can be more easily seen
  * having the ability to pass a categorical x axis value (may be the default behaviour, I'm not familiar with this)
* ability to more easily control colours or use colour more effectively
  * ability to define a schema using hex values (e.g. brand colours from a marketing department) and have these be incorporated (e.g. creating discrete/continuous colour maps, handling scenarios where more categories exist than the provided hex colours and so more are appended on in an extended set of colours)
  * ability to highlight specific datapoints in some relatively easy manner (e.g. having a highlight_points Bool series that matches the data and adjusts the color based on a highlight_colour parameter)
* ability to add annotations to the data in an easy manner (e.g. passing a tooltips Str series that matches the data)
* ability to pass a function that will intelligently adjust the x/y axis scales (e.g. reverse out a "z score tranformation", apply/remove logarithmic scaling)

I'm interested in hearing/incorporating any other frequently suggested data visualisation aspects (e.g. from users of Tableau, Power BI), so the list above is not exhaustive and if there's any major gaps we should try and address those.

I may also want to incorporate some more unique plots that come up in my work as a Data Scientist - for example:
* combination charts
    * bar chart and a line chart on the same plot using different y axes on the left/right side of the chart
    * possibly having function/functionality to combine a list of related data "series" within the one plot (e.g. having a "historical" series and a "forecast" series, and having these display as categories within a line or bar chart)
* line charts that use a seasonality variable and period to overlay successive periods (e.g. taking a time series that has monthly values across 3 years and showing this as 3 lines with 12 points rather than 1 line with 36 points)
* heat maps, contour plots or surface plots
  * correlation matrix - ideally with a default view that only shows the bottom left triangle and not the top triangle or main diagonal, so the scale is more informative
  * geospatial (e.g. specify lat and long inputs and another variable to show as heat)
* confusion matrix - handling simple 2x2 but also more complex such as 3x3 classes
* partial dependence plots
* distribution charts e.g.
  * box (or box and whiskers) plot (vertical lines showing min/max, boxes showing Q1 to Q3, line showing median)
    * if not already covered, ability to make this into a "candlestick" chart with a time variable on the x axis
  * density plot (shows probability distribution of a continuous variable, potentially with multiple lines for different categories of a categorical variable)
  * dumbbell plot that uses two values for each category instead of one, with two points connected by a line to show min/max values for each category
  * "age/sex pyramid" style chart where two categories are shown on the left/right side of a symmetrical x axis, comparing the distribution of each against a variable on the y axis
  * violin plots (combine features of box plot and kernel density plot)
* gantt charts (where there are categories on the y axis, a time variable on the x axis and a series of horizontal bars for each, potentially having multiple bars on one line separated by empty space)
* "funnel" chart as a special type of horizontal bar chart with the bars being centered
* waterfall chart (pass categorical variable for x axis and min/max values for the y axis with vertical bars being shown between the min/max values)

I'm not sure whether I should build upon raw matplotlib or an existing higher level package such as seaborn. My fear is that building on an existing higher level package will make it harder for me to control things and introduce dependency risk if they change the way they do things, however my fear of using raw matplotlib is that I'll end up reinventing the wheel. I think it's safe to assume data will be generally passed as pandas dataframes. It's also worth noting that I don't want to work towards interactive charts at this stage, and want to focus only on more quickly churning out static visualisations.

Finally, I want to have a good dataset that will allow me to test and demo the functionality of the tool within my repo, for example, in a Jupyter notebook. I have some existing code that generates somewhat interesting data but lacks a time series aspect. For reference, this is shown below and it may be a useful starting point for creating the dataset that will be used for demonstrating charts. I'm open to scrapping this and doing something completely different, although ideally the data will be interesting and not the typical "sales by month by region" data that is used everywhere. The data probably needs to be rich enough with both quantitative (numeric) and qualitative (categorical) columns, a temporaral (time) dimension, and potentially include ordinal (ordered) and nominal (not ordered) categories with a mixture of cardinality (many vs few categorical levels).

example of generating data to plot
import numpy as np
import pandas as pd
import os
from scipy.special import softmax

def generate_emotion_data(n=10000, seed=42):
    np.random.seed(seed)

    sleep_hours = np.clip(np.random.normal(7, 1, n), 4, 10)
    steps = np.clip(np.random.normal(6000, 1200, n), 2000, 12000)

    alcohol_prob = np.random.rand(n)
    alcohol = np.where(alcohol_prob < 0.5, 0, np.random.poisson(3, n))
    alcohol = np.clip(alcohol, 0, 8)

    social_mins = np.clip(np.random.normal(90, 30, n), 0, 240)
    work_stress = np.clip(np.random.normal(5, 2, n), 0, 10)
    nutrition_score = np.clip(np.random.normal(6.5, 1.5, n), 0, 10)

    # Feature correlations
    sleep_hours -= alcohol * 0.1
    nutrition_score += (steps - 6000) / 6000

    noise = np.random.normal(0, 1, (n, 3))

    happy_score = (
          0.4 * sleep_hours
        + 0.5 * (10 - work_stress)
        + 0.03 * social_mins
        + 0.4 * nutrition_score
        - 0.2 * alcohol
        + noise[:, 0]
    )

    energetic_score = (
          0.001 * steps
        + 0.5 * sleep_hours
        - 0.4 * alcohol
        + 0.4 * nutrition_score
        - 0.3 * work_stress
        + noise[:, 1]
    )

    engaged_score = (
        -0.3 * work_stress
        + 0.05 * social_mins
        + 0.3 * nutrition_score
        + 0.3 * sleep_hours
        + 0.0005 * steps
        + noise[:, 2]
    )

    logits = np.stack([happy_score, energetic_score, engaged_score], axis=1)
    prob = softmax(logits, axis=1)
    predicted_class = np.array(['Happy', 'Energetic', 'Engaged'])[np.argmax(prob, axis=1)]

    df = pd.DataFrame({
        'sleep_hours': sleep_hours,
        'steps': steps,
        'alcohol': alcohol,
        'social_mins': social_mins,
        'work_stress': work_stress,
        'nutrition_score': nutrition_score,
        'happy_score': happy_score,
        'energetic_score': energetic_score,
        'engaged_score': engaged_score,
        'happy_prob': prob[:, 0],
        'energetic_prob': prob[:, 1],
        'engaged_prob': prob[:, 2],
        'predicted_emotion': predicted_class
    })

    return df


Based on the above context, help me create a repo that meets as many of my specifications as possible. I'm not completely familiar with the limitations of GPT-5, but aim to give me a "one shot" output which gets me as close to my vision as possible in the most appropriate way. For example, this might mean producing a report that clearly outlines the folder structure and contents of each of the files. Alternatively, you might produce all of the files and make them available for me to download. I've been using Gemini models recently and they integrate well with Colab. I'm flexible, so long as I get as close as possible to my vision in a single attempt. I think you will need to think deeply about this task in order to do well. Good luck :)

### Observations

* impressive how much it did, but also some pretty obvious mistakes (e.g. __init__.py tried to include a variable that was being created a few lines later within __init__.py), complete lack of comments/docstrings
* failed to identify more elegant way of doing things using decorators which I came to with vscode copilot (GPT 4.1?) shortly afterwards
* scratched the surface on available charts, but this is probably good given I'll need to refactor it all anyway
* didn't really do a good job of improving the synthetic data generation, might need to do this a specific task, and more generally, try to make small iterative changes with human oversight - which is exactly how I would have approached things before listening to Sam Altman talking up GPT-5 - and a bit of a reality check when people talk about it as if it's already able to operate independently

### Sub-prompt: Synthetic data

Here's my existing function for generating synthetic data:

<TODO: paste function here>

Try to improve this so it will be useful in testing advanced plots. For example, this might be a useful starting point:

* Random user attributes

```python
age_band = np.random.choice(
    ["<20", "20–29", "30–39", "40–49", "50+"], p=[0.1, 0.25, 0.3, 0.2, 0.15]
)
personality = np.random.choice(
    ["Type A", "Type B", "Type C", "Type D", "Type E",
     "Type F", "Type G", "Type H", "Type I", "Type J"],
    p=[0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]
)
```

* Trend parameter per user (representing their "life trajectory") – a slope factor from -0.5 to +0.5 that influences their metrics over time (e.g. alcohol use or exercise levels getting better or worse over time).

* Metric generation

```python 
base_value = np.random.normal(50, 10)
time_series = base_value + slope * np.arange(n_days) + np.random.normal(0, 5, n_days)
```

You could then make alcohol use, mood, etc. each follow their own slopes per user.

I think this might give plenty of variability for charts like: grouped bars by age, trend lines, histograms by category, etc.

## Second "impossible prompt" for GPT-5: Major refactor

I have uploaded my latest python file. I want you to *carefully* refactor my code so it retains as much as possible of my original logic and comments, except where it makes sense to remove things entirely. This will be a major refactor which will implement the following structural changes:
- ChartMonkey will be stateful, with self.chart_params being used throughout in place of the current "passing around kwargs" approach
- retain the public facing kwargs interface but convert immediately into a ChartConfig class, in part to ensure sensible centralised defaults
- consolidate the layout logic, including replacing some or all of the decorators with internal helpers within the class
- cleaner class layout (arrange functions with core data logic > styling and layout > chart primitives > public api with comment lines to delineate the sections)

A few snippets of code that may be helpful in making this transition:

```example of ChartConfig class
from dataclasses import dataclass, field
from typing import Optional, Union, Dict

@dataclass
class ChartConfig:
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    legend: str = "right"
    title_size: Union[int, str] = "medium"
    tick_size: Union[int, str] = "medium"
    label_map: Dict = field(default_factory=dict)
    grid_x: bool = False
    grid_y: bool = True
    # etc ...
```

```conversion of kwargs and sticky parameters into self.chart_params
config = ChartConfig(**{**self._sticky, **kwargs})
self.chart_params = config
```

A few relevant comments from an earlier brainstorming session:

```cleaner class layout
A seasoned dev might structure the class like this:

Core data logic: _aggregate_data, _resolve_sort_order.

Styling / layout: _apply_common_style, _finalize_layout.

Chart primitives: _bar, _line, _pie.

Public API: bar(), line(), pie(), combo().

No decorators, no repeated kwargs threading — all public API methods do:

Merge kwargs + sticky → self.chart_params.

Call _foo.

Call _apply_common_style + _finalize_layout.

Return fig, ax
```

```consolidate layout logic
Instead of _finalize_layout + common_args + FONT_PRESETS sprinkled across decorators, you’d have something like _apply_common_style(ax) that applies all axis/label/font/grid defaults in one place.

_finalize_layout would then just worry about legends & titles
```

```making ChartMonkey stateful
At the start of a chart call, collect all params (kwargs + defaults + sticky) into self.chart_params.

All helpers (_aggregate_data, _finalize_layout) read from self.chart_params, not kwargs.

No need for common_args decorator at all.
```

Because of the size of this task, we will break this down. You'll first provide the "pre class" contents, which will be the new ChartConfig class with appropriate defaults and so on. You'll then ask me if you should proceed to the next step, which will be building the top section of the class - the __init__ and so on, and the first component with all the core data logic. You'll then ask me if you should proceed to the next step, which will be the layout aspects. You'll then ask me if you should proceed to the next step, which will be the chart primitives. Finally, you'll ask me if you should proceed to the next step, which will be the public facing api.

Along the way, you can try and introduce some consistency to my docstrings, with a preference for google style docstrings.

# TODO

## Automate versioning

Get more familiar with setuptools etc:

```chatgpt
Mechanics

Update pyproject.toml and tag:

git commit -am "Bump version to 0.1.0"
git tag -a v0.1.0 -m "0.1.0: post-refactor, tests passing"
git push && git push --tags


Expose __version__ (handy in logs/plots):

# src/lazychart/__init__.py
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("lazychart")
except PackageNotFoundError:
    __version__ = "0+unknown"


(Now from lazychart import __version__ works in users’ envs.)

If you want to automate (optional)

Use setuptools-scm to derive the version from Git tags so you never edit it by hand:

# pyproject.toml
[build-system]
requires = ["setuptools>=61", "wheel", "setuptools-scm>=8"]
build-backend = "setuptools.build_meta"

[project]
dynamic = ["version"]

[tool.setuptools_scm]
version_scheme = "post-release"
local_scheme = "node-and-date"


Then just tag v0.1.0; your wheels/install will carry that version (and dev builds look like 0.2.0.dev3+g<sha>).

```


### Current fix requests

#### Think added but need to check with demo_usage

* return data so we can do fig, ax, grouped = cm.bar(...) and then do display(grouped) to see the figures for ease of reference / commentary
* x axis horizontal for small labels
* no legend if only one category
* provide dict to quickly remap labels e.g. 0/1 becomes "No"/"Yes"
* sort by value doesn't work when no y axis variable, should sort by count in this case
* consider removing ValueError(f"Column '{x}' contains values that cannot be parsed as datetimes.") and just letting values flow through as missings and/or removing missing values at this point, perhaps with a logger warning
* increase default title font size
* add a little bit of padding between x axis label and tick mark labels
* show days with zero data in a x_period chart (e.g. days)
* better way to show weeks in a x_period chart
* legend title based on group_by variable by default (only on right legend?)

#### To be added

* figure out how to handle x targets with categorical x values

* add combo chart once tested, with mix on the right side

I want to add in a compare chart which can compare two charts side-by-side (and ideally allow three side-by-side if it's not overly complex). I note that the _ensure_fig_ax function has been set up to accept an existing axis for this purpose. Ideally I would like the 

* delay chart which takes two date cols and/or date col + date const and calculates the difference then plots bar chart / line chart distribution (based on kind='bar' etc) of counts/outcomes grouped by the delay



### Old code to potentially revisit with ChatGPT in the near future

```python

# -----------------------------------------------------------------------------
# Future development                                                           
# -----------------------------------------------------------------------------

    # ---- Side-by-side combo (restored semantics) -------------------------------
    @add_docstring(COMMON_DOCSTRING)
    def combo(
        self,
        chart1: Callable[[Axes], None],
        chart2: Callable[[Axes], None],
        legend: Optional[str] = "right",
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[Figure, Dict[str, Axes]]:
        """Create a side-by-side figure and draw two user-supplied callables."""
        # store style kwargs so grid/title etc. are respected when we finalise
        cfg = self._set_params(legend=legend, title=title, **kwargs)
        # Create 1x2 figure
        fig, axs = plt.subplots(1, 2, figsize=(cfg.fig_size[0]*2, cfg.fig_size[1]))
        axes = {"D1": axs[0], "D2": axs[1]}

        # Apply per-chart palette (set on both)
        if cfg.palette is not None:
            colors = _resolve_palette(cfg.palette)
            for a in axes.values():
                a.set_prop_cycle(color=list(colors))

        # Draw charts
        chart1(axes["D1"])
        chart2(axes["D2"])

        # Collect legend entries across axes
        handles, labels = [], []
        for a in axes.values():
            h, l = a.get_legend_handles_labels()
            handles += h
            labels += l

        # Apply common styling on both axes
        for a in axes.values():
            self._apply_common_style(a)

        # finalise multi-axes layout
        self._finalise_layout_multi(fig, axes, handles, labels)

        if cfg.save_path:
            fig.savefig(cfg.save_path, bbox_inches="tight", dpi=fig.dpi)
        if cfg.show_fig:
            plt.show()

        return fig, axes

    # ---- Delay chart ----------------------------------------------------------
    @add_docstring(COMMON_DOCSTRING)
    def delay_chart(
        self,
        data: pd.DataFrame,
        start: Union[str, pd.Timestamp],
        date: str,
        group_by: Optional[str] = None,
        cumulative: bool = False,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot distribution of delays (days from `start` to `date`)."""
        cfg = self._set_params(data=data, **kwargs)
        
        # Resolve start series (constant or column)
        if isinstance(start, str) and start in df.columns:
            start_series = pd.to_datetime(df[start], errors="coerce")
        else:
            start_series = pd.to_datetime(pd.Series([start] * len(df)), errors="coerce")

        date_series = pd.to_datetime(df[date], errors="coerce")
        mask = start_series.notna() & date_series.notna()
        delays = (date_series[mask] - start_series[mask]).dt.days

        plot_df = pd.DataFrame({"delay": delays})
        if group_by and group_by in df.columns:
            plot_df[group_by] = df.loc[mask, group_by].values

        # Build counts per delay (and group)
        if group_by and group_by in plot_df.columns:
            counts = plot_df.groupby([group_by, "delay"]).size().reset_index(name="value")
        else:
            counts = plot_df.groupby(["delay"]).size().reset_index(name="value")

        # Cumulative if requested
        if cumulative:
            if group_by and group_by in plot_df.columns:
                counts["value"] = counts.groupby(group_by)["value"].cumsum()
            else:
                counts["value"] = counts["value"].cumsum()

        # Draw
        fig, ax = plt.subplots(figsize=cfg.fig_size)
        if group_by and group_by in counts.columns:
            for key, sub in counts.groupby(group_by, dropna=False):
                sub = sub.sort_values("delay")
                ax.plot(sub["delay"], sub["value"], marker="o", label=str(key))
        else:
            counts = counts.sort_values("delay")
            ax.plot(counts["delay"], counts["value"], marker="o")

        self._apply_common_style(ax)
        # Some sensible axis labels
        ax.set_xlabel("Delay (days)")
        ax.set_ylabel("Count" if not cumulative else "Cumulative count")
        self._finalise_layout(fig, ax)

        if cfg.save_path:
            fig.savefig(cfg.save_path, bbox_inches="tight", dpi=fig.dpi)
        if cfg.show_fig:
            plt.show()

        return fig, ax

    # ---- Combo mix: values vs counts side-by-side (restored) -------------------
    @add_docstring(COMMON_DOCSTRING)
    def combo_mix(
        self,
        *,
        data: pd.DataFrame,
        x: str,
        y: Optional[str] = None,
        group_by: Optional[str] = None,
        aggfunc: Union[str, Callable] = "sum",
        left_title: Optional[str] = None,
        right_title: Optional[str] = None,
        legend: Optional[str] = "right",
        **kwargs: Any,
    ) -> tuple[Figure, Dict[str, Axes]]:
        """Left: aggregated `y` (or counts if None); Right: counts (sample size)."""
        # Left chart (values)
        cfg_left = self._set_params(data=data, x=x, y=y, group_by=group_by, aggfunc=aggfunc, **kwargs)
        agg_left = self._sort(self._aggregate_data())

        # Create multi-axes fig
        fig, axs = plt.subplots(1, 2, figsize=(cfg_left.fig_size[0]*2, cfg_left.fig_size[1]))
        axes = {"D1": axs[0], "D2": axs[1]}

        if cfg_left.palette is not None:
            colors = _resolve_palette(cfg_left.palette)
            for a in axes.values():
                a.set_prop_cycle(color=list(colors))

        # Draw left
        self._bar(agg_left, ax=axes["D1"])
        if left_title:
            axes["D1"].set_title(left_title)

        # Right chart (counts)
        cfg_right = self._set_params(data=data, x=x, y=None, group_by=group_by, aggfunc=aggfunc, **kwargs)
        agg_right = self._sort(self._aggregate_data())
        self._bar(agg_right, ax=axes["D2"])
        if right_title:
            axes["D2"].set_title(right_title)

        # Style
        for a in axes.values():
            self._apply_common_style(a)

        # finalise with shared legend / title
        # Carry over legend + optional global title
        self._set_params(legend=legend, title=kwargs.get("title", None), **kwargs)
        handles, labels = [], []
        for a in axes.values():
            h, l = a.get_legend_handles_labels()
            handles += h
            labels += l
        self._finalise_layout_multi(fig, axes, handles, labels)

        cfg = self._chart_params
        if cfg.save_path:
            fig.savefig(cfg.save_path, bbox_inches="tight", dpi=fig.dpi)
        if cfg.show_fig:
            plt.show()

        # TODO: adjust to have grouped data as an (optional) return value i.e. return fig, ax, data
        # ideally have show_fig happen with this so e.g. you call fig, ax, df = cm.bar(...) and it displays the chart
        # and then you can use display(df) or similar to see the values used in the chart

        return fig, axes





## trash

def line():

    if trendline == 'linear':
        try:
            if pd.api.types.is_datetime64_any_dtype(s[x]):
                xnum = s[x].astype('int64') // 10**9
            else:
                xnum = pd.to_numeric(s[x])
            coeffs = np.polyfit(xnum, s[y].astype(float), 1)
            poly = np.poly1d(coeffs)
            if forecast_steps:
                step = (xnum.max() - xnum.min())/(len(xnum)-1) if len(xnum)>1 else 1
                ext_x = np.concatenate([xnum, xnum.max() + np.arange(1, forecast_steps+1)*step])
                ys = poly(ext_x)
                if pd.api.types.is_datetime64_any_dtype(s[x]):
                    ext_x_dt = pd.to_datetime(ext_x*10**9)
                    ax.plot(ext_x_dt, ys, linestyle='--', linewidth=1)
                else:
                    ax.plot(ext_x, ys, linestyle='--', linewidth=1)
            else:
                xs_sorted = np.sort(xnum)
                ax.plot(pd.to_datetime(xs_sorted*10**9) if pd.api.types.is_datetime64_any_dtype(s[x]) else xs_sorted, np.polyval(coeffs, xs_sorted), linestyle='--', linewidth=1)

def scatter():
    
    df = data.copy()
    fig, ax = plt.subplots(figsize=figsize)
    xs = pd.to_numeric(df[x], errors='coerce')
    ys = pd.to_numeric(df[y], errors='coerce')

    if jitter and not pd.api.types.is_numeric_dtype(df[x]):
        cats = df[x].astype('category')
        xnum = cats.cat.codes.astype(float)
        xnum = xnum + np.random.normal(0, jitter, size=len(df))
        xs = xnum
        ax.set_xticks(range(len(cats.cat.categories)))
        ax.set_xticklabels(cats.cat.categories, rotation=45, ha='right')

    sizes = None
    if size and size in df.columns:
        sizes = (pd.to_numeric(df[size], errors='coerce').fillna(1).values - pd.to_numeric(df[size], errors='coerce').min() + 1) * 10

    colors = None
    if hue and hue in df.columns:
        cats = df[hue].astype('category')
        pal = cycle_palette(self.default_palette, len(cats.cat.categories))
        color_map = {cat: pal[i % len(pal)] for i, cat in enumerate(cats.cat.categories)}
        colors = df[hue].map(color_map).values

    ax.scatter(xs, ys, s=sizes, c=colors if colors is not None else color)

    if highlight and highlight in df.columns:
        mask = df[highlight].astype(bool)
        if mask.any():
            ax.scatter(xs[mask], ys[mask], facecolors='none', edgecolors=highlight_color, linewidths=1.5, s=(sizes[mask] if sizes is not None else 40))

    if annotate and annotate in df.columns:
        for xi, yi, txt in zip(xs, ys, df[annotate]):
            ax.annotate(str(txt), (xi, yi), fontsize=7, alpha=0.8)

def hist():
    pass
    
def pie():
    pass



```


### Prompt playground

Help me decide what to work on next. My options are:
1. Implement a new comparison() chart type which places 2 charts next to each other (and possibly 3) with a shared axis. I've been trying to structure my code to make this easier (e.g. )