# Prompt history

## Initial "impossible prompt" for GPT-5

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

## Prompt to give context to GPT-5 while editing core.py

**Background**

I am developing a python package. The vision is to be able to rapidly generate data visualisations using Python. My aim is to make this process as clean and simple for the user as possible, using:
* sensible defaults and arguments to easily control things like x/y axis limits, x/y axis labels, title, subtitle, legends, presence/absence of gridlines etc without having to look up and add multiple lines of raw matplotlib code
* the capacity (and likely default behaviour) of remembering things such as axis customisations and source datasets, so you can be lazy when creating successive plots (and for this reason the package is called sticky or stickyplot)

I want to be able to generate most of the standard sorts of plots that someone using Excel for example would be able to generate - for example:
* histograms, both in the traditional sense with a division of a numerical data variable's range into bins, as well as against a categorical variable, with it being easy to do either the absolute count of rows or % (i.e. frequency distribution)
* line charts, with different lines (in Excel this would be "series") for each value of a categorical variable
* bar charts (vertical and horizontal, values and proportions (in Excel this would "100% stacked chart")), with stacking/clustering by a categorical variable (or 2 with multiple levels)
* scatter/bubble chart (x/y with optional additional variables for color/size of points)
* pie chart (ideally just need to specify the category variable and then % rows is done automatically, with % value done if another numeric variable is specified)

Additional features that are optional but could be valuable:
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

I'm building upon raw matplotlib to avoid dependencies and give more control. I'm assuming data will be generally passed as pandas dataframes. It's worth noting that I don't want to work towards interactive charts at this stage, and want to focus only on more quickly churning out static visualisations.

The class also has a function to generate synthetic data for the purpose of demonstration/testing. Ideally the data is be interesting and not the typical "sales by month by region" data that is used everywhere. The data needs to be rich enough with both quantitative (numeric) and qualitative (categorical) columns, a temporaral (time) dimension, and potentially include ordinal (ordered) and nominal (not ordered) categories with a mixture of cardinality (many vs few categorical levels).

I have an existing code base, and the current version of my core.py is below:

<TODO: paste core.py here>

**Your task**

<TODO: add specific question(s) at this point>

### Sub-prompt: Refactor

I recently refactored the first half of this code to add decorators so I can centralise logic where possible and make it easier to expand this to incorporate new charts, following a "DRY" (don't repeat yourself) programming mindset. However, I haven't adjusted the individual plotting functions. Rework these to reflect the logic that has been stripped out by the decorator functions, aiming to make them as concise as possible. Add docstrings and comments throughout to make it easier for someone to review the code and easily understand what is going on - although keep this fairly concise - perhaps 2-5 comments per function to identify the general structure and a decent docstring to help users understand the API.

#### Iterative improvements round 1

Let's address some feedback on the bar function before we extend this to the others:

1. I know "hue" is used in matplotlib language but I'd prefer to make this group, group_by, by_col or something along those lines, as it's less about the hue/colour and more about the categorical nature of the column. any other naming suggestions and/or which one do you think fits best / is used in other software/libraries?

2. savefig feels like something that can be handled in the show_plot decorator don't you think? originally I had assumed a user would pass show_func = false and manually handle the saving but on reflection, I think we could have a save_path which if passed leads to the figure being saved at that path? I think save_path can be independent of the show_fig / return fig aspect. Give me revised show_plot wrapper code that handles the saving.

3. in google sheets there are three options for stacking: none (no stacking), standard (stacked), 100% (shows proportions). let's ensure we have these three options, although I'm sure if it makes more sense to have two parameters (e.g. stacked bool and proportions bool) or one parameter with multiple values (e.g. stacking as 'none' 'standard' or '100%'). I suspect the latter makes more sense?

4. aggregation of small categories also feels liek something that can be handled in one of the decorator functions. for example, we might want to group up the small pie chart slices or combine the lines for minor categories in a line chart. which decorator function would this fit best into and have a go at coding this one.

5. figsize and palette feel like these can also be moved to a decorator. I think it might be smarter to have separate palette controlling functions within the class and use settings that apply them to all charts by default, rather than trying to pass settings through for each chart individually, so remove this from the bar function for now. on the topic of figsize, I note we have fig, ax = plt.subplots in def bar, but should this be created in one of the decorator functions also? similar to palette, I feel that figsize is something I want control over but maybe not for each chart individually and/or through a decorator rather than individual plotting functions.

6. formatter - currently this is in my utils folder

def format_axis(ax, which: str = 'y', fmt: Optional[str] = None, decimals: int = 0):
    if fmt is None:
        return
    if fmt == 'percent':
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=decimals))
    elif fmt == 'comma':
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):,}"))
    elif fmt == 'short':
        def short(x, pos):
            x = float(x)
            if abs(x) >= 1_000_000:
                return f"{x/1_000_000:.{decimals}f}M"
            if abs(x) >= 1_000:
                return f"{x/1_000:.{decimals}f}k"
            return f"{x:.{decimals}f}"
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(short))

I think we need to leverage this better e.g. passing values for x_axis_format and y_axis_format arguments that can be 'percent' 'comma' 'short' and apply the formatter if requested. I also think this can be done in a decorator function?

Based on the above feedback, give me a rewrite of bar() and of any of the decorator functions show_plot() sticky_args() common_args() to address as much of my feedback as possible.




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
