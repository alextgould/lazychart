# sticky — quick, consistent static plotting for pandas

`sticky` is a lightweight Python helper to rapidly generate static plots from pandas DataFrames with sensible defaults and *sticky* parameters — it remembers previous arguments (like axis labels, data, palettes) so you can churn out many similar charts quickly without repeating boilerplate.

This repository is a starter implementation. It supports:
- `bar`, `line`, `scatter`, `hist`, `pie`
- Sticky behavior (persist last-used data and parameters)
- Simple trendline (linear) and forecasting extension
- Basic percent / axis formatting helpers and palettes
- Grouping small categories into "Other"
- Jitter for scatter, annotations, highlight points
- Demo data generator (emotion/time series) and usage script

See `examples/demo_usage.py` for runnable demo code.

# notes from initial chat with ChatGPT

How to run locally:

Download and unzip the archive.

Create a virtual environment and install dependencies:

```bash
python -m venv venv && source venv/bin/activate   # or venv\\Scripts\\activate on Windows
pip install -e .[all]  # or pip install -e .
pip install pandas numpy matplotlib scipy
```

Run the example script:

```bash
python examples/demo_usage.py
```

(Use a Jupyter notebook to see inline figures more conveniently.)

# Todo

Add LICENCE file