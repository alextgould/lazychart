
# lazychart - quick, consistent static plots in python

![](./assets/img/gemini_monkey_cartoon.png)

`lazychart` is a lightweight Python helper to rapidly generate static plots from pandas DataFrames. It features sensible defaults and the option to use *sticky* parameters — it can remember previous arguments (like data, variables and axis labels) so you can churn out similar charts quickly without repeating boilerplate.

`lazychart` is currently in its early stages of development


## Testing

With `pyproject.toml` in place, and having activated the virtual environment, install with:

Having activated virtual environment:

```bash
pip install pytest
```

Run all tests in verbose mode:

```bash
pytest tests/test_core.py -v
```

### Installer

The above uses pytest.ini.

An editable install better mimics how users import the package after installation.

Run the following once per virtual env (or occasionally) to keep the installer modern.

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -e .
```

An "editable" (a.k.k. develop) model install writes metadata into the *.egg-info directory. This is in contrast to `pip install .` which builds the distribution and copies the built package into site-packages (located within venv) as a static snapshot.

## Semantic versions

The version number in `pyproject.toml` follow standard versioning i.e. x.y.z with
x = major version
y = new release with features or any breaking change
z = bugfixes and non-breaking tweaks

x/y have corresponding git tags e.g.

```bash
git tag -a v0.1.0 -m "0.1.0: post-refactor, tests passing"
git push && git push --tags
```