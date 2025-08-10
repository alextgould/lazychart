from .core import ChartMonkey

# __all__ controls what is imported when you use "from lazychart import *"
__all__ = ["ChartMonkey", "default_plotter", "bar", "line", "scatter", "hist", "pie"]

cm = ChartMonkey()

# these functions allow you to not have to instantiate ChartMonkey
# useful where you don't need advanced functionality (e.g. generating data)
# and don't need multiple instances of the class (e.g. with different sticky settings)

def bar(*args, **kwargs):
    return cm.bar(*args, **kwargs)

def line(*args, **kwargs):
    return cm.line(*args, **kwargs)

def scatter(*args, **kwargs):
    return cm.scatter(*args, **kwargs)

def hist(*args, **kwargs):
    return cm.hist(*args, **kwargs)

def pie(*args, **kwargs):
    return cm.pie(*args, **kwargs)
