import numpy as np
import pandas as pd

from dhs.verdict import summary


def test_summary_statistics():
    x = pd.Series([1.0, -2.0, 0.5, 0.5, 1.0])
    s = summary(x)
    assert np.isclose(s["mean"], 0.2) and np.isclose(s["annual"], 2.4)
    assert np.isclose(s["t"], 0.2 / x.std() * np.sqrt(5))
    assert s["worst"] == -2.0 and np.isclose(s["pos"], 0.8)
    assert np.isclose(s["dd"], -2.0)  # running total 1, -1, -0.5, 0, 1: peak 1 to trough -1
