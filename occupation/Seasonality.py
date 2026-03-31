from frstl.decompose import fast_robustSTL
import pandas as pd
import numpy as np

class Seasonality:
    """Extract seasonality from time series. Assuming a minimal of 3 weeks history.
    """
    def __init__(self):
        self.season_len = 24 * 7
        # trend extraction regularization factors -> len = 2
        self.trend_regs = [1, 0.1]
        # season regularization factors -> len = 3 * m
        self.season_regs = [[1, 1, 1]] # lambda 1, 2, 3 for seasonality  2
        # weighting factor for each non-local seasonal filter (Eq. 10) -> len = m
        self.alpha = 0.5
        self.z = 1
        # denoising weights -> len = 2
        self.denoise_ds = [4.0, 1.0]
        # season decomposition weights -> len = 2
        self.season_ds = [5.0, 1.0]
        self.K = 3
        self.H = 2

    def compute(self, smp, idx):
        raw, t, s, r =\
        fast_robustSTL(smp, self.season_len, self.trend_regs, self.season_regs,
                       self.alpha, self.z, self.denoise_ds, self.season_ds, self.K, self.H)
        occ = self._postprocess(s, t, idx)
        return occ

    def _postprocess(self, s, t, idx):
        """
        post process data :
            - by taking min value between estimated occupancy and real water data.
            - normalize values to be > 0
            - return only the last week
        Arguments:
            - s [np.array] {p.K} weeks water seasonality
            - t [np.array] {p.K} weeks water trend
            - idx [np.array] sample index
        returns:
        occ [pandas.DataFrame] with columns={'occupancy'}, index=timestamp  occupancy values (m3/h)
        """
        fix = [x if x < 0 else 0 for x in (pd.Series(
            t - np.median(t)).shift(-int(self.H)).fillna(0).values)]
        adjusted = [x - np.median(s)+y if -1*x < y else 0 for (x, y) in zip(s, fix)]
        _df = pd.DataFrame(adjusted, columns=["occupancy_value"], index=idx).shift(int(self.H)).fillna(0)
        # Because only we're only interested in the last week:
        s = _df.iloc[-7*24:]
        return s