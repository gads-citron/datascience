import pandas as pd
import numpy as np
from decompose import fast_robustSTL


class Seasonality:
    """Extract seasonality from time series. Assuming a minimal of 3 weeks history."""

    def __init__(self):
        self.season_len = 24 * 7
        self.trend_regs = [1, 0.1]
        self.season_regs = [[1, 1, 1]]
        self.alpha = 0.5
        self.z = 1
        self.denoise_ds = [4.0, 1.0]
        self.season_ds = [5.0, 1.0]
        self.K = 3
    
    def compute(self, smp, idx):
        raw, t, s, r = fast_robustSTL(
            smp, self.season_len, self.trend_regs, self.season_regs,
            self.alpha, self.z, self.denoise_ds, self.season_ds, self.K
        )
        return self._postprocess(s, t, idx)

    def _postprocess(self, s, t, idx):
        """
        Post-process STL output:
          - clip trend deviation to negatives only
          - zero out negative adjusted occupancy
          - return the last week

        Args:
            s: seasonality array from STL
            t: trend array from STL
            idx: sample DatetimeIndex
        Returns:
            DataFrame with column 'occupancy_value', index=timestamp, last 7*24 rows
        """
        trend_deviation = pd.Series(t - np.median(t)).fillna(0).values
        fix = np.minimum(trend_deviation, 0)

        adjusted = [
            x - np.median(s) + y if -x < y else 0
            for x, y in zip(s, fix)
        ]

        df = pd.DataFrame(adjusted, columns=["occupancy_value"], index=idx)
        df = df.ffill()
        # df = df.shift(int(self.H)).fillna(0)
        return df.iloc[-7 * 24:]
