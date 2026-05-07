import inspect
import numpy as np
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL

class DataProcessor:
    def __init__(self, res, df):
        self.row = res
        self.df = df

    def all_nulls(self, y_name):
        return self.df[y_name].sum() == 0

    def check_continuous(self, y_name):
        return int(self.df[y_name].value_counts().shape[0] < 10)

    def check_missing(self, y_name):
        missing_ratio = self.df[y_name].isna().mean()
        return missing_ratio > 0.8

    def preproc(self, y_name):
        if self.check_missing(y_name):
            self.row["error"] = "more_than_80%"
            return self.row
        if self.all_nulls(y_name):
            self.row["error"] = "Null values"
        if self.check_continuous(y_name):
            self.row["error"] = "continuous_values"
        return self.row

    def serie_typology(self, y_name, tol=0.3):
        score = np.mean((self.df[y_name].shift() - self.df[y_name]) < 0)
        return 0 if score > tol else 1  # 0=volume, 1=index

class Detector:
    def __init__(self, method, param):
        self.x_time = 'payload_expected_time'
        self.detect_fn = getattr(self, method)
        defaults = {
            k: v.default
            for k, v in inspect.signature(self.detect_fn).parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        defaults.update(param)
        self.param = defaults

    def predict(self, df, y_name):
        df['anomaly'] = self.detect_fn(df, y_name, **self.param)
        return df

    def iqr(self, df, y_name, k=100):
        s = df[y_name]
        q25, q75 = np.percentile(s, [25, 75])
        iqr = q75 - q25
        if iqr < 1:
            iqr = 10
        cutoff = iqr * k
        th = q75 + cutoff
        return s > th

    def stl(self, df, y_name, k=3):
        s = df.set_index(self.x_time)[y_name].asfreq('h').interpolate()
        decomp = STL(s, period=168, robust=True).fit()
        resid = decomp.resid
        anomalies = resid.abs() > k * resid.std()
        return anomalies.reindex(df.payload_expected_time).fillna(False).values

    def iso_forest(self, df, y_name, th=2):
        isof = IsolationForest(n_estimators=500)
        df['hour'] = df.payload_expected_time.dt.hour
        df['day_week'] = df.payload_expected_time.dt.day_of_week
        df['hour_week'] = (df.day_week * 24 + df.hour).astype('float32')

        isof.fit(df[['hour_week', y_name]])
        # scores = isof.score_samples(df[['hour_week', y_name]])
        # sns.lineplot(x = np.arange(1, scores.shape[0]+1), y = np.sort(scores))

        # isof.offset_ = np.percentile(scores, th)
        isof.offset_ = th

        isof.decision_function(df[['hour_week', y_name]])
        return isof.predict(df[['hour_week', y_name]]) == -1