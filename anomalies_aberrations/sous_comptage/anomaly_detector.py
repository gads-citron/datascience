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
    def __init__(self, df, y_name):
        self.df = df
        self.y_name = y_name

    def iqr(self, k=100):
        s = self.df[self.y_name]
        q25, q75 = np.percentile(s, [25, 75])
        iqr = q75 - q25
        if iqr < 1:
            iqr = 10
        cutoff = iqr * k
        th = q75 + cutoff
        return s > th

    def stl(self, k=3):
        s = self.df.set_index('payload_expected_time')[self.y_name].asfreq('h').interpolate()
        decomp = STL(s, period=168, robust=True).fit()
        resid = decomp.resid
        anomalies = resid.abs() > k * resid.std()
        return anomalies.reindex(self.df.payload_expected_time).fillna(False).values

    def iso_forest(self, th=2):
        isof = IsolationForest(n_estimators=500)
        self.df['hour'] = self.df.payload_expected_time.dt.hour
        self.df['day_week'] = self.df.payload_expected_time.dt.day_of_week
        self.df['hour_week'] = (self.df.day_week * 24 + self.df.hour).astype('float32')

        isof.fit(self.df[['hour_week', self.y_name]])
        # scores = isof.score_samples(self.df[['hour_week', self.y_name]])
        # sns.lineplot(x = np.arange(1, scores.shape[0]+1), y = np.sort(scores))

        # isof.offset_ = np.percentile(scores, th)
        isof.offset_ = th

        isof.decision_function(self.df[['hour_week', self.y_name]])
        return isof.predict(self.df[['hour_week', self.y_name]]) == -1