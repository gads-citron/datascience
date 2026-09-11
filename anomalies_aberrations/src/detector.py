import numpy as np
import pandas as pd
import ruptures as rpt
import param as p

def null_ranges(df):
    return df["raw_value"].eq(0)

def segmentize(df, reset_ts_list):
    segment = df.copy()
    is_reset = segment['payload_timestamp'].isin(reset_ts_list)
    segment['segment_id'] = (is_reset).cumsum()
    return segment

def rupture_detection(segment, pen=p.PEN, min_ratio=p.MIN_RATIO, clip_negative=p.CLIP_NEGATIVE,
                      apply_log=p.APPLY_LOG, min_size=p.MIN_SIZE):
    is_changepoint = pd.Series(False, index=segment.index)
    smoothed = segment['raw_value']
    valid_index = segment.index[smoothed.notna()]
    raw_signal = smoothed.loc[valid_index].to_numpy()
    signal = np.clip(raw_signal, 0, None) if clip_negative else raw_signal
    if apply_log:
        signal = np.sign(signal) * np.log10(np.abs(signal) + 1)
    signal = signal.reshape(-1, 1)
    if len(signal) < 2:
        return is_changepoint

    algo = rpt.Pelt(model='l2', min_size=min_size, jump=3).fit(signal)
    bkps = algo.predict(pen=pen)
    bounds = [0] + bkps

    if min_ratio is None:
        kept = [mid for start, mid, end in zip(bounds, bounds[1:], bounds[2:])]
    else:
        # Filter over only breakpoint where value is really higher than previous one
        kept = []
        for start, mid, end in zip(bounds, bounds[1:], bounds[2:]):
            before_mean = raw_signal[start:mid].mean()
            after_mean = raw_signal[mid:end].mean()
            is_up = after_mean >= min_ratio * max(before_mean, 1e-9)
            if is_up :
                kept.append(mid)
    if kept:
        is_changepoint.loc[valid_index[kept]] = True
    return is_changepoint

def reset_compteurs(df):
    is_reset = df['raw_value'].diff() < 0
    return is_reset

def find_anomalies(sample):
    anom_df = []    
    start = sample.payload_timestamp.min()
    end = sample.payload_timestamp.max()
    null_val = null_ranges(sample)
    anom_df.append(sample.loc[null_val, ['device_id', 'sensor_id', 'payload_timestamp', 'raw_value']]\
                                    .assign(anomaly_type='is_null', date_start=start, date_end=end))
    sample = sample.loc[~null_val]
    reset_cpt = reset_compteurs(sample)
    anom_df.append(sample.loc[reset_cpt, ['device_id', 'sensor_id', 'payload_timestamp', 'raw_value']]\
                                    .assign(anomaly_type='reset', date_start=start, date_end=end))
    segment = segmentize(sample, sample.loc[reset_cpt, 'payload_timestamp'].tolist())
    for _, segment_df in segment.groupby('segment_id'):
        is_rupture = rupture_detection(segment_df)
        segment_start = pd.to_datetime(segment_df['payload_timestamp'].min(), unit='s')
        segment_end = pd.to_datetime(segment_df['payload_timestamp'].max(), unit='s')
        anom_df.append(segment_df.loc[is_rupture, ['device_id', 'sensor_id', 'payload_timestamp', 'raw_value']]\
                            .assign(anomaly_type='rupture', date_start=segment_start, date_end=segment_end))
    return anom_df

#-----# Deprecated for now
def commissioning(df, start):
    valid = df.dropna(subset=['raw_value'])
    valid = valid[valid['raw_value'] != 0]
    empty_result = pd.DataFrame(columns=['device_id', 'sensor_id', 'ts', 'raw_value', 'anomaly_type', 'date', 'segment_start', 'segment_end'])
    if valid.empty:
        return empty_result

    first = valid.iloc[0]
    date = pd.to_datetime(first['ts'], unit='s', utc=True).tz_convert('Europe/Paris')
    if date.date() == pd.Timestamp(start).date():
        return empty_result
    return pd.DataFrame([{
        'device_id': first['device_id'],
        'sensor_id': first['sensor_id'],
        'ts': first['ts'],
        'raw_value': first['raw_value'],
        'anomaly_type': 'commissioning',
        'date': date,
        'segment_start': date,
        'segment_end': date,
    }])

def multi_freq_detection(df, e=1):
    delta = df['ts'].diff()
    T = delta.median()
    r = delta % T
    proportions = delta[r > 0].value_counts(normalize=True)
    proportions = proportions[proportions >= 0.2]
    if len(proportions) > 1:
        if proportions.index.min() <= 5:
            return proportions
        if np.sum(proportions.index % proportions.index.min()) > 0:
            return proportions
    return []

def multi_frequence(df, start, end):
    periodes = multi_freq_detection(df)
    res = pd.DataFrame(columns=['device_id', 'sensor_id', 'ts',
                                'raw_value', 'anomaly_type', 'date',
                                'segment_start', 'segment_end'])
    if len(periodes) > 1 and periodes.index.max() - periodes.index.min() > 10:
        start_ts = pd.Timestamp(start).tz_localize('Europe/Paris')
        end_ts = pd.Timestamp(end).tz_localize('Europe/Paris')
        res = pd.DataFrame([{
            'device_id': df['device_id'].iloc[0],
            'sensor_id': df['sensor_id'].iloc[0],
            'ts': np.nan,
            'raw_value': np.nan,
            'anomaly_type': 'multi-freq',
            'date': start_ts,
            'segment_start': start_ts,
            'segment_end': end_ts,
        }])
    return res