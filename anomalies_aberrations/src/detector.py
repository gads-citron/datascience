import numpy as np
import pandas as pd
import ruptures as rpt
import param as p

def null_ranges(df, start_list):
    """
    Identify null ranges and valid segments between them.
    Args:
        df: DataFrame with 'ts' (timestamp in seconds)
        start_list: List of timestamps marking start of null periods
    Returns:
        null_ranges: List of tuples (start_date, end_date) for null periods
        valid_segments: List of tuples (preceding_ts, following_ts) for valid segments
    """
    null_ranges_list = []
    valid_segments_list = []
    start_list_sorted = sorted(start_list)

    if start_list_sorted[0] > df.ts.min():
        # Add segment before first null_range: from first reading to last reading before first null_start
        first_before_null = df[df['ts'] < start_list_sorted[0]]['ts'].max()
        valid_segments_list.append((df['ts'].min(), first_before_null))

    # Process each null_range
    i = 0
    while i+1 <= len(start_list_sorted):
        null_start = start_list_sorted[i]
        null_end = df[(df['ts'] > null_start) & (df['raw_value'] == 0)]['ts'].max()
        null_ranges_list.append((null_start, null_end))
        i+=1

    if start_list_sorted[-1] < df.ts.max():
        # Add segment after last null_range
        first_after_null = df[df['ts'] > null_end]['ts'].min()
        valid_segments_list.append((first_after_null, df['ts'].max()))
    return null_ranges_list, valid_segments_list

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

def segmentize(df, min_duration_hours=p.MIN_SEGMENT_DURATION_HOURS):
    segment = df.copy()
    reset_ts_list, null_range_list = reset_compteurs(segment)
    is_reset = segment['ts'].isin(reset_ts_list)
    segment['time_delta_seconds'] = segment['ts'].diff()
    T = segment['time_delta_seconds'].median()
    is_cut = (segment['time_delta_seconds'] > T+600) & (segment['time_delta_seconds'] % T == 0)    
    segment['segment_id'] = (is_cut | is_reset).cumsum()

    segment_duration = segment.groupby('segment_id')['ts'].transform(lambda s: s.max() - s.min())
    segment = segment[segment_duration >= min_duration_hours * 3600]
    return segment, null_range_list

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

    algo = rpt.Pelt(model='l2', min_size=min_size).fit(signal)
    bkps = algo.predict(pen=pen)
    bounds = [0] + bkps

    if min_ratio is None:
        # pas de filtre : on garde toutes les ruptures trouvées par PELT
        kept = [mid - 1 for start, mid, end in zip(bounds, bounds[1:], bounds[2:])]
    else:
        # ne garde que les ruptures où le segment suivant est nettement PLUS HAUT que le précédent
        kept = []
        for start, mid, end in zip(bounds, bounds[1:], bounds[2:]):
            before_mean = raw_signal[start:mid].mean()
            after_mean = raw_signal[mid:end].mean()
            if after_mean >= min_ratio * max(before_mean, 1e-9):
                kept.append(mid - 1)
    if kept:
        is_changepoint.loc[valid_index[kept]] = True
    return is_changepoint

def reset_compteurs(df):
    is_reset = df['raw_value'].diff() < 0
    reset_with_zero = is_reset & (df['raw_value'] == 0)

    null_range_list = []
    flat = []

    if reset_with_zero.any():
        start_list = df.loc[reset_with_zero, 'ts'].tolist()
        null_range_list, valid_segment = null_ranges(df, start_list)
        flat = sorted([t for pair in null_range_list for t in pair])
    return flat, null_range_list

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