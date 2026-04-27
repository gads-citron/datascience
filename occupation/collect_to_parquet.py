import pandas as pd
from pathlib import Path
import datetime as dt

device = "69cd33080630341bd0015c99"
sensor = "662bc1c93f0afb78ad05983d"
site_id = "69cd2f4e0630341bd00149b4"
start = "2026-01-01 00:00:00"
end = "2026-06-30 23:59:59"
x_time = 'payload_expected_time'
output_dir = './data_collect/sample_s3/'

def check_problems(s, errors, y_col):
    """
        Parameters
        ----------
        s : sample data to be checked
        errors : log map to be filled
        y_col : col in "datamart_sous_comptage" used to read data
        Returns
        -------
        errors : map of error description
        flag_return : 0 if no problem; 1 else
        y_col : same as input if no change
        datat : datatype for tracking purpose (data index or volume)
    """
    datat = 'volume'
    flag_return = 0
    if s.empty:
        errors['error'] = "no_data"
        flag_return = 1
        return errors, flag_return, y_col, datat
    if s[y_col].isna().mean() > 0.1:
        errors['error'] = "too_many_NAN"
        flag_return = 1
    if serie_typology(s[y_col]):
        y_col = 'eval_prev_hour'
        datat = 'index'
        if serie_typology(s[y_col]):
            errors['error'] = "index_serie_on_eval_prev"
            flag_return = 1
    if s[y_col].astype(float).sum() == 0:
        errors['error'] = "zero_values"
        flag_return = 1
    if s[y_col].value_counts().shape[0] < 100:
        errors['error'] = "continuous"
        flag_return = 1
    if s.shape[0] < 168 * 3:
        errors['error'] = "history_not_enough"
        flag_return = 1
    return errors, flag_return, y_col, datat

def export_sliding_parquets(df, y_col, start_date, end_date,
                            site_id, device_id, sensor_id, output_dir):
    date_col = 'payload_expected_time'
    if df[date_col].dt.tz is not None:
        df[date_col] = df[date_col].dt.tz_localize(None)
    df.sort_values(date_col, inplace=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    start = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    while start <= end_date:
        end = start + pd.Timedelta(weeks=3)
        window_df = df[(df[date_col] >= start) & (df[date_col] < end)]
        partition_dir = (
                output_path /
                f"site_id={site_id}" /
                f"device_id={device_id}" /
                f"sensor_id={sensor_id}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)

        file_name = (
            f"data_{start.strftime('%Y%m%d')}_"
            f"{end.strftime('%Y%m%d')}.parquet"
        )

        file_path = partition_dir / file_name
        window_df = window_df.copy()
        window_df['raw_value'] = window_df[y_col]
        window_df['site_id'] = site_id
        window_df['raw_value'] = window_df[y_col]
        cols = ['site_id', 'device_id', 'sensor_id', 'payload_expected_time', 'raw_value']
        window_df[cols].to_parquet(file_path, index=False)

        # Décalage de 1 semaine
        start += pd.Timedelta(weeks=1)


df = prod_client.query_df(f"""
    SELECT site_id, device_id, {x_time}, sum_payload_value_hour_curr, eval_prev_hour, sensor_id
    FROM {p.datasource}
    WHERE 1=1
    AND {x_time} between '{start}' and '{end}'
    AND device_id = '{device}'
    AND sensor_id = '{sensor}'
    ORDER BY {x_time} DESC
    settings max_rows_to_read = 10000000000000;""")

error_dict, flag, y_column, datatype = check_problems(df, error_dict, y_column)
if flag:
    print(f"""poor quality data... {error_dict['error']}""")
else:
    export_sliding_parquets(df, y_column, start, end,
                            site_id, device, sensor, output_dir)