import sys, time, logging
sys.path.insert(0, '/opt/ml/processing/dependencies/')
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from Seasonality import Seasonality

def preprocess_elec(s):
    """
    replace negative value by nan and replace nans by previous value
    """
    s[s<0] = np.nan
    s = s.ffill()
    return s
    
def preprocess_water(s, y_col):
    """
    Parameters
    ----------
    s : pandas.Series()
        3 weeks water flow consumption (without spraying/fire network)
    y_col : string column name used for data collect

    Returns
    -------
    s.flow : pandas.Series()
        3 weeks water flow consumption without potential peak anomaly.

    """
    q3 = s.quantile(0.75)
    q1 = s.quantile(0.25)
    u_fence = q3 + (1.5 * (q3 - q1))
    s = pd.DataFrame(s)
    id_error = s.loc[(s[y_col] > u_fence)].index
    s['h'] = s.index.weekday.astype(str) + s.index.hour.astype(str)
    idx = [str(x)+str(y) for x in range(7) for y in range(24)]
    g = s.loc[(s[y_col] <= u_fence)].groupby('h').mean()
    g = pd.DataFrame(index=idx).merge(g,
                                      right_index=True,
                                      left_index=True, how='outer')
    g = g.ffill()
    s['avg'] = s['h'].map(g[y_col])
    s.loc[id_error, y_col] = s.loc[id_error, "avg"].astype(float).values
    return s[y_col]

WINDOW_SIZE = pd.Timedelta(weeks=3)
STEP_SIZE = pd.Timedelta(weeks=1)
#%%
input_path = "/opt/ml/processing/input"
output_path = "/opt/ml/processing/output"
x_time = 'payload_expected_time'
y_column = 'raw_value'

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
t0 = time.time()
logging.info("Démarrage du traitement")
# Lire parquet
scaler = MinMaxScaler(feature_range=(0, 1))

all_files = sorted(Path(input_path).rglob("*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in all_files])

df = df.resample('1h', on=x_time)[y_column].sum().astype(float)
# smp = preprocess(df, y_column)
smp = preprocess_elec(df)
s_extract = Seasonality()
start = smp.index.min()
end = smp.index.max()

success_count = 0
while start + WINDOW_SIZE <= end:
    window_end = start + WINDOW_SIZE
    window = smp.loc[start:window_end]
    scaled = scaler.fit_transform(pd.DataFrame(window)).flatten()
    idx = window.index
    window_label = start.strftime("%Y%m%d")
    out_file = Path(output_path) / f"{window_label}.parquet"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    try : 
        occ = s_extract.compute(scaled, idx)
        occ.reset_index(inplace=True)
        occ.rename({x_time:'payload_timestamp'}, inplace=True, axis=1)
        occ.to_parquet(out_file, index=False)
        logging.info(f"[{input_path}] window {start.date()} → {window_end.date()} : done")
        success_count += 1
    except Exception as e:
        logging.error(f"frstl_pb [{input_path}] window {start.date()}: {e}")
    start += STEP_SIZE
# Sauvegarde
logging.info(f"Job terminé en {time.time() - t0:.1f}s — {success_count} fenêtres traitées sans erreur.")
