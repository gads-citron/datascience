import sys
sys.path.insert(0, '/opt/ml/processing/input/dependencies/')
import time, logging
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

#%%
input_path = "/opt/ml/processing/input"
output_path = "/opt/ml/processing/output"
x_time = 'payload_expected_time'
y_column = 'raw_value'

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
t0 = time.time()
logging.info("Démarrage du traitement")

# Lire parquet
files = list(Path(input_path).glob("*.parquet"))
source_name = files[0].stem 
df = pd.concat([pd.read_parquet(f) for f in files])

# Exemple preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
df = df.resample('1h', on=x_time)[y_column].sum().astype(float)
# smp = preprocess(df, y_column)
smp = preprocess_elec(df)
scaled = scaler.fit_transform(pd.DataFrame(smp)).flatten()
idx = smp.index
s_extract = Seasonality()
try : 
    occ = s_extract.compute(scaled, idx)
    occ.reset_index(inplace=True)
    occ.rename({x_time:'payload_timestamp'}, inplace=True, axis=1)
    occ.to_parquet(f"{output_path}/processed.parquet", index=False)
except Exception as e:
    logging.error(f"frstl_pb: {e}")
# Sauvegarde
logging.info(f"Terminé en {time.time() - t0:.1f}s")
