import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo
pd.options.mode.chained_assignment = None  # default='warn'


def read_data(data_id: int = 360) -> (pd.DataFrame, pd.DataFrame):
    col = ['CO_GT', 'PT08_S1_CO', 'NMHC_GT',
           'C6H6_GT', 'PT08_S2_NMHC', 'NOX_GT', 'PT08_S3_NOX',
           'NO2_GT', 'PT08_S4_NO2', 'PT08_S5_O3', 'T', 'RH', 'AH']
    air_quality = fetch_ucirepo(id=data_id)
    x = air_quality.data.features[air_quality.data.features.columns[2:]]
    x = random_fillna(x)

    air_quality.data.features["Time"] = np.array([int(i.split(":")[0]) for i in air_quality.data.features["Time"]])
    y = pd.DataFrame(air_quality.data.features[air_quality.data.features.columns[1]])
    return x, y


def random_fillna(dataframe) -> pd.DataFrame:
    # making sure we have positive values
    while (dataframe.values < 0).any():
        # replace negative values (which are actually NaN) with NaN
        dataframe.mask(dataframe < 0, inplace=True)
        # generating random values from a normal distribution with mean and std of the column
        M = len(dataframe.index)
        ran_df = pd.DataFrame(columns=dataframe.columns, index=dataframe.index)
        for col in dataframe.columns:
            ran_df[col] = np.random.normal(loc=dataframe[col].mean(), scale=dataframe[col].std(), size=M)
        # replacing NaN values
        dataframe.update(ran_df, overwrite=False)
    return dataframe