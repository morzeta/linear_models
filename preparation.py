import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo

pd.options.mode.chained_assignment = None  # default='warn'

data_columns = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)',
                'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
                'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']


def read_data(data_id: int = 360) -> pd.DataFrame:
    air_quality = fetch_ucirepo(id=data_id)

    air_quality.data.features["Time"] = np.array([int(i.split(":")[0]) for i in air_quality.data.features["Time"]])

    data = random_fillna(air_quality.data.features[air_quality.data.features.columns[1:]])
    return data


# for this project we consider Time as y data and the rest as x data
def get_x_y(data, normalize=True):
    x = data[data.columns[1:]]
    # normalize data with mean and std
    if normalize:
        x = (x - x.mean()) / x.std()

    y = pd.DataFrame(data[data.columns[0]])
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
