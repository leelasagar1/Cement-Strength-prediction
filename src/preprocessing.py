from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd


class Preprocessor:

    # check null present in columns
    def is_null_present(self, data):
        null_counts = data.isna().sum()

        null_present = False
        cols_with_null_vals = []
        for col in null_counts.index:

            if (null_counts[col]) > 0:
                null_present = True
                cols_with_null_vals.append(col)

        return null_present, cols_with_null_vals

    # drop nan values
    def remove_nan(self, data):
        data = data.dropna()
        return data

    # replace missing values with nearest values mean
    def impute_missing_values(self, data):

        imputer = KNNImputer(n_neighbors=3, weights="uniform", missing_values=np.nan)
        array = imputer.fit_transform(data)
        data = pd.DataFrame(data=array, columns=data.columns)
        return data

    # seperate target column from data
    def seperate_target_feature(self, data, target_column):

        X = data.drop(target_column, axis=1)
        y = data[target_column]

        return X, y

    # apply log transform on features
    def log_transformation(self, data):

        for col in data.columns:
            data[col] += 1
            data[col] = np.log(data[col])

        return data

    # standardize data
    def standard_scale_data(self, data):

        scalar = StandardScaler()
        scaled_data = scalar.fit_transform(data)

        return scaled_data
