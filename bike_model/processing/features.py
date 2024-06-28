from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, col_name: str):

        if not isinstance(col_name, str):
            raise ValueError("Column should be a string")

        self.col_name = col_name

    def fit(self, dataframe: pd.DataFrame, target: pd.Series = None):

        return self

    def transform(self, dataframe: pd.DataFrame):

        # print("Imputing the day values from the extracted day information from date.")
        df = dataframe.copy()
        wkday_null_idx = df[df['weekday'].isnull() == True].index
        df.loc[wkday_null_idx, 'weekday'] = df.loc[wkday_null_idx, 'dteday'].dt.day_name().apply(lambda x: x[:3])

        return df


class ColumnDropper(BaseEstimator, TransformerMixin):

    def __init__(self, col_list: list):

        if not isinstance(col_list, list):
            raise ValueError("Columns should be a list of strings")

        self.col_list = col_list

    def fit(self, dataframe = pd.DataFrame, target: pd.Series = None):

        return self

    def transform(self, dataframe = pd.DataFrame, target: pd.Series = None):

        df = dataframe.copy()
        df.drop(columns=self.col_list,inplace=True)

        return df


class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, col_name: str):
        if not isinstance(col_name, str):
            raise ValueError("Column should be a string")

        self.col_name = col_name

    def fit(self, dataframe: pd.DataFrame, target: pd.Series = None):

        df = dataframe.copy()
        self.fill_value=df[self.col_name].mode()[0]
        return self

    def transform(self, dataframe: pd.DataFrame,):

        # print("Imputing the mode value from the weather column.")
        df = dataframe.copy()
        df[self.col_name] = df[self.col_name].fillna(self.fill_value)

        return df
    

class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, col_map: dict):

        if not isinstance(col_map, dict):
            raise ValueError("Mappings should be a dictionary of col, strings pair")
        self.col_map = col_map

    def fit(self, dataframe: pd.DataFrame, target: pd.Series = None):

        return self

    def transform(self, dataframe: pd.DataFrame):
        # print("Ordinal encoding of categorical features.")
        df = dataframe.copy()

        for key, val in self.col_map.items():
            df[key] = df[key].map(val)

        return df
    
    
class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, col_list: list):

        if not isinstance(col_list, list):
            raise ValueError("Columns should be a list of strings")
        self.col_list = col_list
        self.limit_dict = {}

    def fit(self, dataframe: pd.DataFrame, target: pd.Series = None):

        df = dataframe.copy()
        # limit_dict = {}

        for col in self.col_list:

            q1 = df.describe()[col].loc['25%']
            q3 = df.describe()[col].loc['75%']
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            self.limit_dict[col] = [lower_bound, upper_bound]

        self.limits = self.limit_dict
        return self


    def transform(self, dataframe: pd.DataFrame):
        # print("Handling outliers for numerical features.")
        df = dataframe.copy()

        for col in self.col_list:
            for i in df.index:

                if df.loc[i,col] > self.limits[col][1]:
                    df.loc[i,col]= self.limits[col][1]

                if df.loc[i,col] < self.limits[col][0]:
                    df.loc[i,col]= self.limits[col][0]

        return df
    

class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, col_list: list):

        if not isinstance(col_list, list):
            raise ValueError("Columns should be a list of strings")

        self.col_list = col_list
        self.categories_ = {}

    def fit(self, dataframe: pd.DataFrame, target: pd.Series = None):

        df = dataframe.copy()
        for col in self.col_list:
            self.categories_[col] = df[col].unique()
            self.categories_[col] = [col for col in self.categories_[col] if str(col) != 'nan']

        return self

    def transform(self, dataframe: pd.DataFrame):
        # print("One hot encoding for particular features.")
        if not self.categories_:
            raise ValueError("Must fit the transformer before transforming the data.")

        df = dataframe.copy()
        for col in self.col_list:
            categories = self.categories_[col]
            for category in categories:
                new_column_name = f"{col}_{category}"
                df[new_column_name] = (df[col] == category).astype(int)
            df = df.drop(col, axis=1)

        return df
