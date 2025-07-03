import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import pandas as pd
import numpy as np


df = pd.read_csv("C:\\Users\\garma\\Downloads\\archive (1)\\weatherAUS.csv")

# Проверка на NaN
print("NaN по колонкам:")
print(df.isna().sum())

# Проверка на Inf
print("\nInf значения:", (df == float('inf')).sum().sum())

# Проверка на нечисловые значения в числовых колонках
print("\nТипы данных:")
print(df.dtypes)

# # Посмотреть уникальные значения
# print("\nУникальные значения в horsepower:")
# print(df['horsepower'].unique())       

def clean_car_data(path):
    df = pd.read_csv(path)

    # Заменим '?' на NaN
    df.replace('?', np.nan, inplace=True)

    # Преобразуем числовые колонки
    numeric_cols = ['horsepower', 'peakrpm', 'stroke', 'boreratio']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    wordNum_locs = ['doornumber', 'cylindernumber']
    DictWordToNum = {
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
    }
    for col in wordNum_locs:
        df[col] = df[col].str.lower().map(DictWordToNum)

    cat_columns = ['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'fuelsystem']
    for col in cat_columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Удалим строки с пропусками (можно заменить на fillna)
    df = df.dropna()

    # Удалим ID (бесполезен)
    df.drop(columns=['car_ID', 'CarName'], inplace=True)

    return df

def clean_weather_data(path):
    df = pd.read_csv(path)

    # Заменим '?' на NaN
    df.replace('?', np.nan, inplace=True)

    # # Преобразуем числовые колонки
    # numeric_cols = ['horsepower', 'peakrpm', 'stroke', 'boreratio']
    # for col in numeric_cols:
    #     df[col] = pd.to_numeric(df[col], errors='coerce')

    cat_columns=['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    for col in cat_columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Удалим строки с пропусками (можно заменить на fillna)
    df = df.dropna()

    # Удалим ID (бесполезен)
    # df.drop(columns=['car_ID', 'CarName'], inplace=True)


    return df

df = clean_weather_data("C:\\Users\\garma\\Downloads\\archive (1)\\weatherAUS.csv")

# Проверка на нечисловые значения в числовых колонках
print("\nТипы данных:")
print(df.dtypes)

class CSVDataset(Dataset):
    def __init__(
        self, 
        dataframe, 
        target_column, 
        cat_columns=None, 
        num_columns=None, 
        normalize=True, 
        encode='label'
    ):
        """
        :param dataframe: чистый датасет
        :param target_column: название целевого столбца
        :param cat_columns: список категориальных признаков
        :param num_columns: список числовых признаков
        :param normalize: нужно ли нормализовать числовые признаки
        :param encode: 'label' или 'onehot' для кодирования категорий
        """
        self.df = dataframe.copy()
        self.cat_columns = cat_columns or []
        self.num_columns = num_columns or []
        self.target_column = target_column
        self.encode = encode

        # Сохраним преобразователи
        self.encoders = {}
        self.scaler = None

        # Кодирование категориальных признаков
        if self.cat_columns:
            for col in self.cat_columns:
                if encode == 'label':
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col])
                    self.encoders[col] = le
                elif encode == 'onehot':
                    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    onehot = ohe.fit_transform(self.df[[col]])
                    onehot_df = pd.DataFrame(onehot, columns=[f"{col}_{cls}" for cls in ohe.categories_[0]])
                    self.df = pd.concat([self.df.drop(columns=[col]), onehot_df], axis=1)
                    self.encoders[col] = ohe

        # Нормализация числовых признаков
        if normalize and self.num_columns:
            self.scaler = StandardScaler()
            self.df[self.num_columns] = self.scaler.fit_transform(self.df[self.num_columns])

        # Формирование признаков и целевой переменной
        self.y = torch.tensor(self.df[target_column].values, dtype=torch.float32).unsqueeze(1)
        self.X = torch.tensor(self.df.drop(columns=[target_column]).values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
