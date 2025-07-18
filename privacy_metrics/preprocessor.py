import pandas as pd
from typing import List, Dict
from itertools import islice
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from tqdm import tqdm
from enum import Enum

class Data(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    CATEGORICAL_ORD = "categorical_ord"
    DATE = "date"
    DATETIME = "datetime"
    INTEGER = "integer"


def raiser(): raise ValueError('attribute type must be in ["numeric", "categorical", "ordinal", "already_preprocessed"]')

# Possible datatypes: float -> StandardScaler
#                     str -> LabelEncoder
#                     int -> LabelEncoder
#                         -> OrdinalEncoder
#                         -> StandardScaler
#                     datetime ->



class StandardScalerNAN:
    def __init__(self):
        self.preproc = StandardScaler()

    def fit(self, data: pd.Series) -> None:
        data = data.to_frame()
        self.preproc.fit(data)

    def transform(self, data: pd.Series):
        data = data.to_frame()
        data = data.astype(float)
        data.fillna(data.mean(), inplace=True)
        # print(self.preproc.transform(data).shape)
        return self.preproc.transform(data)

class LabelEncoderUnseenCategories:
    def __init__(
            self,
            min_freq: int = 2,
            max_num_categories: int = 100,
    ):
        self.preproc = LabelEncoder()
        self.min_freq = min_freq
        self.max_num_categories = max_num_categories
        self.category_to_index = {}
        self.index_to_category = {}
        self.unknown_token = "__unknown__"

    def _find_common_categories(
            self,
            data: pd.Series
    ) -> List:
        categories_count = data.value_counts().to_dict()
        selected_categories = list(islice([k for k in categories_count.keys() if categories_count[k] > self.min_freq], self.max_num_categories))
        return selected_categories

    def fit(
            self,
            data: pd.Series
    ) -> None:
        selected_categories = self._find_common_categories(data)
        all_categories = selected_categories + [self.unknown_token]
        self.category_to_index = {cat: idx for idx, cat in enumerate(all_categories)}
        self.index_to_category = {idx: cat for cat, idx in self.category_to_index.items()}

    def transform(
            self,
            data: pd.Series
    ) -> pd.Series:
        return data.apply(lambda x: self.category_to_index.get(x, self.category_to_index[self.unknown_token])).values


class Preprocessor:
    def __init__(self,
                 schema: Dict,
                 ordered_categories: Dict = None,
                 verbose: bool = False,
                 ):
        self.schema = schema
        self.preprocessor = {name: StandardScalerNAN() if attribute_type == Data.NUMERIC
                                   else OrdinalEncoder(handle_unknown='use_encoded_value', categories=ordered_categories[name]) if attribute_type == Data.CATEGORICAL_ORD
                                   else LabelEncoderUnseenCategories() if attribute_type == Data.CATEGORICAL
                                   else None if attribute_type == 'already_preprocessed'
                                   else raiser()
                             for name, attribute_type in schema.items()}
        self.verbose = verbose


    def fit(self, data: pd.DataFrame) -> None:
        for name, preproc in tqdm(self.preprocessor.items(), desc="Fitting preprocessor", disable=not self.verbose):
            if preproc is None:
                continue
            preproc.fit(data[name])

    def transform(self, data: pd.DataFrame) -> Dict:
        d = {}
        for name, preproc in tqdm(self.preprocessor.items(), desc="Transforming preprocessor", disable=not self.verbose):
            if preproc is None:
                continue
            d[name] = preproc.transform(data[name])
        return d
