import pandas as pd
from typing import Dict, List

from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, root_mean_squared_error
from scipy.stats import ks_2samp, chisquare
from sklearn.metrics.pairwise import rbf_kernel

from .utils import categorical_kernel, kernel_score, normalized_polynomial_kernel, normalized_linear_kernel
from privacy_metrics.preprocessor import Data

class UtilityMetrics:
    """Utility Metrics class"""

    def __init__(
            self,
            train: pd.DataFrame,
            synthetic: pd.DataFrame,
            control: pd.DataFrame,
            schema: Dict[str, Data]
    ):
        """Contruct the object

        Args:
            train(DataFrame): The training dataset.
            synthetic(DataFrame): The synthetic dataset.
            control(DataFrame): The control dataset.
            schema(Dict): The schema of thedataset(dtypes).
        """
        self.train = train
        self.synthetic = synthetic
        self.control = control
        self.schema = schema

    def _has_schema(self):
        return True if self.schema is not None else False

    def machine_learning_efficacy(self,
    ) -> float:
        """Train a classifier to discriminate between real and synthetic records

        Returns:
            float: The accuracy of the machine learning model.
        """
        self.train.dropna(inplace=True)
        self.synthetic.dropna(inplace=True)
        self.train['source'] = 0
        self.synthetic['source'] = 1

        t1 = self.train.head(int(len(self.train) / 2))
        t2 = self.train.tail(int(len(self.train) / 2))
        s1 = self.synthetic.head(int(len(self.train) / 2))
        s2 = self.synthetic.tail(int(len(self.train) / 2))

        dataset_train = pd.concat([t1, s1], ignore_index=True).sample(frac=1)
        dataset_test = pd.concat([t2, s2], ignore_index=True).sample(frac=1)
        for c in dataset_train.columns:
            if dataset_train[c].dtype == 'object':
                le = LabelEncoder()
                dataset_train[c] = le.fit_transform(dataset_train[c]).astype(int)
                dataset_test[c] = le.fit_transform(dataset_test[c]).astype(int)

        dataset_train_y = dataset_train['source']
        dataset_train_x = dataset_train.drop('source', axis=1, inplace=False)
        dataset_test_y = dataset_test['source']
        dataset_test_x = dataset_test.drop('source', axis=1, inplace=False)
        dataset_train_x.drop('index', axis=1, inplace=True) if 'index' in dataset_train_x.columns else 0
        dataset_test_x.drop('index', axis=1, inplace=True) if 'index' in dataset_test_x.columns else 0
        model = XGBClassifier()
        acc_score_fn = accuracy_score
        model.fit(dataset_train_x.values, dataset_train_y)
        pred = model.predict(dataset_test_x.values)

        return acc_score_fn(dataset_test_y, pred)

    def ml_utility(
            self,
            target: str,
            aux_cols: List[str] | None = None,
            train_test_procedure: str = 'tstr'
    ) -> (float, float):

        train = self.train.copy()
        synthetic = self.synthetic.copy()
        control = self.control.copy()

        if aux_cols is not None:
            cols = [target] + aux_cols
            train = train[cols]
            synthetic = synthetic[cols]
            control = control[cols]

        for c in train.columns:
            if self.schema[c] == Data.CATEGORICAL:
                le = LabelEncoder()
                train[c] = le.fit_transform(train[c]).astype(int)
                control[c] = le.fit_transform(control[c]).astype(int)
                synthetic[c] = le.fit_transform(synthetic[c]).astype(int)

        if train_test_procedure == 'tstr':
            train_y = synthetic[target]
            train_x = synthetic.drop(target, axis=1, inplace=False)
            test_y = train[target]
            test_x = train.drop(target, axis=1, inplace=False)
        elif train_test_procedure == 'trts':
            train_y =  train[target]
            train_x = train.drop(target, axis=1, inplace=False)
            test_y = synthetic[target]
            test_x =  synthetic.drop(target, axis=1, inplace=False)
        elif train_test_procedure == 'tsts':
            train_y = synthetic[target]
            train_x = synthetic.drop(target, axis=1, inplace=False)
            test_y = synthetic[target]
            test_x = synthetic.drop(target, axis=1, inplace=False)
        elif train_test_procedure == 'trtr':
            train_y = train[target]
            train_x = train.drop(target, axis=1, inplace=False)
            test_y = train[target]
            test_x = train.drop(target, axis=1, inplace=False)
        else:
            raise ValueError('train_test_procedure must be "tstr", "trts", "tsts", or "trtr"')

        test_y_control = control[target]
        test_x_control = control.drop(target, axis=1, inplace=False)

        train_x.drop('index', axis=1, inplace=True) if 'index' in train_x.columns else 0
        test_x.drop('index', axis=1, inplace=True) if 'index' in test_x.columns else 0

        if self.schema[target] == Data.CATEGORICAL:
            model = XGBClassifier()
            score_fn = accuracy_score
        elif self.schema[target] == Data.NUMERIC:
            model = XGBRegressor()
            score_fn = root_mean_squared_error
        else:
            raise ValueError('target variable must be "categorical" or "numeric"')
        model.fit(train_x.values, train_y)
        pred = model.predict(test_x.values)
        pred_control = model.predict(test_x_control)
        return score_fn(test_y, pred), score_fn(test_y_control, pred_control)

    def ks_test(self) -> Dict[str, Dict]:
        """Computes Kolmogorow-Smirnov test statistic between all numeric columns

        Returns:
                Dict: result of the computations.
        """
        res = {}
        for c in self.train.columns:
            if self.schema[c] == 'numeric':
                stat, p_val = ks_2samp(self.train[c], self.synthetic[c])
                res[c] = {'statistic': stat, 'p_value': p_val}
        return res

    def chi_squared_test(self) -> Dict[str, Dict]:
        """Computes Chi Squared test statistic between all categorical columns

        Returns:
                Dict: result of the computations.
        """
        res ={}
        for c in self.train.columns:
            if self.schema[c] == Data.CATEGORICAL:
                train_counts = self.train[c].value_counts()
                synthetic_counts = self.synthetic[c].value_counts()

                print(train_counts)
                print(synthetic_counts)

                stat, p_val = chisquare(train_counts, synthetic_counts)
                res[c] = {'statistic': stat, 'p_value': p_val}
        return res

    def MMD(
            self,
            kernel: str = 'rbf',
            gamma: float = 1.0,
            degree: int = 1,
            coef: float = 0.0
    ) -> float:
        """Computes Maximum Mean Discrepancy between real and synthetic data

        Args:
            kernel(str): The kernel type, default: rbf.
            gamma(float): Kernel parameter, only for polynomial kernel and rbf kernel.
            degree(int): Kernel parameter, only for polynomial kernel.
            coef(float): Kernel parameter, only for polynomial kernel.

            Kernel parameter can be:
                linear: k(x_1, x_2) = x_1^T \cdot x_2
                polynomial: k(x_1, x_2) = (gamma x_1 \cdot x_2 + coef)^(degree)
                rbf: k(x_1, x_2) = exp(-gamma ||x_1 - x_2||^2)

        Returns:
            float: MMD score
        """
        numeric_attributes = [name for name in self.schema.keys() if self.schema[name] == Data.NUMERIC]
        categorical_attributes = [name for name in self.schema.keys() if self.schema[name] == Data.CATEGORICAL]

        train_numeric = self.train[numeric_attributes]
        train_categorical = self.train[categorical_attributes]
        synthetic_numeric = self.synthetic[numeric_attributes]
        synthetic_categorical = self.synthetic[categorical_attributes]

        w_num = len(numeric_attributes)/len(self.train.columns)
        w_cat = len(categorical_attributes)/len(self.train.columns)

        XX_cat = categorical_kernel(train_categorical, train_categorical)
        YY_cat = categorical_kernel(synthetic_categorical, synthetic_categorical)
        XY_cat = categorical_kernel(train_categorical, synthetic_categorical)

        if kernel == 'rbf':
            XX_num = rbf_kernel(train_numeric.to_numpy(), train_numeric.to_numpy(), gamma)
            YY_num = rbf_kernel(synthetic_numeric.to_numpy(), synthetic_numeric.to_numpy(), gamma)
            XY_num  = rbf_kernel(train_numeric.to_numpy(), synthetic_numeric.to_numpy(), gamma)

        elif kernel == 'linear':
            XX_num = normalized_linear_kernel(train_numeric.to_numpy(), train_numeric.to_numpy())
            YY_num = normalized_linear_kernel(synthetic_numeric.to_numpy(), synthetic_numeric.to_numpy())
            XY_num = normalized_linear_kernel(train_numeric.to_numpy(), synthetic_numeric.to_numpy())

        elif kernel == 'polynomial':
            XX_num = normalized_polynomial_kernel(train_numeric.to_numpy(), train_numeric.to_numpy(), degree, gamma, coef)
            YY_num = normalized_polynomial_kernel(synthetic_numeric.to_numpy(), synthetic_numeric.to_numpy(), degree, gamma, coef)
            XY_num = normalized_polynomial_kernel(train_numeric.to_numpy(), synthetic_numeric.to_numpy(), degree, gamma, coef)
        else:
            raise NotImplementedError


        XX = w_num * XX_num + w_cat * XX_cat
        YY = w_num * YY_num + w_cat * YY_cat
        XY = w_num * XY_num + w_cat * XY_cat

        score = kernel_score(XX, YY, XY)

        return score
