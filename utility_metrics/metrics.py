import pandas as pd
from typing import Dict
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

class UtilityMetrics:
    """Utility Metrics class"""

    def __init__(
            self,
            train: pd.DataFrame,
            synthetic: pd.DataFrame,
            control: pd.DataFrame,
            schema: Dict | None = None
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