import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

from anonymeter.evaluators import LinkabilityEvaluator, SinglingOutEvaluator, InferenceEvaluator
from tqdm import tqdm
import itertools
from sklearn.preprocessing import minmax_scale, LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor, XGBClassifier

from .utils import risk_fn

from .utils import replace_missing_values, weap, pad_target_keys_array
from .utils import normalized_rmse

from .utils import compute_values_below_alpha_percentile
from .preprocessor import Preprocessor, Data
from .indexes_neighbors_search import NearestNeighbor_faiss, NearestNeighbor_sklearn

from .utils import transform_mia, transform_points, compute_dxy
from sklearn.metrics import auc

from domias.bnaf.density_estimation import compute_log_p_x, density_estimator_trainer
import torch
from sklearn.model_selection import train_test_split

class PrivacyMetrics:
    """Privacy metrics class"""

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


    # IMS
    @staticmethod
    def __ims(
            train: pd.DataFrame,
            synth: pd.DataFrame,
    ) -> float:
        """ Internal method to compute identical match sharing

        Args:
            train(DataFrame): The training dataset.
            synth(DataFrame): The synthetic dataset.

        Returns:
            float: The identical match sharing metric.
        """
        return len(pd.merge(train, synth, how='inner')) / len(synth)


    def __compute_confidence_interval_identical_match_sharing(
            self,
            confidence_level: float = 0.95,
            iterations: int = 1000
    ) -> float:
        """Computes the confidence interval for identical match sharing via bootstrapping.

        Args:
            confidence_level (float): The confidence level of the confidence interval.
            iterations (int): The number of iterations to compute the confidence interval.

        Returns:
            float: Radius of the confidence interval.
        """

        train_samples = [self.train.sample(n=len(self.train), replace=True, random_state=i) for i in range(iterations)]
        synth_samples = [self.synthetic.sample(n=len(self.train), replace=True, random_state=i) for i in
                         range(iterations)]
        ims_scores = [self.__ims(t, s) for t, s in
                      tqdm(zip(train_samples, synth_samples), desc='Computing confidence interval')]

        lower_confidence_level = (1 - confidence_level) / 2
        upper_confidence_level = 1 - lower_confidence_level

        upper_bound = np.percentile(ims_scores, upper_confidence_level)
        lower_bound = np.percentile(ims_scores, lower_confidence_level)
        return (upper_bound - lower_bound) / 2


    def identical_match_sharing(
            self
    ) -> (float, float):
        """Computes the identical match sharing metric between two dataframes, e.g. the
            portion of rows in the synthetic dataset that are copies of rows in the train.

        Returns:
            float: The identical match sharing metric.
        """
        score = self.__ims(self.train, self.synthetic)
        error = self.__compute_confidence_interval_identical_match_sharing()
        return score, error

    # DCR
    @staticmethod
    def __compute_confidence_interval_distance_to_closest_record(
            rrd: np.ndarray,
            srd: np.ndarray,
            alpha: int,
            confidence_level: float = 0.95,
            iterations: int = 1000
    ) -> float:
        """ Computes the confidence interval for distance to the closest record via bootstrapping.
        Args:
            rrd (np.ndarray): The real-to-real distance distribution.
            srd (np.ndarray): The synthetic-to-real distance distribution.
            alpha (int): The percentile to use on the RRD.
            confidence_level (float): The confidence level of the confidence interval.
            iterations (int): The number of iterations to compute the confidence interval via bootstrapping.

        Returns:
            float: radius of the confidence interval.
        """
        samples_srd = [np.random.choice(srd, size=len(srd), replace=True) for _ in range(iterations)]
        samples_rrd = [np.random.choice(rrd, size=len(rrd), replace=True) for _ in range(iterations)]

        risk_values = [compute_values_below_alpha_percentile(rr, sr, alpha) for sr, rr in
                       tqdm(zip(samples_srd, samples_rrd), desc='Computing confidence interval')]

        lower_confidence_level = (1 - confidence_level) / 2
        upper_confidence_level = 1 - lower_confidence_level

        upper_bound = np.percentile(risk_values, upper_confidence_level)
        lower_bound = np.percentile(risk_values, lower_confidence_level)
        return (upper_bound - lower_bound) / 2


    def distance_to_closest_record(
            self,
            k: int = 1,
            alpha: int = 2,
            plot: bool = False,
            faiss: bool = False,
    ) -> (np.ndarray, np.ndarray):
        """ Computes the distance to the closest record for a pair of datasets, e.g. the portion of synthetic records
            with a smaller distance to real records than a fixed percentile of the Real-to-Real distance distribution.

        Args:
            k: Number of nearest neighbors to consider. If k=1 we are computing common DCR.
            alpha (int): The percentile to use on the RRD, check how many synthetic records have a smaller SRD.
            plot (bool): Whether to plot SRD vs RRD.
            faiss (bool): Whether to use faiss index or not (sklearn otherwise).

        Returns:
            tuple(float): score and confidence interval for DCR.
        """
        preproc = Preprocessor(self.schema)
        index = NearestNeighbor_faiss(k=k) if faiss else NearestNeighbor_sklearn(k=k)

        print('fitting')
        preproc.fit(self.train)
        print('transforming')
        preproc_real = preproc.transform(self.train)
        preproc_synth = preproc.transform(self.synthetic)

        preproc_real = np.concatenate([arr.reshape(-1, 1) for arr in preproc_real.values()], axis=1)
        preproc_synth = np.concatenate([arr.reshape(-1, 1) for arr in preproc_synth.values()], axis=1)

        print('Computing Index')
        nearest_neighbor_data = index.compute_nearest_neighbors(preproc_real, preproc_synth)

        srd = nearest_neighbor_data.d_ratio
        rrd = nearest_neighbor_data.dx_ratio

        print('Computing Scores')
        score = compute_values_below_alpha_percentile(rrd, srd, alpha)
        error = self.__compute_confidence_interval_distance_to_closest_record(rrd, srd, alpha)

        if plot:
            plt.hist(srd, alpha=0.5, bins=np.linspace(0, 2, 40), density=True, label='SRD')
            plt.hist(rrd, alpha=0.5, bins=np.linspace(0, 2, 40), density=True, label='RRD')
            plt.legend()
            plt.show()

        return score, error


    # MIA
    def __mia(
            self,
            train: pd.DataFrame,
            synth: pd.DataFrame,
            control: pd.DataFrame,
            interval: int,
            factor: float,
            rows: int | None = None,
            plot: bool = False,
    ) -> (float, float):
        """Internals to compute the no-box Membership Inference Attack

        Args:
            train (DataFrame): The training dataset.
            synth (DataFrame): The synthetic dataset.
            control (DataFrame): The control dataset, unseen rows.
            interval (int): How many points we want in the threshold array.
            factor (float): the power used to recompute the thresholds.
            rows (int): number of rows to be in the training ang control dataset. If None the dataset is split in half.
            plot (bool): Whether to plot True-Positive, False-Positive, True-Negative and False-Negative.

        Returns:
            (float, float): AUC score for positive and negative values
        """

        real_records = pd.concat([train.sample(n=rows, random_state=42), control.sample(n=rows, random_state=44)])
        correct_classification = [1] * rows + [0] * rows

        # computing distance between the two sets of records
        distance = compute_dxy(self.schema, real_records, synth.sample(n=len(real_records), random_state=42))

        # Threshold values needed to classify a record as a training one or not
        thresholds = np.linspace(0, np.max(distance), interval)

        # transform the thresholds to have not uniformly separated records
        thresholds = transform_points(thresholds, factor)
        tpr_list = []
        fpr_list = []
        tnr_list = []
        fnr_list = []

        # For each threshold compute the true/false positive/negative
        for th in thresholds:
            predictions = [1 if d < th else 0 for d in distance]
            tp = sum(1 for x, y in zip(correct_classification, predictions) if x == y == 1)
            fp = sum(1 for x, y in zip(correct_classification, predictions) if x == 0 and y == 1)
            tpr_list.append(tp / rows)
            fpr_list.append(fp / rows)

            tn = sum(1 for x, y in zip(correct_classification, predictions) if x == y == 0)
            fn = sum(1 for x, y in zip(correct_classification, predictions) if x == 1 and y == 0)
            tnr_list.append(tn / rows)
            fnr_list.append(fn / rows)

        # compute AUC for positive and negative cases
        auc_p = auc(fpr_list, tpr_list)
        auc_n = auc(fnr_list, tnr_list)

        if plot:
            plt.plot(fpr_list, tpr_list)
            plt.plot(fnr_list, tnr_list)
            plt.title(f"ROC CURVE, AUC={auc_p}")
            plt.show()

        # Get the results in [0, 1]
        auc_p = transform_mia(auc_p)
        auc_n = transform_mia(auc_n)
        return auc_p, auc_n

    def __compute_confidence_interval_no_box_membership_inference_attack(
            self,
            interval: int,
            factor: float,
            rows: int,
            confidence_level: float = 0.95,
            iterations=1000
    ) -> float:
        """ Computes the confidence interval for distance to the closest record via bootstrapping.

        Args:
            interval (int): How many points we want in the threshold array.
            factor (float): the power used to recompute the thresholds.
            rows (int): number of rows to be in the training ang control dataset. If None the dataset is split in half.
            confidence_level (float): The confidence level of the confidence interval.
            iterations (int): The number of iterations to compute the confidence interval via bootstrapping.

        Returns:
            float: radius of teh confidence interval.
        """

        train_samples = [self.train.sample(n=len(self.train), replace=True, random_state=i) for i in range(iterations)]
        synth_samples = [self.synthetic.sample(n=len(self.synthetic), replace=True, random_state=i) for i in
                         range(iterations)]

        mia_scores = [
            self.__mia(t, s, self.control, interval=interval, rows=rows, factor=factor)[0]
            for t, s in tqdm(zip(train_samples, synth_samples), desc='Computing confidence interval')
        ]

        lower_confidence_level = (1 - confidence_level) / 2
        upper_confidence_level = 1 - lower_confidence_level

        upper_bound = np.percentile(mia_scores, upper_confidence_level)
        lower_bound = np.percentile(mia_scores, lower_confidence_level)
        return (upper_bound - lower_bound) / 2

    def no_box_membership_inference_attack(
            self,
            interval: int = 100,
            factor: float = 1.0,
            rows: int | None = None,
    ) -> (float, float):
        """Computes no box membership inference attack.

        Args:
            interval (int): How many points we want in the threshold array.
            factor (float): the power used to recompute the thresholds.
            rows (int): number of rows to be in the training ang control dataset. If None the dataset is split in half.

        Returns:
            (float, float): score and confidence interval for MIA
        """
        if rows is None:
            rows = int(len(self.train) / 2)
        risk = self.__mia(self.train, self.synthetic, self.control, interval, factor, rows)[0]
        ci = self.__compute_confidence_interval_no_box_membership_inference_attack(interval, factor, rows)
        return risk, ci

    # ML Inference
    def __inference_with_ML(
            self,
            train: pd.DataFrame,
            test: pd.DataFrame,
            target: str,
    ) -> float:
        """Internal computation for Machine Learning Inference.

        Args:
            train (pd.Dataframe): dataset used to train the model, usually is the synthetic data
            test (pd.Dataframe): dataset used to test the model, usually the train data
            target (str): name of the target column

        Returns:
            float: Accuracy of the inference attack.
        """
        train_y = train[target]
        train_x = train.drop(target, axis=1, inplace=False)
        test_y = test[target]
        test_x = test.drop(target, axis=1, inplace=False)
        train_x.drop('index', axis=1, inplace=True) if 'index' in train_x.columns else 0
        test_x.drop('index', axis=1, inplace=True) if 'index' in test_x.columns else 0

        if self.schema[target] == Data.CATEGORICAL:
            model = XGBClassifier()
            acc_score_fn = accuracy_score
        elif self.schema[target] == Data.NUMERIC:
            model = XGBRegressor()
            acc_score_fn = normalized_rmse
        else:
            print('Target variable must be "category" or "numeric"')
            return 0

        model.fit(train_x.values, train_y)
        pred = model.predict(test_x.values)
        return acc_score_fn(test_y, pred)

    def __compute_confidence_interval_inference_with_ML(
            self,
            target: str,
            confidence_level: float = 0.95,
            iterations=1000
    ) -> float:
        """Computes the confidence interval for distance to the closest record via bootstrapping.

        Args:
            target (str): name of the target column
            confidence_level (float): The confidence level of the confidence interval.
            iterations (int): The number of iterations to compute the confidence interval via bootstrapping.

        Returns:
            float: radius of teh confidence interval.
        """
        train_samples = [self.train.sample(n=len(self.train), replace=True, random_state=i) for i in range(iterations)]
        synth_samples = [self.synthetic.sample(n=len(self.synthetic), replace=True, random_state=i) for i in
                         range(iterations)]

        ml_scores = [self.__inference_with_ML(t, s, target) for t, s in
                     tqdm(zip(train_samples, synth_samples), desc='Computing confidence interval')]

        lower_confidence_level = (1 - confidence_level) / 2
        upper_confidence_level = 1 - lower_confidence_level

        upper_bound = np.percentile(ml_scores, upper_confidence_level)
        lower_bound = np.percentile(ml_scores, lower_confidence_level)
        return (upper_bound - lower_bound) / 2


    def machine_learning_inference(
            self,
            target: str,
            confidence_level: float = 0.95,
            iterations: int = 1000
    ) -> (float, float):
        """ Computes inference attack using machine learning.

        Args:
            target (str): Target column name.
            confidence_level (float): The confidence level of the confidence interval.
            iterations (int): The number of iterations to compute the confidence interval via bootstrapping.

        Returns:
            (float, float): score and confidence interval for Machine Learning Inference

        """
        for c in self.train.columns:
            if self.schema[c] == Data.CATEGORICAL:
                le = LabelEncoder()
                self.train[c] = le.fit_transform(self.train[c]).astype(int)
                self.control[c] = le.fit_transform(self.control[c]).astype(int)
                self.synthetic[c] = le.fit_transform(self.synthetic[c]).astype(int)
        self.train.dropna(inplace=True)
        self.control.dropna(inplace=True)
        self.synthetic.dropna(inplace=True)
        risk_train = self.__inference_with_ML(self.synthetic, self.train, target)
        risk_control = self.__inference_with_ML(self.synthetic, self.control, target)
        risk = (risk_train - risk_control) / (1 - risk_control)
        ci = self.__compute_confidence_interval_inference_with_ML(target, confidence_level, iterations)
        return risk, ci

    # GTCAP

    @staticmethod
    def __gtcap(
            train: pd.DataFrame,
            synth: pd.DataFrame,
            schema: Dict,
            keys: list,
            target: str,
            radius: float,
            numeric_cols: list,
            minmax: bool = True
    ) -> float:
        """ Internal method to compute GTCAP score

        Args:
            train (pd.Dataframe): training dataset.
            synth (pd.DataFrame): synthetic dataset.
            schema (dict): The schema of the dataset (dtypes).
            keys (list): The attributes of the dataset.
            target (str): Name of the target column.
            radius (float): Radius to compute equivalence between numeric variables.
            numeric_cols (list): List of numeric column names.
            minmax (bool): If true, compute min_max scaler.

        Returns:
            float: GTCAP score.
        """
        k_t = keys + [target]

        train = train[k_t]
        synth = synth[k_t]

        # Using a scaler is mandatory to make the "replace_missing_values" work
        for col in numeric_cols:
            if minmax:
                train[col] = minmax_scale(train[col])
            synth[col] = minmax_scale(synth[col])

        train = replace_missing_values(train, schema)
        synth = replace_missing_values(synth, schema)

        cat_k = list(set(keys) - set(numeric_cols))
        cat_t_k = list(set(k_t) - set(numeric_cols))
        num_k = list(set(keys) - set(cat_k))

        numeric_cols_x = [x + '_x' for x in numeric_cols]
        numeric_cols_y = [x + '_y' for x in numeric_cols]
        numeric_k_x = [x + '_x' for x in num_k]
        numeric_k_y = [x + '_y' for x in num_k]

        gtcap_score = 0.0

        if len(numeric_cols) < 1:  # There are numeric columns?
            synth = weap(synth, keys, target, radius, numeric_cols)
            # print('Length after weap only categorical', len(synth))
            sum_t_k = 0.0
            sum_k = 0.0
            for i, r in synth.iterrows():  # Compute equivalence class as is done in the weap
                t_k_eq = pd.merge(r.to_frame().T, train, how='inner', on=cat_t_k)
                k_eq = pd.merge(r.to_frame().T, train, how='inner', on=cat_k)
                if len(k_eq):
                    sum_t_k += len(t_k_eq)
                    sum_k += len(k_eq)
            gtcap_score = sum_t_k / sum_k if sum_k != 0 else 0
        else:
            synth, rows, idxs = weap(synth, keys, target, radius, numeric_cols)
            # print('Length after weap', len(synth))
            sum_scores = 0.0
            # sum_t_k = 0.0
            for i in idxs:
                row = rows[i]
                row = row.to_frame().T.astype(train.dtypes)

                t_k_eq = pd.merge(row, train, how='inner', on=cat_t_k)
                k_eq = pd.merge(row, train, how='inner', on=cat_k)

                # Compute if the distance between 2 numeric attributes is within the specified radius
                diff_t_k = np.abs(t_k_eq[numeric_cols_x].values - t_k_eq[numeric_cols_y].values) / radius
                score_t_k = np.mean([np.maximum(0, 1 - r) for r in diff_t_k], axis=1) if len(diff_t_k) > 0 else 0
                if target in numeric_cols and len(numeric_cols) == 1:  # Only the target is a numeric attribute
                    score_k = np.ones(len(k_eq))
                else:
                    diff_k = np.abs(k_eq[numeric_k_x].values - k_eq[numeric_k_y].values) / radius
                    score_k = np.mean([np.maximum(0, 1 - r) for r in diff_k], axis=1) if len(diff_k) > 0 else 0

                # Ensure consistency between the frames
                score_t_k = pad_target_keys_array(t_k_eq, k_eq, score_t_k)
                numerator = score_t_k * score_k

                numerator = np.sum(numerator)
                denominator = np.sum(score_k)

                if denominator:
                    sum_scores += numerator / denominator

            gtcap_score = sum_scores / len(idxs) if len(idxs) != 0 else 0

        return gtcap_score

    def __compute_confidence_interval_gtcap(
            self,
            keys: list,
            target: str,
            radius: float,
            numeric_cols: list,
            confidence_level: float = 0.95,
            iterations=1000,
            minmax: bool = True
    ) -> float:
        """Computes the confidence interval for distance to the closest record via bootstrapping.

        Args:
            keys (list): The attributes of the dataset.
            target (str): Name of the target column.
            radius (float): Radius to compute equivalence between numeric variables.
            numeric_cols (list): List of numeric column names.
            minmax (bool): If true, compute min_max scaler.

        Returns:
            float: radius of the confidence interval.
        """
        train_samples = [self.train.sample(n=len(self.train), replace=True, random_state=i) for i in range(iterations)]
        synth_samples = [self.synthetic.sample(n=len(self.synthetic), replace=True, random_state=i) for i in
                         range(iterations)]

        mia_scores = [self.__gtcap(t, s, self.schema, keys, target, radius, numeric_cols, minmax) for t, s in
                      tqdm(zip(train_samples, synth_samples), desc='Computing confidence interval')]

        lower_confidence_level = (1 - confidence_level) / 2
        upper_confidence_level = 1 - lower_confidence_level

        upper_bound = np.percentile(mia_scores, upper_confidence_level)
        lower_bound = np.percentile(mia_scores, lower_confidence_level)
        return (upper_bound - lower_bound) / 2

    def gtcap(
            self,
            keys: list,
            target: str,
            radius: float,
            numeric_cols: list,
            minmax: bool = True,
    ) -> (float, float):
        """ Internal method to compute GTCAP score

        Args:
            keys (list): The attributes of the dataset.
            target (str): Name of the target column.
            radius (float): Radius to compute equivalence between numeric variables.
            numeric_cols (list): List of numeric column names.
            minmax (bool): If true, compute min_max scaler.

        Returns:
            float: GTCAP score.
        """
        risk_train = self.__gtcap(self.train, self.synthetic, self.schema, keys, target, radius, numeric_cols, minmax)
        risk_control = self.__gtcap(self.control, self.synthetic, self.schema, keys, target, radius, numeric_cols,
                                   minmax)
        risk = risk_fn(risk_train, risk_control)
        ci = self.__compute_confidence_interval_gtcap(keys, target, radius, numeric_cols)
        return risk, ci

    # Anonymeter metrics
    def singling_out_risk(
            self,
            na: int,
            max_multivariate_cols: None | int = None,
            min_multivariate_cols: None | int = None,
            univariate_mode: bool = True,
            confidence_level: float = 0.95,
            keep_unreliable: bool = False
    ) -> dict:
        """Calculates Singling Out risk for univariate and multivariate estimations and returns the highest value.

        Args:
            na: int. Number of attacks.
            max_multivariate_cols: int. Maximum number of columns to use when building multivariate features
                                   to single out records.
            min_multivariate_cols: int. Minimum number of columns to use when building multivariate features
                                   to single out records.
            confidence_level: float. Confidence level of the risk estimation.
            keep_unreliable: bool. Whether to include attacks whose rate is smaller than the baseline model.

        Returns:
            Tuple: The maximum risk computed, its confidence interval, the number of features used in that attack.
        """
        singling_out_risks = {'value': [], 'ci': [], 'train_score': [], 'control_score': [], 'baseline_score': []}

        if univariate_mode:
            univariate_evaluator = SinglingOutEvaluator(ori=self.train, syn=self.synthetic, control=self.control,
                                                        n_attacks=na)

            univariate_evaluator.evaluate(mode='univariate')
            results = univariate_evaluator.results()

            if keep_unreliable or (results.attack_rate.value > results.baseline_rate.value):
                singling_out_risks['value'].append(univariate_evaluator.risk(confidence_level=confidence_level).value)
                singling_out_risks['ci'].append(univariate_evaluator.risk(confidence_level=confidence_level).ci)
                singling_out_risks['train_score'].append(results.attack_rate.value)
                singling_out_risks['control_score'].append(results.control_rate.value)
                singling_out_risks['baseline_score'].append(results.baseline_rate.value)

        if max_multivariate_cols is None:
            max_cols = self.train.shape[1]
        else:
            max_cols = min(max_multivariate_cols, self.train.shape[1])

        if min_multivariate_cols is None:
            min_cols = 1
        else:
            min_cols = max(1, min_multivariate_cols)

        for n_attributes in tqdm(range(min_cols, max_cols + 1)):
            multivariate_evaluator = SinglingOutEvaluator(ori=self.train, syn=self.synthetic, control=self.control,
                                                          n_attacks=na,
                                                          n_cols=n_attributes)
            multivariate_evaluator.evaluate(mode='multivariate')
            results = multivariate_evaluator.results()
            if keep_unreliable or (results.attack_rate.value > results.baseline_rate.value):
                singling_out_risks['value'].append(multivariate_evaluator.risk(confidence_level=confidence_level).value)
                singling_out_risks['ci'].append(multivariate_evaluator.risk(confidence_level=confidence_level).ci)
                singling_out_risks['train_score'].append(results.attack_rate.value)
                singling_out_risks['control_score'].append(results.control_rate.value)
                singling_out_risks['baseline_score'].append(results.baseline_rate.value)

        if len(singling_out_risks['value']) > 0:
            max_index = np.argmax(singling_out_risks['value'])
            mean = np.mean(singling_out_risks['value'])

            results = {key: val[max_index] for key, val in singling_out_risks.items()}
            results['mean_value'] = mean
            results['max_num_columns'] = int(max_index + 1 + min_multivariate_cols)

            return results

        else:

            return {}

    def linkability_risk(
            self,
            na: int,
            n_neighbors: int = 1,
            ordered_sample: bool = True,
            confidence_level: float = 0.95,
            keep_unreliable=False
    ) -> dict:
        """Calculates Linkability risk for different columns combinations and returns the highest value.

        Args:
            na: int. Number of attacks.
            n_neighbors: int. Number of nearest neighbors to be considered when computing linkability risk.
            ordered_sample: bool. How to choose the two subsets of columns to compare. When True it
                               will split taking the first columns as one subset and the last one as the othe. When False
                                it  will consider all random splits.
            confidence_level: float. Confidence level of the risk estimation.
            keep_unreliable: bool. Whether to include attacks whose rate is smaller than the baseline model.


        Returns:
            Tuple:The maximum risk computed, its confidence interval, a list containing the column split used in the attack.
        """
        cols = list(self.train.columns)
        linking_risks = {'value': [], 'ci': [], 'col_splits': [], 'train_scores': [], 'control_scores': [],
                         'baseline_score': []}
        if ordered_sample:
            for i in range(2, len(cols) - 2):  # Splits with only one column will be equivalent to inference attacks
                aux_cols = [cols[:i], cols[i:]]
                evaluator = LinkabilityEvaluator(ori=self.train,
                                                 syn=self.synthetic,
                                                 control=self.control,
                                                 n_attacks=na,
                                                 aux_cols=aux_cols,
                                                 n_neighbors=n_neighbors)
                evaluator.evaluate(n_jobs=-2)
                results = evaluator.results()
                if keep_unreliable or (results.attack_rate.value > results.baseline_rate.value):
                    linking_risks['value'].append(evaluator.risk(confidence_level=confidence_level).value)
                    linking_risks['ci'].append(evaluator.risk(confidence_level=confidence_level).ci)
                    linking_risks['col_splits'].append(aux_cols)
                    linking_risks['train_scores'].append(results.attack_rate.value)
                    linking_risks['control_scores'].append(results.control_rate.value)
                    linking_risks['baseline_score'].append(results.baseline_rate.value)
        else:
            # cycling over all combinations lead to repetitions when sampling i or len(cols)-i elements
            for i in range(2, int(len(cols) // 2) + 1):
                for split in itertools.combinations(cols, i):
                    remaining = list(set(cols) - set(split))
                    aux_cols = [split, remaining]
                    evaluator = LinkabilityEvaluator(ori=self.train,
                                                     syn=self.synthetic,
                                                     control=self.control,
                                                     n_attacks=na,
                                                     aux_cols=aux_cols,
                                                     n_neighbors=n_neighbors)
                    evaluator.evaluate(n_jobs=-2)
                    results = evaluator.results()
                    if keep_unreliable or (results.attack_rate.value > results.baseline_rate.value):
                        linking_risks['value'].append(evaluator.risk(confidence_level=confidence_level).value)
                        linking_risks['ci'].append(evaluator.risk(confidence_level=confidence_level).ci)
                        linking_risks['col_splits'].append(aux_cols)
                        linking_risks['train_scores'].append(results.attack_rate.value)
                        linking_risks['control_scores'].append(results.control_rate.value)
                        linking_risks['baseline_score'].append(results.baseline_rate.value)

        if len(linking_risks['value']) > 0:
            max_index = np.argmax(linking_risks['value'])
            mean = np.mean(linking_risks['value'])

            return {'max_value': linking_risks['value'][max_index],
                    'max_ci': linking_risks['ci'][max_index],
                    'mean_value': mean,
                    'col_split': linking_risks['col_splits'][max_index],
                    'train_score': linking_risks['train_scores'][max_index],
                    'cntrol_score': linking_risks['control_scores'][max_index],
                    'baseline_score': linking_risks['baseline_score'][max_index]}

    def inference_risk(
            self,
            na: int,
            ordered_columns: bool = True,
            min_col_number: int = 1,
            confidence_level: float = 0.95,
            keep_unreliable: bool = False
    ) -> dict:
        """Calculates Linkability risk for different columns combinations and returns the highest value.

        Args:
            na: int. Number of attacks.
            ordered_columns: bool. How to choose the auxiliary columns. When True, for a fixed number of columns n
                             the first n columns will be used as auxiliary information. If False, all possible
                             combinations of n columns will be considered.
            min_col_number: int. Minimum number of auxiliary columns to be used.
            confidence_level: float. Confidence level of the risk estimation.
            keep_unreliable: bool. Whether to include attacks whose rate is smaller than the baseline model.


        Returns:
            dict: dictionary containing the risk for every column
        """
        cols = self.train.columns.tolist()
        inference_risks = {'column': [], 'max_value': [], 'max_ci': [], 'mean_value': [],
                           'train_score': [], 'control_score': [], 'baseline_score': []}

        for target in cols:
            input_cols = list(set(cols) - {target})
            risks = []
            cis = []

            control_scores = []  # JUST FOR DEBUGGING
            train_scores = []
            baseline_rate = []

            for r in range(min_col_number, len(input_cols) - 1):
                if ordered_columns:
                    aux_cols = input_cols[:r]
                    evaluator = InferenceEvaluator(ori=self.train,
                                                   syn=self.synthetic,
                                                   control=self.control,
                                                   aux_cols=aux_cols,
                                                   secret=target,
                                                   n_attacks=na)
                    evaluator.evaluate(n_jobs=-2)
                    results = evaluator.results()

                    if keep_unreliable or (results.attack_rate.value > results.baseline_rate.value):
                        evaluation = evaluator.risk(confidence_level=confidence_level)
                        risks.append(evaluation.value)
                        cis.append(evaluation.ci)
                        train_scores.append(results.attack_rate.value)
                        control_scores.append(results.control_rate.value)
                        baseline_rate.append(results.baseline_rate.value)


                else:
                    for aux_cols in itertools.combinations(input_cols, r):
                        evaluator = InferenceEvaluator(ori=self.train,
                                                       syn=self.synthetic,
                                                       control=self.control,
                                                       aux_cols=list(aux_cols),
                                                       secret=target,
                                                       n_attacks=na)
                        evaluator.evaluate(n_jobs=-2)
                        results = evaluator.results()

                        if keep_unreliable or (results.attack_rate.value > results.baseline_rate.value):
                            evaluation = evaluator.risk(confidence_level=confidence_level)
                            risks.append(evaluation.value)
                            cis.append(evaluation.ci)
                            train_scores.append(results.attack_rate)
                            control_scores.append(results.control_rate)
                            baseline_rate.append(results.baseline_rate.value)

            if len(risks) > 0:
                max_index = np.argmax(risks)
                inference_risks['max_value'].append(risks[max_index])
                inference_risks['max_ci'].append(cis[max_index])
                inference_risks['mean_value'].append(np.mean(risks))
                inference_risks['column'].append(target)
                inference_risks['train_score'].append(train_scores[max_index])
                inference_risks['control_score'].append(control_scores[max_index])
                inference_risks['baseline_score'].append(baseline_rate[max_index])

            else:
                inference_risks['max_value'].append(None)
                inference_risks['max_ci'].append(None)
                inference_risks['mean_value'].append(None)
                inference_risks['column'].append(target)
                inference_risks['train_score'].append(None)
                inference_risks['control_score'].append(None)

        return inference_risks


    def DOMIAS(
            self,
            device: str
    ) -> (float, float):
        """ DOMIAS proposed in: https://arxiv.org/pdf/2302.12580,
            original code: https://github.com/vanderschaarlab/DOMIAS/tree/main,

        Args:
            device: str. Device to use.

        Returns:
            (float, float): DOMIAS and baseline score.
        """

        train = self.train.copy()
        synthetic = self.synthetic.copy()
        control = self.control.copy()

        for c in self.train.columns:
            if self.schema[c] == Data.CATEGORICAL:
                le = LabelEncoder()
                train[c] = pd.to_numeric(le.fit_transform(train[c]))
                synthetic[c] = pd.to_numeric(le.fit_transform(synthetic[c]))
                control[c] = pd.to_numeric(le.fit_transform(control[c]))

        train_sets = train_test_split(train, test_size=0.5, random_state=42)
        synthetic_sets = train_test_split(synthetic, test_size=0.5, random_state=42)
        control_sets = train_test_split(control, test_size=0.5, random_state=42)

        X_test = pd.concat([train_sets[1], synthetic_sets[1]],
                           ignore_index=True).to_numpy()
        X_baseline = pd.concat([train_sets[1], control_sets[1]],
                           ignore_index=True).to_numpy()

        _, real_model = density_estimator_trainer(train_sets[0].to_numpy())
        real_density = np.exp(
                compute_log_p_x(real_model, torch.as_tensor(X_test).float().to(device))
                .cpu()
                .detach()
                .numpy()
        )
        _, synth_model = density_estimator_trainer(synthetic_sets[0].to_numpy())
        synth_density = np.exp(
            compute_log_p_x(synth_model, torch.as_tensor(X_test).float().to(device))
            .cpu()
            .detach()
            .numpy()
        )

        _, real_model_baseline = density_estimator_trainer(train_sets[0].to_numpy())
        real_density_baseline = np.exp(
            compute_log_p_x(real_model, torch.as_tensor(X_baseline).float().to(device))
            .cpu()
            .detach()
            .numpy()
        )

        _, baseline_model = density_estimator_trainer(control_sets[0].to_numpy())
        baseline_density = np.exp(
            compute_log_p_x(baseline_model, torch.as_tensor(X_baseline).float().to(device))
            .cpu()
            .detach()
            .numpy()
        )

        return synth_density/(real_density + 1e-8), baseline_density/(real_density_baseline + 1e-8)

