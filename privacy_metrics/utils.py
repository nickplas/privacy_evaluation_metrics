import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.metrics import root_mean_squared_error
from .preprocessor import Preprocessor
from .indexes_neighbors_search import NearestNeighbor_sklearn, NearestNeighbor_faiss


def risk_fn(
        risk_train: float,
        risk_control: float
) -> float:
    """Computing risk using the control set as a baseline for the risk

    Args:
        risk_train (float): The risk value for the train set
        risk_control (float): The risk value for the control set

    Returns:
        float: The risk value
    """
    return (risk_train - risk_control) / (1 - risk_control)


def replace_missing_values(
        dataset: pd.DataFrame,
        schema: Dict,
) -> pd.DataFrame:
    """ Replace missing values in the dataset.

    Args:
        dataset (pd.DataFrame): Dataset to be modified.
        schema (Dict): The schema of the dataset (dtypes).

    Returns:
        pd.DataFrame: Modified dataset.
    """
    for col, col_type in schema.items():
        if col_type == 'categorical':
            dataset[col] = dataset[col].fillna('blank')
        if col_type == 'numeric':
            dataset[col] = dataset[col].fillna(-0.1) # dataset must be scale dbefore
    return dataset


def weap(
        synth: pd.DataFrame,
        keys: list,
        target: str,
        radius: float,
        numeric_cols: list
) -> tuple[pd.DataFrame, Dict, List]:
    """ Compute the WEAP.
    Args:
        synth (pd.DataFrame): Synthetic dataset.
        keys (list): List of column names.
        target (str): Target column name.
        radius (float): Radius to compute equivalence between numeric variables.
        numeric_cols (list): List of numeric column names.

    Returns:
        tuple[pd.DataFrame, Dict, List]: Returns a tuple containing: the modified dataset, the original rows
        with their indexes, and the indexes
    """
    k_t = keys + [target]

    # get the only attribute needed
    synth = synth[k_t]

    # subset of categorical and numeric variables with and without the target
    cat_k = list(set(keys) - set(numeric_cols))
    cat_t_k = list(set(k_t) - set(numeric_cols))
    num_k = list(set(keys) - set(cat_k))

    # Containers
    indices = []
    t_k_merge = {} # merge between dataset with target and keys
    k_merge = {} # merge between dataset with only keys
    rows = {}

    # Having duplicates may cause overestimation of the WEAP
    synth = synth.drop_duplicates()

    if len(cat_k) > 0: # do we have categorical attributes in the dataset?
        for i, row in synth.iterrows():
            k_eq = pd.merge(row[keys].to_frame().T, synth, how='inner', on=cat_k) # rows that share keys
            t_k_eq = pd.merge(row.to_frame().T, synth, how='inner', on=cat_t_k) # rows that share Keys + target
            if len(t_k_eq) / len(k_eq) >= 1: # this will not be never >1, we are checking just =1
                indices.append(i) # save original index
                t_k_merge[i] = t_k_eq # save the common t_k rows
                k_merge[i] = k_eq # save the common k rows
                rows[i] = row # save the original row

        # subset the dataset with the previously calculated indexes
        synth = synth.loc[indices]

    if len(numeric_cols) < 1: # do we have numeric attributes in the dataset?
        return synth

    indices_2 = []
    numeric_cols_x = [x + '_x' for x in numeric_cols]
    numeric_cols_y = [x + '_y' for x in numeric_cols]
    numeric_k_x = [x + '_x' for x in num_k]
    numeric_k_y = [x + '_y' for x in num_k]

    for i in indices:
        # row = rows[i]
        k_eq = k_merge[i]
        t_k_eq = t_k_merge[i]

        # count_k = 0.0
        # count_t_k = 0.0

        # How many rows are in the same class of equivalence using a radius for the numeric attributes
        count_t_k = (np.abs(t_k_eq[numeric_cols_x].values - t_k_eq[numeric_cols_y].values) < radius).mean(axis=1).sum()
        if target in numeric_cols and len(numeric_cols) == 1: # if target is the only numeric col
            count_k = len(k_eq)
        else:
            count_k = (np.abs(k_eq[numeric_k_x].values - k_eq[numeric_k_y].values) < radius).mean(axis=1).sum()

        if count_k > 0 and count_t_k / count_k >= 1: # Saving indices of the interested rows
            indices_2.append(i)

    # Subset the dataset with the previously calculated indexes
    synth = synth.loc[indices_2]

    return synth, {i: rows[i] for i in indices_2}, indices_2

def get_original_format_df(
        df: pd.DataFrame
) -> pd.DataFrame:
    """ Drop the addeed columns that ends with '_x'

    Args:
        df (pd.DataFrame): Dataset to be modified.

    Returns:
        pd.DataFrame: Modified dataset.
    """
    cols_to_drop = df.columns[df.columns.str.endswith('_x')]
    df.drop(cols_to_drop, axis=1, inplace=True)
    df.columns = df.columns.str.replace('_y', '')
    return df

def pad_target_keys_array(
        t_k_eq: pd.DataFrame,
        k_eq: pd.DataFrame,
        score_t_k: np.ndarray,
) -> np.ndarray:
    """ Pad the dataset with target and keys

    Args:
        t_k_eq (pd.DataFrame): Target+Keys dataset to be padded.
        k_eq (pd.DataFrame): Only keys dataset.
        score_t_k (np.ndarray): Score for the target+keys dataset.

    Returns:
        np.ndarray: New score for the padded dataset.
    """
    if len(t_k_eq) == len(k_eq): # No need to pad the dataset
        return score_t_k
    else:
        k_eq = get_original_format_df(k_eq)
        t_k_eq = get_original_format_df(t_k_eq)
        indicators = k_eq.isin(t_k_eq.to_dict(orient='list')).all(axis=1).astype(int).to_numpy()
        new_score_t_k = np.zeros(len(k_eq))
        new_score_t_k[indicators == 1] = score_t_k
    return new_score_t_k

# ML Inference Utility
def normalized_rmse(
        real: np.array,
        pred: np.array
) -> float:
    """Computed the normalized RMSE

    Args:
        real (np.ndarray): the real data
        pred (np.ndarray): the predicted data

    Returns:
        float: the normalized RMSE
    """
    range_y = max(real) - min(real)
    return 1 - root_mean_squared_error(real, pred)/range_y

# Index for nearest neighbor



# Compute values under a percentile for srd and rrd
def compute_values_below_alpha_percentile(
        rrd: np.ndarray,
        srd: np.ndarray,
        alpha: int
) -> float:
    """ Compute the normalized portion of synthetic records that have smaller distance to real records than a fixed
        percentile on the Real-toReal distance distribution.

    Args:
        rrd (np.ndarray): Real-to-Real distance distribution.
        srd (np.ndarray): Synthetic-to-Real distance distribution.
        alpha (int): percentile on the Real-toReal distance distribution.

    Returns:
        float: Normalized portion of synthetic records that have smaller distance to real records than the alpha
        percentile on the RRD.
    """
    perc = np.percentile(rrd, alpha)
    Q_SRD = len(srd[srd < perc])
    portion = Q_SRD / len(srd)
    normalized_portion = (portion - alpha / 100) / (1 - alpha / 100)
    return normalized_portion

# mia utils

def transform_points(
        points: List | np.ndarray,
        factor: float
) -> np.ndarray:
    """Transform the points in a vector and normalize them.

    Args:
        points (List | np.array): Points to transform.
        factor (float): Factor to use to compute the power of the points.
    Returns:
        np.array: transformed points.
    """
    transformed_points = np.array(points) ** factor
    transformed_points *= points[-1] / transformed_points[-1]
    return transformed_points

def transform_mia(
        mia_score: float
) -> float:
    """Get the MIA score in the [0.1] range.

    Args:
        mia_score (float): MIA score.

    Returns:
          float: MIA score in the [0.1] range.
    """
    return max(0, (mia_score - 0.5))*2


def compute_dxy(
        schema: Dict,
        real: pd.DataFrame,
        synth: pd.DataFrame,
        faiss: bool = False
) -> List[float] | np.ndarray:
    """Compute the distances between each synthetic and real records.

    Args:
        schema (dict): The schema of the dataset (dtypes).
        real (pd.DataFrame): The real dataset.
        synth (pd.DataFrame): The synthetic dataset.
        faiss (bool): Whether to use faiss index or not (sklearn otherwise).

    Returns:
        List[float] | np.ndarray: The distances between each synthetic and real records.
    """

    knn_index = NearestNeighbor_faiss() if faiss else NearestNeighbor_sklearn()
    preprocessor = Preprocessor(schema)
    preprocessor.fit(real)

    real_transform = preprocessor.transform(real)
    synth_transform = preprocessor.transform(synth)

    real_transform = np.concatenate([arr.reshape(-1, 1) for arr in real_transform.values()], axis=1)
    synth_transform = np.concatenate([arr.reshape(-1, 1) for arr in synth_transform.values()], axis=1)

    knn_results = knn_index.compute_nearest_neighbors(real_transform, synth_transform)
    return knn_results.d_xy
