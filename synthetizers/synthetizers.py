import pandas as pd
import numpy as np
from typing import List
from privacy_metrics.preprocessor import Data

def leaky_synthesize(release_df: pd.DataFrame, train_df: pd.DataFrame, leak_frac: float) -> pd.DataFrame:
    """Create a mock synthetic dataset consisting of records from the independent dataframe release_df where a
       leak_frac percentage has been replaced with records from train_df."""

    if leak_frac < 0 or leak_frac > 1:
        raise ValueError("The value of the parameter 'leak_frac' must be between 0 and 1")

    synth_size = release_df.shape[0]
    leak_records = int(synth_size * leak_frac)
    if leak_frac == 1.:
        synth_df = train_df
    else:
        synth_df = pd.concat([train_df.sample(leak_records), release_df.sample(synth_size - leak_records)])

    return synth_df

def column_noiser(x: pd.Series, numeric_noise_level: float | None = None, integer_noise_level: float | None = None,
                  cat_switch_prob: float | None = None, switch_categories: pd.Series | None = None,
                  cat_relative_frequencies: List[float] | None = None) -> np.array:
    if (numeric_noise_level is not None) and (cat_switch_prob is not None) and (integer_noise_level is not None):
        raise ValueError("Can't specify all 'num_noise_level', 'cat_switch_probs' and 'integer_noise_level'.")
    if ((numeric_noise_level is not None) and (cat_switch_prob is not None)) or \
            ((integer_noise_level is not None) and (cat_switch_prob is not None)) or \
            ((numeric_noise_level is not None) and (integer_noise_level is not None)):
        raise ValueError("Can't specify two types of noise at the same time.")
    if (numeric_noise_level is None) and (cat_switch_prob is None) and (integer_noise_level is None):
        raise ValueError("One between 'num_noise_level','integer_noise_level' and "
                         "'cat_switch_probs' must be specified.")
    if integer_noise_level is not None:
        if integer_noise_level < 0:
            raise ValueError("The value of 'integer_noise_level' must be greater or equal to 0.")
        x += np.multiply(np.random.choice([-1, 1], size=x.shape), np.random.poisson(integer_noise_level, size=x.shape))
    if numeric_noise_level is not None:
        x += x * np.random.normal(scale=numeric_noise_level, size=x.shape)
    elif cat_switch_prob is not None:
        if switch_categories is None:
            raise ValueError("Please specify the categories to sample from with the argument 'switch_categories'.")
        if cat_relative_frequencies is None:
            raise ValueError("Please specify the relative frequency of the categories to sample from with the argument"
                             " 'cat_relative_frequencies'.")

        if (cat_switch_prob > 1) or (cat_switch_prob < 0):
            raise ValueError("The value of 'cat_switch_prob' must be between 0 and 1")
        random_col = np.random.choice(switch_categories, size=x.shape, p=cat_relative_frequencies)
        x = np.where(np.random.random(size=x.shape) < cat_switch_prob, random_col, x)

    return x


def noisy_leak_synthesize(release_df: pd.DataFrame, train_df: pd.DataFrame, leak_frac: float,
                          col_to_noise: dict[str, str], numeric_noise_level: float = 1e-2,
                          cat_switch_prob: float = 1e-2, integer_noise_level: float = 1e-2,
                          ) -> pd.DataFrame:
    """Create a mock synthetic dataset consisting of records from the independent dataframe release_df where a
       leak_frac percentage has been replaced with noised records from train_df. Noise is added as an independent
       gaussian for numerical columns and as a probability to random switch category for categorical columns."""

    if leak_frac < 0 or leak_frac > 1:
        raise ValueError("The value of the parameter 'leak_frac' must be between 0 and 1")

    if leak_frac == 0.:
        return release_df

    synth_size = train_df.shape[0]
    leak_size = int(synth_size * leak_frac)
    leak_records = train_df.sample(leak_size)
    for col, col_type in col_to_noise.items():
        if col_type == Data.CATEGORICAL:
            categories = leak_records[col].unique()
            val_counts = leak_records[col].value_counts(dropna=False)
            switch_frequencies = [val_counts.loc[c] / len(leak_records) for c in categories]
            leak_records[col] = column_noiser(leak_records[col], cat_switch_prob=cat_switch_prob,
                                              switch_categories=categories, cat_relative_frequencies=switch_frequencies)
        elif col_type == Data.NUMERIC:
            leak_records[col] = column_noiser(leak_records[col], numeric_noise_level=numeric_noise_level)
        elif col_type == Data.INTEGER:
            leak_records[col] = column_noiser(leak_records[col], integer_noise_level=integer_noise_level)
    if leak_frac == 1.:
        synth_df = leak_records
    else:
        synth_df = pd.concat([leak_records, release_df.sample(synth_size - leak_size)])
    return synth_df