# Script for embedding creation on Census dataset

from typing import List, Dict
import pandas as pd
import numpy as np
from privacy_eval.semiparametric.processing_embed import Col
import torch
from torch.nn import GELU, LayerNorm, Dropout, ModuleList
from privacy_eval.semiparametric.scores import unified_framework_privacy_risk
from privacy_eval.semiparametric.evaluator_embed import SemiParametricRiskEvaluatorEmbed
from privacy_eval.utils import EarlyStop
from sklearn.neighbors import LocalOutlierFactor

class StupidModelEmbed(torch.nn.Module):
    def __init__(
            self,
            in_shape: int,
            hidden_dims: List[int],
            emb_dim: int,
            dropout_p: float,
            num_embeddings: list,
            embedding_dim: list
    ):
        super().__init__()
        self.first = torch.nn.Linear(in_shape, hidden_dims[0])
        self.embedding_layer = torch.nn.ModuleList([
            torch.nn.Embedding(num_embeddings[i] + 1, embedding_dim[i]) for i in range(len(num_embeddings))
        ])

        hidden = []
        norm = []
        dropout = []
        for i in range(len(hidden_dims) - 1):
            hidden.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            norm.append(LayerNorm(hidden_dims[i + 1]))
            dropout.append(Dropout(dropout_p))
        self.hidden = torch.nn.ModuleList(hidden)
        self.norm = ModuleList(norm)
        self.drop = ModuleList(dropout)
        self.last = torch.nn.Linear(hidden_dims[-1], emb_dim)
        self.activation = GELU()

    def forward(
            self,
            x
    ):
        x_categorical = x['categorical']
        embedded_x_cat = []
        for i, emb_layer in enumerate(self.embedding_layer):
            embedded_x_cat.append(emb_layer(x_categorical[:, i].long()))

        x_categorical = torch.cat(embedded_x_cat, dim=1)
        x_numeric = x['numeric']
        combined = torch.cat([x_numeric, x_categorical], dim=1)
        combined = self.first(combined)
        combined = self.activation(combined)
        for i, l in enumerate(self.hidden):
            combined = l(combined)
            combined = self.drop[i](combined)
            combined = self.norm[i](combined)
            combined = self.activation(combined)
        x = self.last(combined)
        x = torch.nn.functional.normalize(x, dim=1)
        return x

def train_model(train_df: pd.DataFrame, synth_df: pd.DataFrame, control_df: pd.DataFrame,
                col_types: Dict[str, Col], d_emb: int, out_dim: int,
                device: str) -> SemiParametricRiskEvaluatorEmbed:
    torch.manual_seed(11)
    estimator = SemiParametricRiskEvaluatorEmbed(train_df, synth_df, control_df, col_types,
                                                     unified_framework_privacy_risk, d_emb)
    in_dim = estimator.get_features_shape()
    original_dataset = pd.concat([train_df, control_df, synth_df])
    num_embeddings = []
    cat_uniq = []
    for c in col_types:
        if col_types[c] in [Col.CATEGORICAL, Col.CATEGORICAL_ORD]:
            length_category = len(original_dataset[c].unique())
            cat_uniq.append(length_category)
            num_embeddings.append(length_category)
    embedding_dim = [d_emb] * len(num_embeddings)
    model = StupidModelEmbed(in_dim, [1024] * 3, out_dim, 0.1, num_embeddings, embedding_dim)
    early_stop = EarlyStop(window=20, patience=10, min_rel_improvement=0.)
    train_history = estimator.fit_contrastive_embedder(model, batch_size=1024, device=device, num_workers=4,
                                                       n_epochs=300, lr=1e-3, early_stop=early_stop)
    return estimator

def remove_outliers(estimator: SemiParametricRiskEvaluatorEmbed, train_df: pd.DataFrame, device: str,
                    contamination: float) -> pd.DataFrame:

    embs = estimator.embed_data(estimator.train_encoded, col_to_mask=[], batch_size=1024,
                                num_workers=4, device=device)

    outlier_detector = LocalOutlierFactor(contamination=contamination, n_jobs=-1)

    outlier_labels = outlier_detector.fit_predict(embs)
    outliers_indices = np.arange(len(outlier_labels))[outlier_labels == -1]
    df = train_df.drop(index=outliers_indices)
    return df
