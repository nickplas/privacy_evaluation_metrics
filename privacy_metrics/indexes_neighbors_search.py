import numpy as np
import faiss
from typing import NamedTuple
from sklearn.neighbors import NearestNeighbors

class NearestNeighborData(NamedTuple):
    """ Dataclass to contain the information computed by the NearestNeighbor Index"""
    d_xx: np.ndarray
    d_yy: np.ndarray
    d_xy: np.ndarray
    d_ratio: np.ndarray
    dx_ratio: np.ndarray


class NearestNeighbor_faiss:
    def __init__(
            self,
            k: int = 1 # number of considered neighbors
    ):
        self.k = k


    def compute_nearest_neighbors(self,
                                  real: np.array,
                                  synth: np.array,
    ) -> NearestNeighborData:
        """Creating an index to compute the distances between records

        Args:
            real (np.array): the real preprocessed data
            synth (np.array): the synthetic preprocessed data

        Returns:
            tuple(np.array, np.ndarray): Synthetic-to-Real and Real-to-Real distributions
        """

        n = real.shape[0]
        m = synth.shape[0]

        x_index = faiss.IndexFlatL2(real.shape[1])
        x_index.add(real)

        y_index = faiss.IndexFlatL2(synth.shape[1])
        y_index.add(synth)

        d_xx = np.sqrt(x_index.search(real, self.k + 1)[0][:, -1]) # k + 1 = 2 to have smaller distance, add k parameter
        d_yy = np.sqrt(y_index.search(synth, self.k + 1)[0][:, -1]) ### Can be used to compute SSD
        d_xy = np.sqrt(y_index.search(real, self.k)[0][:, -1]) # just K , add k parameter

        print('Computing Synthetic to Real Distribution')
        with np.errstate(divide='ignore'):
            d_ratio = d_xy / d_xx
        d_ratio = d_ratio[np.isfinite(d_ratio)]

        x_perm = np.random.default_rng().permutation(real)
        x1, x2 = x_perm[:int(n / 2)], x_perm[int(n / 2):]

        x1_index = faiss.IndexFlatL2(x1.shape[1])
        x1_index.add(x1)

        x2_index = faiss.IndexFlatL2(x2.shape[1])
        x2_index.add(x2)

        d_11 = np.sqrt(x1_index.search(x1, 2)[0][:, -1])
        d_12 = np.sqrt(x2_index.search(x1, 1)[0][:, -1])

        print('Computing Real to Real Distribution')
        with np.errstate(divide='ignore'):
            dx_ratio = d_12 / d_11
        dx_ratio = dx_ratio[np.isfinite(dx_ratio)]

        return NearestNeighborData(
            d_xx=d_xx,
            d_yy=d_yy,
            d_xy=d_xy,
            d_ratio=d_ratio,
            dx_ratio=dx_ratio,
        )

class NearestNeighbor_sklearn:
    def __init__(
            self,
            k: int = 1
    ):
        self.k = k


    def compute_nearest_neighbors(self,
                                  real: np.array,
                                  synth: np.array,
    ) -> NearestNeighborData:
        """Creating an index to compute the distances between records

        Args:
            real (np.array): the real preprocessed data
            synth (np.array): the synthetic preprocessed data

        Returns:
            tuple(np.array, np.ndarray): Synthetic-to-Real and Real-to-Real distributions
        """

        n = real.shape[0]
        m = synth.shape[0]

        nn_x = NearestNeighbors(n_neighbors=self.k + 1, metric='euclidean')
        nn_x.fit(real)
        distances_x, indices_x = nn_x.kneighbors(real)

        nn_y = NearestNeighbors(n_neighbors=self.k + 1, metric='euclidean')
        nn_y.fit(synth)
        distances_y, indices_y = nn_y.kneighbors(synth)

        d_xx = distances_x[:, -1]
        d_yy = distances_y[:, -1]

        nn_xy = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
        nn_xy.fit(synth)
        distances_xy, indices_xy = nn_xy.kneighbors(real)
        d_xy = distances_xy[:, -1]

        with np.errstate(divide='ignore'):
            d_ratio = d_xy / d_xx
        d_ratio = d_ratio[np.isfinite(d_ratio)]

        # Compute privacy score
        x_perm = np.random.default_rng().permutation(real)
        x1, x2 = x_perm[:int(n / 2)], x_perm[int(n / 2):]

        nn_1 = NearestNeighbors(n_neighbors=self.k + 1, metric='euclidean')
        nn_1.fit(x1)
        distances_1, indices_1 = nn_1.kneighbors(x1)

        nn_2 = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
        nn_2.fit(x2)
        distances_12, indices_12 = nn_2.kneighbors(x1)

        d_11 = distances_1[:, -1]
        d_12 = distances_12[:, -1]

        with np.errstate(divide='ignore'):
            dx_ratio = d_12 / d_11
        dx_ratio = dx_ratio[np.isfinite(dx_ratio)]

        return NearestNeighborData(
            d_xx=d_xx,
            d_yy=d_yy,
            d_xy=d_xy,
            d_ratio=d_ratio,
            dx_ratio=dx_ratio,
        )