import os
import ray
import pickle
import numpy as np
from ripser import ripser
from typing import Union, Tuple
from scipy.spatial import cKDTree


class LandmarkSampler:
    """
    Samples landmarks from a point cloud using topological scoring.
    """

    def __init__(self,
                 point_cloud           : np.ndarray,
                 n_samples             : int,
                 topological_radius    : float,
                 dimension             : int,
                 scoring_version       : str,
                 landmark_type         : str,
                 ignore_super_outliers : bool = False,
                 n_workers             : int = None,
                 disable_cache         : bool = False,):

        """
        Initialize sampler parameters and build a KD-tree.

        :param point_cloud           : Array of shape (N, d) representing the point cloud.
        :param n_samples             : Number of landmarks to sample.
        :param topological_radius    : Radius within which local topological features are scored.
        :param dimension             : Max dimension for persistent homology calculation.
        :param scoring_version       : Either 'restricted' or 'multi' to define scoring strategy.
        :param landmark_type         : Either 'representative' or 'vital' to order scores.
        :param ignore_super_outliers : If True, ignores low-density super outliers.
        :param n_workers             : Number of parallel workers for Ray tasks.
        """

        assert scoring_version in ['restricted', 'multi'], "Invalid scoring version"
        assert landmark_type   in ['representative', 'vital'], "Invalid landmark type"

        if hasattr(point_cloud, 'cpu'):
            point_cloud = point_cloud.cpu().numpy()

        self.point_cloud           = point_cloud
        self.n_samples             = n_samples
        self.topological_radius    = topological_radius
        self.landmark_type         = landmark_type
        self.dimension             = dimension
        self.scoring_version       = scoring_version
        self.n_workers             = n_workers or 1
        self.ignore_super_outliers = ignore_super_outliers

        self.kd_tree = cKDTree(self.point_cloud)

        self.disable_cache = disable_cache
        self.cache_base_path = f"results/cache_{scoring_version}_{landmark_type}_{dimension}d"

    def compute_landmarks(self) -> Tuple[list, int]:
        """
        Compute landmark indices based on outlier scores and topological features.

        :return: A tuple (landmark_idxs, n_super_outlier_landmarks) where:
                 - landmark_idxs is the list of selected landmark indices,
                 - n_super_outlier_landmarks is how many were super outliers.
        """
        print("[INFO] Computing inital scores for outliers...")
        # Boolean mask to track which points are still active
        # Initially, all points are active
        is_active = np.ones(len(self.point_cloud), dtype=bool)        

        if self.disable_cache:
            cache_filenames = []
        else:
            os.makedirs("results", exist_ok=True)
            cache_prefix    = f"cache_{self.scoring_version}_{self.landmark_type}_{self.dimension}d_"
            cache_filenames = [f for f in os.listdir("results") if f.startswith(cache_prefix)]
            cache_stage     = [int(os.path.splitext(f)[0].split("_")[-1]) for f in cache_filenames]
        cache_idx = None
        
        if len(cache_filenames) > 0:
            cache_stage, cache_filenames = zip(*sorted(zip(cache_stage, cache_filenames)))
            for i, stage in enumerate(cache_stage):
                if stage <= self.n_samples:
                    cache_idx = i

        if cache_idx is not None:
            cache_path = os.path.join("results", cache_filenames[cache_idx])
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
                outlier_scores            = cache[0]
                super_outlier_idxs        = cache[1]
                landmark_idxs             = cache[2]
                n_super_outlier_landmarks = cache[3]
            is_active[landmark_idxs] = False
            active_idxs = np.where(is_active)[0]
            print("\t[INFO] Loaded initial outlier scores and super_outlier_indices from cache")
        else:
            n_super_outlier_landmarks = 0
            landmark_idxs = []
            active_idxs = np.where(is_active)[0]
            outlier_scores, super_outlier_idxs = self.get_outlier_scores(self.point_cloud,
                                                                         is_active,
                                                                         active_idxs)
            print("\t[INFO] Computed and cached initial outlier scores and super_outlier_indices.")
   
        sorted_idxs = self.sort_cloud_by_scores(outlier_scores,
                                                super_outlier_idxs.shape[0],
                                                active_idxs)

        for _ in range(self.n_samples - len(landmark_idxs)):
            print(f"\t[INFO] Landmark {len(landmark_idxs) + 1} / {self.n_samples}", end='\r')
            landmark_idx = sorted_idxs[0]
            landmark_idxs.append(landmark_idx)
            if landmark_idx in super_outlier_idxs:
                n_super_outlier_landmarks += 1
            is_active[landmark_idx] = False

            landmark      = self.point_cloud[landmark_idx]
            local_indices = self.kd_tree.query_ball_point(landmark, r=self.topological_radius)
            local_indices = [i for i in local_indices if i != landmark_idx and is_active[i]]

            local_outlier_scores, local_super_outlier_indices = self.get_outlier_scores(
                self.point_cloud, is_active, local_indices
            )

            current_active_indices = np.where(is_active)[0]
            position_map = {idx: pos for pos, idx in enumerate(current_active_indices)}

            # Update outlier_scores for local points
            for li, score in zip(local_indices, local_outlier_scores):
                outlier_scores[position_map[li]] = score

            # Re-align outlier_scores and super_outlier_indices with current_active_indices
            outlier_scores = outlier_scores[[position_map[i] for i in current_active_indices]]
            super_outlier_idxs = super_outlier_idxs[np.isin(super_outlier_idxs,
                                                                  current_active_indices)]

            super_outlier_idxs = np.unique(np.concatenate((super_outlier_idxs,
                                                              local_super_outlier_indices)))
            
            sorted_idxs = self.sort_cloud_by_scores(outlier_scores, 
                                                       super_outlier_idxs.shape[0], 
                                                       current_active_indices)

        if not self.disable_cache:
            
            cache_path = f"{self.cache_base_path}_{self.n_samples}.pkl"
            with open(cache_path, "wb") as f:
                pickle.dump((
                    outlier_scores, 
                    super_outlier_idxs,
                    landmark_idxs,
                    n_super_outlier_landmarks
                ), f)
            
        return landmark_idxs, n_super_outlier_landmarks

    def sort_cloud_by_scores(self,
                             outlier_scores   : np.ndarray,
                             n_super_outliers : int,
                             active_idxs      : np.ndarray) -> np.ndarray:
        """
        Sort active points by outlier scores.

        :param outlier_scores   : Array of outlier scores for active points.
        :param n_super_outliers : Number of super outliers among active points.
        :param active_indices   : Indices of points currently considered active.

        :return: Sorted indices of active points according to the sampling strategy.
        """
        sorted_order = np.argsort(outlier_scores)

        if self.landmark_type == 'vital':
            sorted_order = np.flip(sorted_order)

        if not self.ignore_super_outliers:
            if n_super_outliers > 0:
                permuted_super_outlier_idxs = np.random.permutation(sorted_order[:n_super_outliers])
                sorted_idxs_no_super_outliers = sorted_order[n_super_outliers:]
                sorted_order = np.concatenate(
                    (sorted_idxs_no_super_outliers, permuted_super_outlier_idxs) 
                    if self.landmark_type == 'representative' else
                    (permuted_super_outlier_idxs, sorted_idxs_no_super_outliers)
                )

        return active_idxs[sorted_order]

    def get_outlier_scores(self,
                           point_cloud    : np.ndarray,
                           is_active      : np.ndarray,
                           target_indices : Union[list, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute outlier scores in parallel for selected points.

        :param point_cloud    : Full point cloud data array.
        :param is_active      : Boolean array indicating which points are active.
        :param target_indices : Indices of points for which to compute scores.

        :return: A tuple (landmark_idxs, n_super_outlier_landmarks) where:
                 - landmark_idxs is the list of selected landmark indices,
                 - n_super_outlier_landmarks is how many were super outliers.
        """
        if not isinstance(target_indices, np.ndarray):
            target_indices = np.array(target_indices)

        total_points = len(target_indices)
        outlier_scores = np.empty(total_points, dtype=float)
        super_outlier_indices = []

        if total_points == 0:
            return outlier_scores, np.array(super_outlier_indices)

        futures_map = {}
        tasks_in_flight = min(self.n_workers, total_points)
        for i in range(tasks_in_flight):
            idx = target_indices[i]
            f = get_point_score.remote(point_cloud,
                                       idx,
                                       self.kd_tree,
                                       self.topological_radius,
                                       self.dimension,
                                       self.scoring_version,
                                       is_active)
            futures_map[f] = i

        next_i = tasks_in_flight
        completed_count = 0

        for f in as_completed(list(futures_map.keys())):
            res = ray.get(f)
            i = futures_map.pop(f)
            completed_count += 1
            outlier_scores[i] = res['outlier_score']
            if res['super_outlier_idx'] is not None:
                super_outlier_indices.append(res['super_outlier_idx'])

            if completed_count % self.n_workers == 0:
                print(f"\t[INFO] Processed {completed_count}/{total_points} points", end='\r')

            if next_i < total_points:
                idx = target_indices[next_i]
                nf = get_point_score.remote(point_cloud,
                                            idx,
                                            self.kd_tree,
                                            self.topological_radius,
                                            self.dimension,
                                            self.scoring_version,
                                            is_active)
                futures_map[nf] = next_i
                next_i += 1

            if completed_count == total_points:
                break

        return outlier_scores, np.array(super_outlier_indices)
    

@ray.remote
def get_point_score(point_cloud        : np.ndarray,
                    point_index        : int,
                    kd_tree            : cKDTree,
                    topological_radius : float,
                    dimension          : int,
                    scoring_version    : str,
                    is_active          : np.ndarray) -> dict:
    """
    Compute the outlier score for a single point by constructing a local persistent homology.

    :param point_cloud        : Full point cloud data array.
    :param point_index        : Index of the point to score.
    :param kd_tree            : KD-tree built on the entire point cloud.
    :param topological_radius : Neighborhood search radius.
    :param dimension          : Max dimension for persistent homology calculation.
    :param scoring_version    : 'restricted-dim' or 'multi' for scoring.
    :param is_active          : Boolean mask indicating which points are still active.

    :return: A dictionary with 'outlier_score' and 'super_outlier_idx'.
    """
    point = point_cloud[point_index]
    indices = kd_tree.query_ball_point(point, r=topological_radius)
    indices = [i for i in indices if i != point_index and is_active[i]]

    if len(indices) < 2:
        return {'outlier_score': 0.0, 'super_outlier_idx': point_index}

    delta_point_cloud = point_cloud[indices, :]
    diagrams = ripser(delta_point_cloud, maxdim=dimension)['dgms']

    if scoring_version == 'restricted-dim':
        intervals = diagrams[dimension] if dimension < len(diagrams) else np.array([])
        outlier_score = get_max_persistence(intervals)
    else:  # multidim scoring
        max_persistence_over_dims = 0
        for dim in range(dimension + 1):
            if dim < len(diagrams):
                intervals = diagrams[dim]
                max_persistence = get_max_persistence(intervals)
                if max_persistence > max_persistence_over_dims:
                    max_persistence_over_dims = max_persistence
        outlier_score = max_persistence_over_dims

    return {'outlier_score': outlier_score, 'super_outlier_idx': None}


def get_max_persistence(ripser_pd):
    """
    Compute the maximum persistence (length) among finite intervals in a diagram.

    :param ripser_pd: Persistence diagram of shape (N, 2).
    :return: The maximum (death - birth) value over finite intervals.
    """
    if ripser_pd.size == 0:
        return 0
    finite_bars = ripser_pd[np.isfinite(ripser_pd[:,1])]
    if finite_bars.size == 0:
        return 0
    return np.max(finite_bars[:,1] - finite_bars[:,0])


def as_completed(futures):
    """
    Generator that returns completed Ray futures in order.

    :param futures: List of Ray futures to track.
    :yield: Completed future objects as they finish.
    """
    not_done = set(futures)
    while not_done:
        done, not_done = ray.wait(list(not_done), num_returns=1, timeout=None)
        for d in done:
            yield d
