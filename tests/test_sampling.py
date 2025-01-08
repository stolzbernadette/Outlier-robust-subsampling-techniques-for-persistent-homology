import numpy as np
import pytest
import ray

from ph_landmarks.ph_landmarks import LandmarkSampler

@pytest.mark.parametrize("n_points, dim, n_components, n_landmarks", [
    (10_000, 5, 3, 500),  # Example 1: moderate N
    (2_000, 10, 2, 200),  # Example 2: smaller N, higher dimension
])
def test_mixture_gaussians_subsampling(n_points, dim, n_components, n_landmarks):
    """
    Generate a mixture of Gaussians in R^dim, then test
    LandmarkSampler on the resulting point cloud.
    """
    np.random.seed(42)

    means = [np.random.randn(dim) * 5 for _ in range(n_components)]
    points_per_gaussian = n_points // n_components

    point_cloud = np.vstack([np.random.randn(points_per_gaussian, dim) + mean
                             for mean in means])

    # If there's leftover points due to integer division, add them to the last cluster
    if point_cloud.shape[0] < n_points:
        remainder    = n_points - point_cloud.shape[0]
        extra_points = np.random.randn(remainder, dim) + means[-1]
        point_cloud  = np.vstack([point_cloud, extra_points])

    n_workers = 16
    ray.init(num_cpus=n_workers, ignore_reinit_error=True)

    # Create and run the LandmarkSampler
    sampler = LandmarkSampler(point_cloud=point_cloud,
                              n_samples=n_landmarks,
                              topological_radius=1.0,
                              dimension=1,
                              scoring_version='restricted',
                              landmark_type='representative',
                              ignore_super_outliers=True,
                              n_workers=n_workers,
                              disable_cache=True)
    
    landmark_idxs, n_super_outlier_landmarks = sampler.compute_landmarks()

    assert len(landmark_idxs) == n_landmarks, (
        f"Expected {n_landmarks} landmarks, got {len(landmark_idxs)}."
    )
    assert 0 <= n_super_outlier_landmarks <= n_landmarks, (
        "Number of super outliers must be between 0 and the number of landmarks."
    )

    print(f"\nTest with N={n_points}, d={dim}, k={n_components} passed.")
    print(f"Selected {len(landmark_idxs)} landmarks.")
    print(f"Found {n_super_outlier_landmarks} super outliers.")

    ray.shutdown()
