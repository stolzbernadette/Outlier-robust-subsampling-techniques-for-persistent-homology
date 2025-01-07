# Landmark Sampling: outlier-robust subsampling techniques for persistent homology

**Project founder:**  
[Bernadette J. Stolz](https://www.maths.ox.ac.uk/people/bernadette.stolz)

**Implementation:**  
[Błażej Banaszewski](https://github.com/blazejba)

## Description
This repository contains Python code for the manuscript “[Outlier-robust subsampling techniques for persistent homology](https://arxiv.org/pdf/2103.14743.pdf).” The script `ph_landmarks.py` implements a landmark selection method, using Ray for parallelization, that iteratively:

1. Computes local outlier (topological) scores for active points. 
2. Selects one landmark at a time.  
3. Re-scores neighbors of that landmark.  
4. Repeats until the desired number of landmarks is chosen.

Landmark Sampling offers a balanced alternative to random and maxmin sampling, where the former tends to ignore all outliers, and the latter selects them overly. In contrast, Landmark Sampling considers the local structure of the point cloud and selects points that are either `vital` to preserving its topology or `representative` of their neighborhood.

This is an efficient implementation that scales to high-dimensional datasets and large point clouds. For example, it was tested on subsampling 500,000 points in 128 dimensions down to 100,000 points. The sampling process took negligible time compared to running Ripser on the subsampled points.

## Usage

```python
from ph_landmarks import LandmarkSampler

ray.init(num_cpus=n_workers, ignore_reinit_error=True)
sampler = LandmarkSampler(
    point_cloud=...,
    n_samples=...,
    topological_radius=...,
    dimension=...,
    scoring_version='restricted' or 'multi',
    landmark_type='representative' or 'vital',
    ignore_super_outliers=True,
    n_workers=n_workers,
    disable_cache=True
)
landmark_idxs, n_super_outlier_landmarks = sampler.compute_landmarks()
ray.shutdown()
```

### Arguments

- **point_cloud**: `(N, d)` array of data points where `N` is the number of points and `d` is the dimension.  
- **n_samples**: Number of landmarks to select.  
- **topological_radius**: Neighborhood radius for local persistent homology.  
- **dimension**: Maximum homology dimension to use when scoring points.  
- **scoring_version**: `'restricted'` (scores from a single dimension) or `'multi'` (max scores across multiple dimensions).  
- **landmark_type**: `'representative'` (low scores) or `'vital'` (high scores) chosen first.  
- **ignore_super_outliers**: If `True`, points with fewer than two neighbors aren’t emphasized.  
- **n_workers**: Number of Ray workers. Recommended ~80% of the available CPU cores. Should equal the number of cpus in `ray.init`.
- **disable_cache**: If `False`, previously generated scores are loaded if if reusable, and if more points were generated they are saved.  

### Outputs

- **landmark_idxs**: Indices of selected landmark points in the original point cloud.  
- **n_super_outlier_landmarks**: Count of points with fewer than two neighbors in their local neighborhood (super outliers).

## Installation
```
# Create an environement
mamba create -n landmark_sampling
mamba activate landmark_sampling

# Install dependencies
mamba install -c conda-forge --file requirements.txt

# Install the package
pip install -e . 
```

## Reference
[1] B.J. Stolz, *Outlier-robust subsampling techniques for persistent homology*, arXiv:2103.14743, 2021.
