# Outlier-robust subsampling techniques for persistent homology

<h3> Project founder: </h3> Bernadette J. Stolz (for more information, see <a href="https://www.maths.ox.ac.uk/people/bernadette.stolz"> here</a>.)
  
<h3> Description </h3>
  
This repository was created for Python code used in the manuscript <a href="#GA">[1]</a>.
The manuscript can be found <a href="https://arxiv.org/pdf/2103.14743.pdf"> here</a>. 

<h3> Python code </h3>

The python function to select landmarks is: getPHLandmarks.py

<ul>

<p>
<li>
Input: 
<ul>

<p>
<li> point_cloud: array-like structure of the form (N,d), where N is the number of points in R^d.

<p>
<li> topological_radius: neighbourhood radius in which local persistent homology computations are performed around each point.

<p>
<li> sampling_density: fraction of landmark points from point_cloud.

<p>
<li> scoring_version: can take values 'restrictedDim' (restriction of landmark score computation to specific dimension) or 'multiDim' (when considering the maximal persistence over multiple dimensions for landmark scores).

<p>
<li> dimension: for 'restrictedDim' this value specifies the persistent homology dimension that is computed to determine the landmark scores, for 'multiDim' this value represents the maximal persistent homology dimension that is computed and considered for the landmark scores.


<p>
<li> landmark_type: can take values 'representative' (points with low landmark scores are selected as landmarks) or 'vital' (points with high landmark scores are selected as landmarks).

</ul>

<p>
<li>
Output: 
  
  <ul>
<p>
<li> PH_landmarks: array-like structure of the form (m,d) containing coordinates of landmark points.

<p>
<li> sorted_indices_point_cloud_original_order: row indices of landmark points in point_cloud.
  
<p>
<li> number_of_super_outliers: number of points with less than two neighbours in their local neighbourhood.
  


</ul>

  
</ul>

All other functions necessary can be found in the same file.

Examples for calling the function:
<ul>

<p>
<li> PH_landmarks, sorted_indices_point_cloud_original_order, number_of_super_outliers = getPHLandmarks(point_cloud, topological_radius, sampling_density, 'multiDim', max_dim_ripser, 'representative')
  
<p>
<li> PH_landmarks, sorted_indices_point_cloud_original_order, number_of_super_outliers = getPHLandmarks(point_cloud, topological_radius, sampling_density, 'restrictedDim', max_dim_ripser, 'vital')
  
 </ul>

<h3> References </h3>
<a name="GA">[1]</a> Outlier-robust subsampling techniques for persistent homology. BJ Stolz, arXiv: 2103.14743, 2021.
