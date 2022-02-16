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
<li> point_cloud

<p>
<li> topological_radius

<p>
<li> sampling_density

<p>
<li> scoring_version: 'restrictedDim' for restriction to specific dimension or 'multiDim' when considering the maximal persistence over multiple dimensions.

<p>
<li> dimension

<p>
<li> landmark_type

</ul>

<p>
<li>
Output
  
</ul>

All other functions necessary can be found in the same file.

<h3> References </h3>
<a name="GA">[1]</a> Outlier-robust subsampling techniques for persistent homology. BJ Stolz, arXiv: 2103.14743, 2021.
