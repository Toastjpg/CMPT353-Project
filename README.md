# Reddit Submissions Analysis

## Required Libraries

pyspark, pandas, numpy, seaborn, matplotlib, scipy.stats

## Project Structure

/data-subsets
- This is a directory containing the resampled and balanced dataset
- Outputted as a result of running 3-initial_analysis.py

/figures
- Visualizations written to file from 3-initial_analysis.py

/submissions-transformed
- A **subset** of the outputted transformed data from running 2-transform.py on cluster

## Order of Execution

Begin by running the scripts in numbered order. 

The following scripts should be run on the cluster in their named number order: 
- 0-extract.py
- 1-filter.py
- 2-transform.py

The folowing analysis scripts can be then run locally, with the submissions-transformed output data copied to an accessible directory
- 3-initial_analysis.py
- 4-relationship_analysis.py

Lastly, the model.py script should be run on the cluster to avoid long training times. The model must be run after files 0-2, and can be run with no input arguments.

## Notes
- The partial outputs of running 2-transform.py have been provided in the /submissions-transformed of the directory.
- The inputs for the analysis in 4-relationship_analysis.py have also been included in /data-subsets.
