# nsclc-classify
Code repository for "An autoantibody-based machine learning classifier for the detection of early-stage non-small cell lung cancer" (Kung et al., 2024)

## Associated files:

*Notebook of scripts for formatting datasets*

* `formatting-scripts.ipynb`

*Notebook of scripts evaluating various models*

* `model-evaluation.ipynb`

*Best-performing ensemble model*

* `full-model.py`

*Folder of scripts used for blinded external validation*

* `validation-pipeline`
  * run_validation.sh *master shell script*
  * map2library.py *formatting read counts*
  * filter_testset.py *sorting features to use*
  * train_test.py *training on previous dataset, testing on new dataset*
  * output_files.py *writing output files*

## Running trained model on new file
 * `weight_files.zip` *zipped directory of weight files for component models*
   * peptide_NN.pth *peptide-level neural network model from PyTorch*
   * peptide_LR.pkl *peptide-level logistic regression model from sklearn*
   * gene_LR.pkl *gene-level logistic regression model from sklearn*
   * feature_lists.pkl *ordered lists of input features*
   * peptide_gene_mapping.pkl *mapping file for collapsing by gene*
 * `nsclc-predict.py` *script for evaluating trained model on new read count files*
 * `example_read_counts` *directory of example read count files*
   * example_read_counts_POS.csv
   * example_read_counts_NEG.csv

**Usage:**
* Required packages: numpy, sklearn, torch
* Download and unzip weight file directory
* Run `python nsclc-predict.py /path/to/csv_directory /path/to/output_file.txt`
   * *Input:* directory of sample files (peptide names and counts in line-delimited CSV, see examples)
   * *Output:* text file of file names and predictions between 0 (healthy) and 1 (NSCLC)

Pre-print: https://www.medrxiv.org/content/10.1101/2025.01.09.24315050v1
