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
 * `test_new.py` *script for testing new read count file*

Usage: download and unzip weight file directory, run `python test_new.py new_read_count_file.csv` from parent directory
