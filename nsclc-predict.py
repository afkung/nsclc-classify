# script for evaluating read count files on trained NSCLC-vs-healthy model
# input: directory containing read counts in CSV format
# output: text file with rows of file names and predictions
# usage: python nsclc-predict.py /path/to/csv_directory /path/to/output_file.txt


import sys
import os
import numpy as np
import pickle as pkl
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim

# PyTorch architecture
class torch_NN(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.layer1 = nn.Linear(input_features, 100)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(100, 50)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

# ensemble model
def ensemble_prediction(value1, value2, value3):
    final_prediction = np.average([value1, max(value1, value2, value3)])
    return final_prediction

# Ensure correct usage
if len(sys.argv) != 3:
    print("USAGE: python nsclc-predict.py /path/to/csv_directory /path/to/output_file.txt")
    exit()

# Get input directory and output file from command line
csv_directory = sys.argv[1]
output_file = sys.argv[2]

# Ensure the directory exists
if not os.path.isdir(csv_directory):
    print(f"Error. Directory {csv_directory} does not exist.")
    exit()

print("NSCLC Predict")

# Load models and feature lists
print("-- Loading models and feature lists...")
feature_lists = pkl.load(open('weight_files/feature_lists.pkl', 'rb'))
gene_mapping = pkl.load(open('weight_files/peptide_gene_mapping.pkl', 'rb'))
trained_model = torch_NN(input_features=len(feature_lists['peptide']))
trained_model.load_state_dict(torch.load('weight_files/peptide_NN.pth',weights_only=True))

pep_LR = pkl.load(open('weight_files/peptide_LR.pkl', 'rb'))
gene_LR = pkl.load(open('weight_files/gene_LR.pkl', 'rb'))

# Open the output file for writing results
with open(output_file, 'w') as output:
    output.write("Filename\tPrediction\n")  # Header row

    # Loop over each file in the input directory
    for filename in os.listdir(csv_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(csv_directory, filename)

            # Reading input file
            input_peptide_dict = {}
            total_counts = 0
            with open(file_path, 'r') as g:
                for line in g:
                    elements = line.strip().split(',')
                    input_peptide_dict[elements[0]] = int(elements[1])
                    total_counts += int(elements[1])

            # Formatting peptide-level input
            peptide_features = feature_lists['peptide']
            pep_values = []
            for peptide in peptide_features:
                if peptide in input_peptide_dict:
                    value = float(input_peptide_dict[peptide]) / total_counts * 100000
                else:
                    value = 0
                pep_values.append(np.log2(value + 1))
            pep_values = np.array(pep_values)

            # Collapsing peptide-level counts to gene-level counts
            input_gene_dict = {}
            for peptide in input_peptide_dict:
                peptide_count = input_peptide_dict[peptide]
                associated_gene = gene_mapping.get(peptide)
                if associated_gene in input_gene_dict:
                    input_gene_dict[associated_gene] += peptide_count
                else:
                    input_gene_dict[associated_gene] = peptide_count

            # Formatting gene-level input
            gene_features = feature_lists['gene']
            gene_values = []
            for gene in gene_features:
                if gene in input_gene_dict:
                    value = float(input_gene_dict[gene]) / total_counts * 100000
                else:
                    value = 0
                gene_values.append(np.log2(value + 1))
            gene_values = np.array(gene_values)

            # Evaluate the models
            #print("Evaluating...")
            prediction1 = float(trained_model(torch.tensor(pep_values, dtype=torch.float32)))
            prediction2 = pep_LR.predict_proba([pep_values])[0][1]
            prediction3 = gene_LR.predict_proba([gene_values])[0][1]

            # Compute ensemble prediction
            final_prediction = ensemble_prediction(prediction1, prediction2, prediction3)
            print(f"-- Processed: {file_path} \t {final_prediction:.2g}")

            # Write the result to the output file
            output.write(f"{filename}\t{final_prediction:.2g}\n")

print(f"-- Results written to {output_file}")