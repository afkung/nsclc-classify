# script for testing new file on trained NSCLC-vs-healthy model
# usage: python evaluation.py read_count_file.csv
# Andrew Kung (DeRisi Lab, UCSF)

# import libraries
try:
    import sys
    import numpy as np
    import pickle as pkl
    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
except:
    print("Please install the following packages: numpy, sklearn, torch")
    print("Example call: pip install sklearn")
    exit()

# PyTorch architectue
class torch_NN(nn.Module):
    def __init__(self,input_features):
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


if len(sys.argv) != 2:
    print("USAGE: python evaluation.py read_count_file.csv")
    exit()

count_file = sys.argv[1]
# reading input file
print("Reading input file: " + count_file)

input_peptide_dict = {}
total_counts = 0
g = open(count_file, 'r')
for line in g:
    elements = line.strip().split(',')
    input_peptide_dict[elements[0]] = int(elements[1])
    total_counts += int(elements[1])
g.close()


print("Formatting inputs")
# loading feature lists
feature_lists = pkl.load(open('weight_files/feature_lists.pkl','rb'))

# formatting peptide-level input
peptide_features = feature_lists['peptide']
pep_values = []
for peptide in peptide_features:
    if peptide in input_peptide_dict:
        value = float(input_peptide_dict[peptide]) / total_counts * 100000
    else:
        value = 0
    pep_values.append(np.log2(value + 1))
pep_values = np.array(pep_values)

# collapsing peptide-level counts to gene-level counts
gene_mapping = pkl.load(open('weight_files/peptide_gene_mapping.pkl','rb'))
input_gene_dict = {}
for peptide in input_peptide_dict:
    peptide_count = input_peptide_dict[peptide]
    associated_gene = gene_mapping[peptide]
    if associated_gene in input_gene_dict:
        input_gene_dict[associated_gene] = input_gene_dict[associated_gene] + peptide_count    
    else:
        input_gene_dict[associated_gene] = peptide_count

# formatting gene-level input        
gene_features = feature_lists['gene']
gene_values = []
for gene in gene_features:
    if gene in input_gene_dict:
        value = float(input_gene_dict[gene]) / total_counts * 100000
    else:
        value = 0
    gene_values.append(np.log2(value + 1))
gene_values = np.array(gene_values)


print("Loading models")
# loading NN model
trained_model = torch_NN(input_features = len(peptide_features))
trained_model.load_state_dict(torch.load('weight_files/peptide_NN.pth'))

# loading peptide-level logistic regression
pep_LR = pkl.load(open('weight_files/peptide_LR.pkl', 'rb'))

# loading gene-level logistic regression
gene_LR = pkl.load(open('weight_files/gene_LR.pkl', 'rb'))


print("Evaluating...")
# evaluation
prediction1 = float(trained_model(torch.tensor(pep_values, dtype = torch.float32)))
prediction2 = pep_LR.predict_proba([pep_values])[0][1]
prediction3 = gene_LR.predict_proba([gene_values])[0][1]


print("PREDICTION VALUE:")
print(ensemble_prediction(prediction1, prediction2, prediction3))