# Andrew Kung
# DeRisi Lab, UCSF

import sys
import os
import numpy as np
import pickle as pkl
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim

random_seed = 5

# pytorch architecture
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

def train_NN(model, X_train, y_train):

    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    number_epochs = 500
    batch_size = len(X_train)
    batch_start = torch.arange(0, len(X_train), batch_size)

    for epoch in range(number_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_function(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return model
    

test_probs1 = []
test_probs2 = []
test_probs3 = []


# loading and formatting peptide data
case_csv = 'nsclc_rpk.csv'
control_csv = 'healthy_rpk.csv'

case_df = pd.read_csv(open(case_csv, 'r'), header = 0, index_col = 0)
control_df = pd.read_csv(open(control_csv, 'r'), header = 0, index_col = 0)

# transpose
case_df = case_df.T
control_df = control_df.T

# log-transform
case_df = case_df.transform(lambda x: np.log2(x+1))
control_df = control_df.transform(lambda x: np.log2(x+1))

# load into lists
pos_list = case_df.values
neg_list = control_df.values


# loading and formatting gene data
case_csv_gene = 'nsclc_rpk_gene.csv'
control_csv_gene = 'healthy_rpk_gene.csv'

case_df_gene = pd.read_csv(open(case_csv_gene, 'r'), header = 0, index_col = 0)
control_df_gene = pd.read_csv(open(control_csv_gene, 'r'), header = 0, index_col = 0)

# transpose
case_df_gene = case_df_gene.T
control_df_gene = control_df_gene.T

# log-transform
case_df_gene = case_df_gene.transform(lambda x: np.log2(x+1))
control_df_gene = control_df_gene.transform(lambda x: np.log2(x+1))

# load into lists
pos_list_gene = case_df_gene.values
neg_list_gene = control_df_gene.values


# setting up k-fold cross-validation
number_slices = 10
pos_number = len(pos_list)
neg_number = len(neg_list)
pos_slice_size = int(len(pos_list)/number_slices)
neg_slice_size = int(len(neg_list)/number_slices)
pos_extra_slices = len(pos_list) % number_slices
neg_extra_slices = len(neg_list) % number_slices

np.random.seed(random_seed)
pos_test_list = np.random.choice(len(pos_list), len(pos_list), replace = False)
np.random.seed(random_seed)
neg_test_list = np.random.choice(len(neg_list), len(neg_list), replace = False)

pos_index = 0
neg_index = 0


# cross-validation
for k_slice in range(0,number_slices):

    training_input = []
    training_output = []
    test_input = []
    test_output = []

    training_input_gene = []
    training_output_gene = []
    test_input_gene = []
    test_output_gene= []

    if k_slice < pos_extra_slices:
        new_pos_index = pos_index + pos_slice_size + 1
    else:
        new_pos_index = pos_index + pos_slice_size
    pos_test_indices = list(pos_test_list[pos_index:new_pos_index])
    pos_index = new_pos_index

    for index in range(len(pos_list)):
        if index not in pos_test_indices:
            training_input.append(pos_list[index])
            training_output.append(1)
            training_input_gene.append(pos_list[index])
            training_output_gene.append(1)
        else:
            test_input.append(pos_list[index])
            test_output.append(1)
            test_input_gene.append(pos_list[index])
            test_output_gene.append(1)
            
    if k_slice < neg_extra_slices:
        new_neg_index = neg_index + neg_slice_size + 1
    else:
        new_neg_index = neg_index + neg_slice_size
    neg_test_indices = list(neg_test_list[neg_index:new_neg_index])
    neg_index = new_neg_index

    for index in range(len(neg_list)):
        if index not in neg_test_indices:
            training_input.append(neg_list[index])
            training_output.append(0)
            training_input_gene.append(neg_list[index])
            training_output_gene.append(0)
        else:
            test_input.append(neg_list[index])
            test_output.append(0)
            test_input_gene.append(neg_list[index])
            test_output_gene.append(0)

	# component 1: logistic regression on peptide-level
	clf1 = LogisticRegression(penalty = 'l2', random_state=None, solver = 'liblinear', max_iter = 500).fit(training_input, training_output)
	prob_array1 = clf1.predict_proba(test_input)
	for item in prob_array1:
			test_probs1.append(item[1])
	
	# component 2: feed-forward neural network on peptide-level
	nn_model = torch_NN(input_features = number_features)
	nn_predictions = train_test_NN(nn_model, torch.tensor(training_input, dtype=torch.float32), torch.tensor(training_output, dtype=torch.float32).unsqueeze(-1), torch.tensor(test_input, dtype=torch.float32), torch.tensor(test_output, dtype=torch.float32).unsqueeze(-1))
	for item in nn_predictions:
		test_probs2.append([float(i) for i in predictions[0]])
	
	# component 3: logistic regression on gene-level
	clf3 = LogisticRegression(penalty = 'l2', random_state=None, solver = 'liblinear', max_iter = 500).fit(training_input_gene, training_output_gene)
	prob_array3 = clf3.predict_proba(test_input_gene)
	for item in prob_array3:
			test_probs3.append(item[1])

	for item in zip(test_probs1, test_probs2, test_probs3):
		ensemble_pred.append(np.average([item[0], max(item[0], item[1], item[2])]))
	for label in test_output:
		test_labels.append(label)


# plotting results
fpr, tpr, thresholds = metrics.roc_curve(test_labels, ensemble_pred, pos_label = 1.0)
ROC_auc = metrics.auc(fpr, tpr)
plt.rcParams["font.size"] = 12
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='red', lw=lw, label='AUC = %0.2f' % ROC_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Case (n=%i) vs. Control (n=%i)' % (pos_number, neg_number) )
plt.legend(loc="lower right")