# Andrew Kung
# DeRisi Lab, UCSF
# Training on Previous Cohort, Testing on New Cohort

import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

def format_training(pos_data_file, neg_data_file):
    pos_df = pd.read_csv(open(pos_data_file,'r'), header = 0, index_col = 0)
    neg_df = pd.read_csv(open(neg_data_file,'r'), header = 0, index_col = 0)
    pos_df = pos_df.T
    neg_df = neg_df.T
    pos_df = pos_df.transform(lambda x: np.log2(x+1))
    neg_df = neg_df.transform(lambda x: np.log2(x+1))
    pos_list = pos_df.values
    neg_list = neg_df.values

    values = []
    labels = []
    for item in pos_list:
        values.append(item)
        labels.append(1)
    for item in neg_list:
        values.append(item)
        labels.append(0)

    return np.asarray(values,dtype = np.float32), np.asarray(labels,dtype = np.float32)

def format_test(test_data_file):
    test_df = pd.read_csv(open(test_data_file,'r'), header = 0, index_col = 0)
    test_df = test_df.T
    test_df = test_df.transform(lambda x: np.log2(x+1))
    values = test_df.values
    return np.asarray(values,dtype = np.float32)

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


run_name = sys.argv[1]
training_set = sys.argv[2]

names_file = 'validation_script/files/%s/%s_samples.csv' % (run_name,run_name)
names_list = []
g = open(names_file, 'r')
for line in g:
    names_list.append(line.strip())
g.close()


# peptide level
training_case = 'validation_script/training_sets/%s_case.csv' % training_set
training_control = 'validation_script/training_sets/%s_control.csv' % training_set
test_data = 'validation_script/files/%s/%s_%s_mapped.csv' % (run_name, run_name, training_set)

training_input, training_output = format_training(training_case,training_control)
test_input = format_test(test_data)

test_predictions1 = []
clf1 = LogisticRegression(penalty = 'l2', random_state=None, solver = 'liblinear', max_iter = 500).fit(training_input, training_output)
prob_array1 = clf1.predict_proba(test_input)
prob_values1 = [i[1] for i in prob_array1]

test_predictions2 = []
nn_model = torch_NN(input_features = len(training_input[1]))
clf2 = train_NN(nn_model, torch.tensor(training_input, dtype=torch.float32), torch.tensor(training_output, dtype=torch.float32).unsqueeze(-1))
prob_array2 = clf2(torch.tensor(test_input, dtype=torch.float32))
prob_values2 = [float(i) for i in prob_array2]

# gene level
training_case_gene = 'validation_script/training_sets/%s_gene_case.csv' % training_set
training_control_gene = 'validation_script/training_sets/%s_gene_control.csv' % training_set
test_data_gene = 'validation_script/files/%s/%s_mapped_gene.csv' % (run_name, run_name)

training_input, training_output = format_training(training_case_gene,training_control_gene)
test_input = format_test(test_data_gene)

test_predictions3 = []
clf3 = LogisticRegression(penalty = 'l2', random_state=None, solver = 'liblinear', max_iter = 500).fit(training_input, training_output)
prob_array3 = clf3.predict_proba(test_input)
prob_values3 = [i[1] for i in prob_array3]


# ensemble prediction
test_probs = []
for item in zip(prob_values1,prob_values2,prob_values3):
    test_probs.append(np.average(item[0], np.max(item))

output_file = 'validation_script/files/%s/%s_%s_predictions.csv' % (run_name, run_name, training_set)

h = open(output_file, 'w')
for item in zip(names_list, test_probs):
    h.write(str(item[0]) + ',' + str(item[1]) + '\n')
h.close()