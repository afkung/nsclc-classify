# Andrew Kung
# DeRisi Lab, UCSF
# filtering and organizing features in test set based on training set
# usage: python filter_testset.py run_name training_set

import sys
run_name = sys.argv[1]
training_set = sys.argv[2]

test_csv = 'validation/files/' + run_name + '/' + run_name + '_mapped.csv'
training_csv = 'validation/training_sets/' + training_set + '_case.csv'
filtered_test_csv = 'validation/files/' + run_name + '/' + run_name + '_' + training_set + '_mapped.csv'

# load dictionary keys
peptide_name_list = []
peptide_dict = {}
g = open(training_csv, 'r')
for line in g:
	peptide_name = line.strip().split(',')[0]
	peptide_name_list.append(peptide_name)
	peptide_dict[peptide_name] = ""
g.close()

# load dictionary values
g2 = open(test_csv, 'r')
for line in g2:
	peptide_name = line.strip().split(',')[0]
	if peptide_name in peptide_dict:
		peptide_dict[peptide_name] = line
g2.close()

# write new file
h = open(filtered_test_csv, 'w')
for item in peptide_name_list:
	h.write(peptide_dict[item])
	h.write('\n')
h.close()