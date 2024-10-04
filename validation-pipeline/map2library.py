# Andrew Kung
# DeRisi Lab, UCSF
# script formatting individual read count files
# usage: python map2library.py file_names

# importing libraries
from collections import Counter
import copy
import sys
import os
from subprocess import call

# iterative mapping function
def map(all_dict, csv_file):
	total = 0
	temp_dict = {}
	h = open(csv_file,'r')
	for line in h:
		elements = line.split(',')
		temp_dict[elements[0].strip()] = elements[1].strip()
		total += float(elements[1].strip())
	h.close()

	for item in all_dict:
		all_dict[item] += ','
		if item in temp_dict:
			all_dict[item] += str(float(temp_dict[item]) * 100000 / total)
		else:
			all_dict[item] += '0'
	return all_dict

# building initial dictionaries
library_path = 'library_fasta/'
seq_dict_pep = {}
g1 = open(library_path + 'peptidome_t7_seq_nospace.fasta', 'r')
for line in g1:
	if line[0] == '>':
		seq_dict_pep[line[1:].strip()] = ""
g1.close()
seq_dict_gene = {}
g2 = open(library_path + 'peptidome_genes.txt', 'r')
for line in g2:
	seq_dict_gene[line.strip()] = ""
g2.close()
training_path = 'all_names.csv'
training_dict = {}
peptide_name_list = []
g3 = open(training_path, 'r')
for line in g3:
	peptide_name = line.strip().split(',')[0]
	peptide_name_list.append(peptide_name)
	training_dict[peptide_name] = ""
g3.close()


# writing new files

output_file_pep = 'sample_files/' + sample_name + '_mapped.csv'
output_file_gene = 'sample_files/' + sample_name + '/' + sample_name + '_mapped_gene.csv'

# populating dictionary
gene_names = 'gene' # initialize first line
pep_names = 'peptide'

name = str(file).strip().split('/')[-1].split('.csv')[0]
gene_names += ',' + name
pep_names += ',' + name
temp_seq_dict_pep = copy.deepcopy(seq_dict_pep)
temp_seq_dict_pep = map(temp_seq_dict_pep, file)

call(['python','collapsebygene.py', file])
temp_seq_dict_gene = copy.deepcopy(seq_dict_gene)
temp_seq_dict_gene = map(temp_seq_dict_gene, file[:-4] + '_gene.csv')

# writing to output file
w1 = open(output_file_pep, 'w')
w1.write(pep_names + '\n')
for peptide in peptide_name_list:
	w1.write(str(peptide) + temp_seq_dict_pep[peptide] + '\n')
w1.close()

w2 = open(output_file_gene, 'w')
w2.write(gene_names + '\n')
for gene in temp_seq_dict_gene:
	w2.write(str(gene) + temp_seq_dict_gene[gene] + '\n')
w2.close()