# Andrew Kung
# DeRisi Lab, UCSF
# generate output file and html page

import sys
from subprocess import call
from datetime import date

run_name = sys.argv[1]

output_dict = {}

sample_names = 'validation_script/files/%s/%s_samples.csv' % (run_name,run_name)
g = open(sample_names, 'r')
for item in g:
    output_dict[item.strip()] = str(item.strip())
g.close()

training_set_list = ['all']

for training_set in training_set_list:
    prediction_file = 'validation_script/files/%s/%s_%s_predictions.csv' % (run_name, run_name, training_set)
    g = open(prediction_file, 'r')
    for line in g:
        elements = line.strip().split(',')
        sample = elements[0]
        prediction = elements[1]
        output_dict[sample] = output_dict[sample] + ',' + str(prediction)
    g.close()

merged_file = 'validation_script/files/%s/%s_RESULTS.csv' % (run_name, run_name)

h1 = open(merged_file, 'w')
h1.write('sample_name,all\n')
for sample in output_dict:
    h1.write(output_dict[sample] + '\n')
h1.close()

html_page = 'validation_script/files/%s/%s-result.html' % (run_name, run_name[0:8])
header = '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"        "http://www.w3.org/TR/html4/loose.dtd"><html><head>	<title>Your Results</title>	<link rel="stylesheet" href="../jd-bootstrap-min.css" rel="stylesheet" type="text/css">	<meta name="viewport" content="width=device-width, initial-scale=1"></head><body><nav class="navbar navbar-inverse">  <div class="container-fluid">    <div class="navbar-header">      <a class="navbar-brand" href="index.php">Uploads</a>    </div>    <ul class="nav navbar-nav">      <li class="active"><a href="derisi_upload.html">Home</a></li>    </ul>  </div></nav><br><div class="container">    <div class="panel panel-primary">      <div class="panel-heading">Processed file download</div>      <div class="panel-body"><h3>Your Results:</h3><br>'
footer = '</div></div></div></div></body></html>'

file_name = run_name + '_RESULTS.csv'

h2 = open(html_page, 'w')
h2.write(header)
h2.write('<b>File Name:</b> %s <br>' % file_name)
h2.write('<b>Date:</b> %s <br><br>' % str(date.today()))
h2.write('<a href=%s  download="%s" >Download as CSV</a>' % (file_name,file_name))
h2.write(footer)
h2.close()