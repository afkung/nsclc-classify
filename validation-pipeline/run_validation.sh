#!/bin/bash

# $1 file name

file=$(basename $1)
run=${file:0:-4}
ID=${run:0:8}

if [ -e /data/external-uploads/outgoing/$ID"-result.html" ]; then
	echo "File exists. Exiting"
	exit 1
fi

# moving count files
cp /data/external-uploads/$run".zip" validation_script/files
mkdir validation_script/files/$run
mkdir validation_script/files/$run"/read_counts"
unzip validation_script/files/$run".zip" -d validation_script/files/$run"/read_counts"

# creating RPK matrices
python validation_script/map2library.py $run

#Training sets:
#All (301 samples)

# run against test set
python validation_script/filter_testset.py $run all
python validation_script/train_test.py $run all

python validation_script/output_files.py $run

cp validation_script/files/$run/$run"_RESULTS.csv" /data/external-uploads/outgoing
cp validation_script/files/$run/$ID"-result.html" /data/external-uploads/outgoing