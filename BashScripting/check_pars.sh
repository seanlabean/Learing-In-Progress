#!/usr/bin/env bash

# 
# This script is purpose built to be run in a TORCH "simulation" directory.
# What this script does (two options):
# 1. If simulation is new (output directory defined in flash.par is empty),
#    copys flash.par file to the data directory and exits.
# 2. If simulation is a RESTART (output directory is NOT empty),
#    runs a diff between current and old flash.par files so user can confirm
#    that only intended changes have been made to the parameter file.
#
# - Sean C. Lewis; Drexel University (Dec. 2019)
#

unset FILE
unset line
unset words
unset output
unset restart

FILE=flash.par
line=''

# Check if input FILE exists. https://linuxize.com/post/bash-check-if-file-exists/
if [ ! -f "$FILE" ]; then
    echo "ERROR: $FILE does not exist."
    echo "Maybe you moved it somewhere?"
    echo "Exiting..."
    # Exit .sh script without terminating shell. https://stackoverflow.com/questions/11141120
    kill -INT $$
else 
    echo ""
fi

# Read in lines from FILE, divide each line into array of words.
# https://stackoverflow.com/questions/10929453
while read -a words; do
    if [[ ${words[0]} == "output_directory=" ]]; then
	# Remove quotations around output directory name.
	# https://stackoverflow.com/questions/9733338
	temp=${words[1]}
	temp="${temp%\"}"
	output="${temp#\"}"
	echo "output directory is ${words[1]}"
	echo "Checking status of $output"
    fi
done < "$FILE"

# Check if output directory is empty. https://superuser.com/questions/352289
if [ -n "$(find "$output" -maxdepth 0 -type d -empty 2>/dev/null)" ]; then
    echo "Empty directory, we need to copy over .par file."
    # Prompt user for confirmation. https://stackoverflow.com/questions/226703
    while true; do
	read -p "This is a new simulation run, correct? (y/n): " yn
	case $yn in
            [Yy]* ) break;;
            [Nn]* ) kill -INT $$;;
            * ) echo "Please answer y or n.";;
	esac
    done
    echo "Copying $FILE to $output."
    cp $FILE $output
    echo "Simulation commencing..."
else
    echo "$output is NOT empty."
    echo "Therefore, RESTART is assumed. Performing .par diff check."
    echo "Expect AT LEAST: checkpointFileNumber & plotFileNumber diffs."
    diff $FILE $output/$FILE
    while true; do
	read -p "Check these differences, is everything expected? (y/n): " yn
	case $yn in
            [Yy]* ) break;;
            [Nn]* ) kill -INT $$;;
            * ) echo "Please answer y or n.";;
	esac
    done
    echo "Simulation RESTART initiated..."
fi

# I am not sure if the unsetting of the variables defined within this script is entirely
# necessary, but I like it because it leaves my shell's env vars less cluttered. - SCL
unset FILE
unset line
unset words
unset output
unset restart
unset yn
