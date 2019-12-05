#!/usr/bin/env bash
unset FILE
unset line
unset words
unset output
unset restart

FILE=./flash.par
output=output_dir
line=''
#restart=False
if [ ! -f "$FILE" ]; then
    echo "$FILE does not exist."
    echo "Maybe you moved it somewhere?"
else 
    echo "$FILE exists."
    echo "Checking if a Torch restart."
fi

while read -a words; do
    if [[ ${words[0]} == "restart" ]]; then
	echo "You are issuing a RESTART."
    fi
done < "$FILE"


while read -a words; do
    if [[ ${words[0]} == "restart" ]]; then
	echo "You are issuing a RESTART."
    fi
    
    if [[ ${words[0]} == "output_dir" ]]; then
	echo "output directory is ${words[2]}"
	echo "Checking"
    fi
done < "$FILE"

unset FILE
unset line
unset words
unset output
unset restart
