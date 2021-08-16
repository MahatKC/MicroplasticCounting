#!/bin/bash
# sorts the microplastics by their count and place them in the folder "sorted"

mkdir sorted
find . -name "*.docx" -print0 | while read -d $'\0' file
do
    echo "Processing $file"
    label=$(echo $file | awk -F '/' '{print $NF}' | sed 's/\.docx//g')
    echo $label
    python3 ./sort.py -r -l "$file" -o ./sorted -i $label -p "$label/RAW"
done
