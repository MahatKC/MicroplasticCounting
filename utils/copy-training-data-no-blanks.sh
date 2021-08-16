#!/bin/bash

# used to copy training data to the output directory
# use this script in the same folder where all the training data is
# pass the output directory as the first argument

out_dir=$1

mkdir $out_dir/imgs
mkdir $out_dir/masks

for f in $(fd raw | grep RAW)
do
    if [[ -d $f ]]; then
        continue
    fi
    base_directory=$(echo "$f" | cut -d "/" -f1)
    seg=$(echo $f | sed 's/jpeg/tiff/g' | sed 's/RAW/SEGMENTED/' | sed 's/raw/seg/')
    file_name_raw=$(basename $f)
    file_name_mask=$(basename $seg)
    new_file_name_mask=$(echo $file_name_mask | sed 's/_seg//g')
    new_file_name_raw=$(echo $file_name_raw | sed 's/_raw//g')

    if [ -f "$seg" ]; then
        cp $f $out_dir/imgs/
        mv $out_dir/imgs/$file_name_raw $out_dir/imgs/"$base_directory"_"$new_file_name_raw"
        cp $seg $out_dir/masks/
        mv $out_dir/masks/$file_name_mask $out_dir/masks/"$base_directory"_"$new_file_name_mask"
    else
        echo "$f has no corresponding mask, skipping"
    fi
done
