# simple script for compressing the mask tiff images into png format

for i in $(ls | grep tiff)
do
    o=$(echo $i | sed 's/tiff/png/g')
    convert "$i" "$o"
done
