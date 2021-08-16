# blob_counter.py
Simple blob counter which will count the number of blobs based on the mask images and place them into the out_dir. Change the path variables in the script as needed

# compress.sh
Simple script to compress tiff masks to png masks. Huge space savings to be had converting to png images. To use it, move the script into the directory containing the masks and run it: `./compress.sh`

# copy-training-data-no-blanks.sh
Copies the training data which can be downloaded from onedrive into an output directory. Used to prepare data for training U-Net. To use, call the script in the same directory as the downloaded folders and pass the output directory as the first argument. Does not copy over images with no microplastics.

# copy-training-data-with-blanks.sh
Copies the training data which can be downloaded from onedrive into an output directory. Used to prepare data for training U-Net. To use, call the script in the same directory as the downloaded folders and pass the output directory as the first argument. Copies over images with no microplastics and assign those images with a blank mask template.

# sort.py
Should not be used directly, instead should be used by sort-microplastics-by-count.sh

# sort-microplasticss-by-count
Sorts microplastics by their counts, and places them into a directory called `sorted`. Copy this script and `sort.py` into a directory containing all the images downloaded from OneDrive and run this script: `./sort-microplastics-by-count`
