#!/usr/bin/python3

import docx
import argparse
import os
from shutil import copyfile
import shutil

parser = argparse.ArgumentParser(description='Sorts images into their corresponding classes based on their counts and places it in the output dir')
parser.add_argument('-l', '--labels', help='The docx containing the labels of the images', required=True)
parser.add_argument('-p', '--photos-dir', help='The folder containing the images to be sorted', required=True)
parser.add_argument('-o', '--output-dir', help='The directory in which the script will place the sorted images into their corresponding classes', required=True)
parser.add_argument('-i', '--identifier', help='Add a unique identifier to the image filename to prevent overwritting of same image name from different batch', required=True)
parser.add_argument('-r', '--replace', help='Replace seg with raw for image names, needed for MP2 dataset as the docx records file names as \'seg\' while the raw images use \'raw\'', required=False, default=True, action='store_true')
args = parser.parse_args()

doc = docx.Document(args.labels)
folders_created = []
missing_flag = False

identifier = args.identifier

# add trailing slash to photos dir
photos_dir = args.photos_dir
if not photos_dir.endswith(os.path.sep):
    photos_dir += os.path.sep

# add trailing slash to output dir
output_dir = args.output_dir
if not output_dir.endswith(os.path.sep):
    output_dir += os.path.sep

for l in doc.paragraphs:
    l = l.text
    l = l.split(',')
    if l[0] == '' or l[0] == ' ':
        continue
    elif len(l) < 2:
        print(l[0] + ' is not labeled')
        continue
    #print(l)
    counts = l[1]
    counts = counts.replace(' ', '')
    counts = counts.replace('\t', '')
    jpg = l[0].replace(' ', '')
    jpg += '.jpg'
    if (args.replace):
        jpg = jpg.replace('seg', 'raw')
        jpg = jpg.replace('jpg', 'jpeg')

    # will create the class folder in the output dir if does not exist
    if counts not in folders_created:
        if not os.path.exists(output_dir + counts):
            os.makedirs(output_dir + counts)
        folders_created.append(counts)

    # attempt to copy image over to their corresponding class in the output directory
    # will ignore if image does not exist
    try:
        copyfile(photos_dir + jpg, output_dir + counts + os.path.sep + identifier + '.' + jpg)
    except OSError as e:
        if missing_flag == False:
            print('Missing:')
            missing_flag = True
        e = str(e)
        e = e.split()
        print("\t" + " ".join(e[7:]).replace('\'', ''))
        continue
