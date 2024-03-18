# -*- coding: utf-8 -*-
import os 
import json 
import pandas as pd
import rawpy
import numpy as np
from PIL import Image
from tqdm import tqdm
import re


def raw_to_rgb(path="../Dataset/"):
  counter = 1
  images_files = os.listdir(path)
  not_imported = []

#   print(f"Number of images in the directory: {len(images_files)}")

  for img in tqdm(images_files,desc="No. of Images converted:",position=1,leave=False):
    format = img.split(".")[1]

    if format == "jpeg" or format =="jpg" or format =="png": 
       continue # a jpeg/jpg/png file already exists 
    
    try:
        with rawpy.imread(path+img) as raw:
        # Access the image data
            rgb = raw.postprocess(use_auto_wb=True)
            im = Image.fromarray(rgb,mode='RGB')
            path_save = path+img.split(".")[0]+'.jpeg'
            im.save(path_save,quality=100,subsampling=0)
    except:
        not_imported.append(path+img)
        
  
    #   print(f"{counter} image(s) converted to .jpeg format")
    #   print(f"Following image path(s) not converted: {not_imported}")
    return None


def excel_to_json(import_path="../Dataset/Metadata.xlsx"):
    """ groundtruth creation """

    dtype_spec = {
        'sample_id': int,
        'composition_raw_text': str,
        'ds': str,
        'Composition Flag': str,
        'cotton': float, 
        'polyester': float,
        'elastane': float,
        'nylon': float,
        'viscose': float,
        'rayon': float,
        'modal': float,
        'tencel': float,
        'cupro': float,
        'micromodal': float,
        'nylon_6': float,
        'nylon_66': float
        }
   
    df = pd.read_excel(import_path)
    print(df.head())
    json_list = []  # stores data as json string.

    for index, row in df.iterrows():
        composition = {
            'cotton': row['cotton'],
            'polyester': row['polyester'],
            'elastane': row['elastane'],
            'nylon': row['nylon'],
            'viscose': row['viscose'],
            'rayon': row['rayon'],
            'modal': row['modal'],
            'tencel': row['tencel'],
            'cupro': row['cupro'],
            'micromodal': row['micromodal'],
            'nylon_6': row['nylon_6'],
            'nylon_66': row['nylon_66']
            
        }
        if pd.isna(row['Raw Text']) or row['SUM'] == 0:  # skip the sample if SUM (sume of percentages) is 0 or Raw Text is Nan This indicates problem in reading or missing data.
            continue
        
        # Remove materials name and value if value is 0. 
        composition = {key: value for key, value in composition.items() if value > 0}
        
        # Create the dictionary for the current row
        row_dict = {
            'sample_id': row['sample_id'],
            'composition': composition,
            'composition_raw_text': row['Raw Text'] ,
            'description' : row['ds'], 
            'composition_flag': row['Composition Flag'] if not pd.isna(row['Composition Flag']) else None #NaN is not a concept in json, unlike in python or javascript. None in Python corresponds to mull in json
        }
        
        # Convert the dictionary to a JSON string and add it to the list
        json_list.append(row_dict)

    # saving to file. 
    with open('Metadata.json', 'w',  encoding='utf-8') as f:
        json.dump(json_list, f, indent=4)


if __name__ == "__main__":

    directories_images = os.listdir("../Dataset")
    assert("Metadata.xlsx") in directories_images, "Metadata groundtruth file not available in the directory"
    gt_metadata_path = "../Dataset/Metadata.xlsx"
    directories_images.remove("Metadata.xlsx")

    img_files = []

    for folder in tqdm(directories_images,desc="Folders progress",position=0):  
       if re.search("Folder",folder) is not None: #making sure to access a custom folder dataset
        raw_to_rgb(path="../Dataset/"+folder+"/")
        img_files.extend(os.listdir("../Dataset/"+folder+"/"))

    if "Metadata.json" not in os.listdir():
        excel_to_json(gt_metadata_path)
    
    img_files = [i for i in img_files if i.lower().endswith((".jpeg", ".jpg", ".png"))] # all the images in the dataset 



   
