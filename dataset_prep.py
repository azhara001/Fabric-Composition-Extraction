# -*- coding: utf-8 -*-
import os 
import json 
import pandas as pd
import rawpy
import numpy as np
from PIL import Image
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split 
import shutil
from datasets import load_dataset


def raw_to_rgb(path="../Dataset/"):
    counter = 1
    images_files = os.listdir(path)
    not_imported = []
    #   print(f"Number of images in the directory: {len(images_files)}")
    for img in tqdm(images_files,desc="No. of Images converted:",position=1,leave=False):
        format = img.split(".")[1]

        if img.split(".")[0]+"jpeg" or img.split(".")[0]+"jpg" in images_files:
           continue
        # if format == "jpeg" or format =="jpg" or format =="png": 
        #     continue # a jpeg/jpg/png file already exists 
    
        try:
            with rawpy.imread(path+img) as raw:
            # Access the image data
                rgb = raw.postprocess(use_auto_wb=True)
                im = Image.fromarray(rgb,mode='RGB')
            
                if im.size[0] < im.size[1]: # rotate image 
                    im = im.transpose(Image.ROTATE_90)
                path_save = path+img.split(".")[0]+'.jpg'
                im.save(path_save,quality=100,subsampling=0)
        except:
            not_imported.append(path+img)
        
    # print(f"{counter} image(s) converted to .jpeg format")
    # print('\n')
    # print(f"Following image path(s) not converted: {not_imported}")

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
            # 'description' : row['ds'], # commented out on: 03/18/24
            'composition_flag': row['Composition Flag'] if not pd.isna(row['Composition Flag']) else None #NaN is not a concept in json, unlike in python or javascript. None in Python corresponds to mull in json
        }
        
        # Convert the dictionary to a JSON string and add it to the list
        json_list.append(row_dict)

    # saving to file. 
    with open('Metadata.json', 'w',  encoding='utf-8') as f:
        json.dump(json_list, f, indent=4)

def jpeg_read(path="",meta_groundtruth = [], h=3024,w=4032,downsample=False,rgb_to_gray=True):
  
  """
  Imports images from a specified path folder and preprocesses it to return a numpy array with corressponding groundtruth

  Parameters:
  -----------
    path (str) : Path of the folder with images (images named as sample_id)
    meta_groundtruth (list) : Metadata groundtruth which is a list of dicts 
    h (int) : default height (number of rows) of the image array
    w (int) : default width (number of columns) of the image array 
    downsample (boolean) : downsamples the image by a factor of 2 if True
    rgb_to_gray (boolean) : converts rgb image into grayscale

    Returns:
    ---------
    images_array (list) : images
    ground_truth (list) : subsequent groundtruth (index matched)

  """

  files_jpeg = os.listdir(path) # Todo: use Pathlib for less verbose code! 
  assert len(files_jpeg) > 0 , "No files in the path folder provided"

  ground_truth = [] # a list storing ground-truth 
  images_array = [] # a list storing image files 

  gt_hashtable = {int(sample['sample_id']): sample for sample in meta_groundtruth} # creating a hash function for easy lookup with sample_id 

  if downsample:
    h,w = h//2,w//2 # changes the default width and height
  
  for i,file in enumerate(tqdm(files_jpeg,desc="Number of Images: ",position=0,leave=True)): #looping over the images

    # check if any other format encountered 
    if file.split('.')[1] != 'jpg' and 'jpeg' and 'png': 
      # print(f"Encountered a file without jpg, jpeg, or png extension: {file}")
      continue # loop-over
    
    sample_id = int(file.split('.')[0]) 
    #checks if image has corressponding groundtruth in gt_meta
    if sample_id not in gt_hashtable: 
      continue

    # reading the image using PIL
    if rgb_to_gray:
      im = Image.open(path+file).convert('L') 
    else:
      im = Image.open(path+file)
    x = np.array(im)
    
    # normalizing image 
    if x.max() == 255.0: # normalize image
      x = x/255

    # downsampling image
    if downsample:
      if rgb_to_gray:
        x = x[::2,::2]
      else:
        x = x[::2,::2,:]
      
    # rotation of landscape images into potrait 
    if x.shape[0] == w and x.shape[1] == h: # in landscape mode
      x = np.rot90(x)
 
    images_array.append(x)
    ground_truth.append(gt_hashtable[sample_id]) # append groundtruth

  return images_array, ground_truth

def create_folder_and_copy_images(image_paths, destination_folder="Dataset_HF/train/",split="train"):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    gt_keys = {}
    # Copy each image from the source folder to the destination folder
    for image_path in tqdm(image_paths,desc=split):
        source_path = "../Dataset/"+image_path
        sample_id, format = image_path.rsplit('/', 1)[-1].split('.')[0], image_path.rsplit('/', 1)[-1].split('.')[1]
        
        # if sample_id in gt_keys:
        #    print(f"repeated image: {sample_id}")
           
        gt_keys[sample_id] = format
        
        # Check if the source image exists
        if os.path.exists(source_path):
            # Extract the filename from the source image path
            filename = os.path.basename(image_path)
            
            # Construct the full path of the destination image
            destination_path = os.path.join(destination_folder, filename)
            
            # Copy the image to the destination folder
            shutil.copyfile(source_path, destination_path)
            
        else:
            print(f"Image '{image_path}' not found.")
    return gt_keys
    
def list_of_dicts_to_jsonlines(list_of_dicts=[], filename="Dataset_HF/train/metadata.jsonl"):
    with open(filename, 'w') as f:
        for item in list_of_dicts:
            json.dump(item, f)
            f.write('\n')


if __name__ == "__main__":

    directories_images = os.listdir("../Dataset")
    assert("Metadata.xlsx") in directories_images, "Metadata groundtruth file not available in the directory"
    gt_metadata_path = "../Dataset/Metadata.xlsx"
    directories_images.remove("Metadata.xlsx")

    img_files = []

    for folder in tqdm(directories_images,desc="Folders progress",position=0):  

        if re.search("Folder",folder) is not None: #making sure to access a custom folder dataset
            raw_to_rgb(path="../Dataset/"+folder+"/")
            img_files.extend([f"{folder}/{file}" for file in os.listdir(f"../Dataset/{folder}/")])

    if "Metadata.json" not in os.listdir():
        excel_to_json(gt_metadata_path)
    
    img_files = [i for i in img_files if i.lower().endswith((".jpeg", ".jpg", ".png"))] # all the images in the dataset 

    train_1, test = train_test_split(img_files, random_state=104,  test_size=0.10,  shuffle=True)
    train, eval = train_test_split(train_1, random_state=104,  test_size=0.20,  shuffle=True)
    # # tested till here (number of images in the folders match img_files ) 

    with open('Metadata.json') as f:
       groundtruth = json.load(f)
    
    # create folders for dataset splits for HuggingFace dataset creation
    gt_meta_train = create_folder_and_copy_images(train, destination_folder="Dataset_HF/train/")
    gt_meta_validation = create_folder_and_copy_images(eval, destination_folder="Dataset_HF/validation/",split="validation")
    gt_meta_test = create_folder_and_copy_images(test, destination_folder="Dataset_HF/test/",split="test")

    # jsonl creation for HF dataset 
    meta_data_train, meta_data_validation, meta_data_test = [],[],[]
    not_included = []

    for sample in groundtruth:
        check=0
        id = str(int(sample['sample_id']))
        
        if id in gt_meta_train:
           meta_data_train.append({'file_name':id+'.'+gt_meta_train[id],"text":sample})
           check=1
        
        if id in gt_meta_validation:
           meta_data_validation.append({'file_name':id+'.'+gt_meta_validation[id],"text":sample})
           check=1
        
        if id in gt_meta_test:
           meta_data_test.append({'file_name':id+'.'+gt_meta_test[id],"text":sample})
           check = 1
        
        if check == 0:
           not_included.append(id)
    
    meta_data_total = meta_data_train.copy()
    meta_data_total.extend(meta_data_validation)
    meta_data_total.extend(meta_data_test)

    list_of_dicts_to_jsonlines(meta_data_total, filename="Dataset_HF/train/metadata.jsonl")
    list_of_dicts_to_jsonlines(meta_data_total, filename="Dataset_HF/validation/metadata.jsonl")
    list_of_dicts_to_jsonlines(meta_data_total, filename="Dataset_HF/test/metadata.jsonl")

    # # # creating a HF format dataset
    dataset = load_dataset("imagefolder", data_dir="Dataset_HF")

    # push to hub
    dataset.push_to_hub("Fabric-Composition-Tags-v2",private=True)





#%%

# %%
