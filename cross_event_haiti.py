import os
import pandas as pd
import shutil
import random
import csv


cwd = os.getcwd()
print(cwd)
IMAGE_FOLDER = 'ASONAM17_Damage_Image_Dataset'
DATA_PATH = os.path.join(cwd,'data')
IMAGES_PATH = os.path.join(DATA_PATH,IMAGE_FOLDER)
HAITI_IMAGES_PATH = os.path.join(DATA_PATH,'haiti_eq')
DAMAGE_CSV_PATH = os.path.join(DATA_PATH,'damage_csv')

# Loop through copy to 
NEW_HAITI_IMAGES_PATH = os.path.join(IMAGES_PATH,'haiti_eq')
os.makedirs(NEW_HAITI_IMAGES_PATH,exist_ok=True)

print(NEW_HAITI_IMAGES_PATH)
def set_file_label(dir):
    if dir == 'none':
        return 0
    elif dir == 'mild':
        return 1
    else:
        return 2

data_raw_file = []
for dir in os.listdir(HAITI_IMAGES_PATH):
    dir_path =os.path.join(HAITI_IMAGES_PATH,dir)
    for image_file in os.listdir(dir_path):
        csvline = []
        label = set_file_label(dir)
        image = f"haiti_eq/{image_file}"
        csvline.append(image)
        csvline.append(int(label))
        shutil.copyfile(os.path.join(dir_path,image_file), os.path.join(NEW_HAITI_IMAGES_PATH,image_file))
        data_raw_file.append(csvline)
    

print("total",len(data_raw_file))
random.shuffle(data_raw_file)
split = int(len(data_raw_file)*0.6)
train = data_raw_file[:split]
valid = data_raw_file[split:]

#split dev and test
random.shuffle(valid)
split_valid = int(len(valid)*0.5)
dev = valid[:split_valid]
test = valid[split_valid:]

print("train:",len(train))
print("dev:",len(dev))
print("test:",len(test))

# create CSV in damage csv
def _save_train_file_as_csv(damage_csv_path,event_file, data_raw_file, file_type):
    os.makedirs(f"{damage_csv_path}/{event_file}/", exist_ok=True)
    if file_type == 'test':
        file_csv = 'test.csv'
    if file_type =='dev':
        file_csv = 'dev.csv'
    elif file_type =='train':
        file_csv = 'train.csv'
    file_path = os.path.join(damage_csv_path,event_file,file_csv)
    print(f"creating:{file_path}")

    with open(file_path, mode='w', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in data_raw_file:
            writer.writerow(row)
        f.close()

_save_train_file_as_csv(DAMAGE_CSV_PATH,'haiti',data_raw_file,'train')
_save_train_file_as_csv(DAMAGE_CSV_PATH,'haiti',data_raw_file,'dev')
_save_train_file_as_csv(DAMAGE_CSV_PATH,'haiti',data_raw_file,'test')
