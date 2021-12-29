import os
from damage_classifier.data.download import download_images, extract_images
from damage_classifier.data.download import save_all_files
from damage_classifier.train import train


target_path = '../data/'
images_path = '../data/ASONAM17_Damage_Image_Dataset/'
damage_csv_path = '../data/damage_csv'
images_file_name = "ASONAM17_Damage_Image_Dataset.tar.gz"
event_files = ['ecuador','nepal','matthew','ruby','gg']

# check if folder or images already exits first
if not os.path.isdir(images_path):
    print("images not folder found")
    print("download and extracting images data")
    filename = download_images(target_path)
    extract_images(target_path=target_path, filename=filename)
else:
    print("images folder found")

# create damage_csv and convert all files path and labels to csv
if os.path.isdir(images_path):
    for event_file in event_files:
        save_all_files(damage_csv_path,images_path,event_file)
