import requests
import tarfile
import wget
import csv
import os


def download_images(target_path):
    url = 'https://crisisnlp.qcri.org/data/ASONAM17_damage_images/ASONAM17_Damage_Image_Dataset.tar.gz'
    target_path = target_path
    filename = wget.download(url=url, out=target_path)
    return filename


def extract_images(target_path, filename='ASONAM17_Damage_Image_Dataset.tar.gz'):

    print(target_path + filename)

    tar = tarfile.open(target_path+filename)
    tar.extractall(path=target_path)
    tar.close()


def _get_train_files_by_event(images_path, event_file):
    files = []
    for f in os.listdir(images_path):
        if not f.startswith('.'):
            if f.startswith(event_file) and os.path.isfile(os.path.join(images_path, f)):
                path = os.path.join(images_path, f)
                files.append(path)
    return files


def _get_train_file_data(file_path):
    data_raw_file = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            csvline = []
            image, label = line.split()
            csvline.append(image)
            csvline.append(int(label))
            data_raw_file.append(csvline)

    return data_raw_file


def _save_train_file_as_csv(damage_csv_path,event_file, data_raw_file, file_path):
    os.makedirs(f"{damage_csv_path}/{event_file}/", exist_ok=True)
    if file_path.endswith('.dev'):
        file_csv = 'dev.csv'
    elif file_path.endswith('.train'):
        file_csv = 'train.csv'
    elif file_path.endswith('.test'):
        file_csv = 'test.csv'

    with open(f"{damage_csv_path}/{event_file}/{file_csv}", mode='w', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in data_raw_file:
            writer.writerow(row)
        f.close()


def save_all_files(damage_csv_path,images_path, event_file):
    """
        Save as CSV all data for an event
    """

    files = _get_train_files_by_event(images_path, event_file)
    print(files)
    for f in files:
        data_file = _get_train_file_data(f)
        _save_train_file_as_csv(damage_csv_path,event_file, data_file, f)




