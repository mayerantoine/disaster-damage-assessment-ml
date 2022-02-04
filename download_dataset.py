import os
from damage_classifier.data.download import download_images, extract_images
from damage_classifier.data.download import save_all_files
import pandas as pd

def download_data(cwd):
    print("working dir", cwd)
    target_path = os.path.join(cwd, 'data')
    images_path = os.path.join(cwd, 'data', images_folder)
    damage_csv_path = os.path.join(cwd, 'data', damage_folder)

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
            save_all_files(damage_csv_path, images_path, event_file)


def create_cross_event_dataset(cwd,cross_events,test_event,train_frac=0.6):
    damage_csv_path = os.path.join(cwd, 'data', damage_folder)
    train_cross_df =[]
    for event in cross_events:
        train_files = []
        for file in ['train.csv','dev.csv','test.csv']:
            df = pd.read_csv(os.path.join(damage_csv_path,event,file),names=['path','label'])
            # print(len(df.columns))
            # print(f"{event} : {file} : {len(df)}")
            train_files.append(df)
        event_files_merge = pd.concat(train_files,axis=0)
        print(event,len(event_files_merge))

        if event != test_event :
            # take 60% from each event for training
            df_event = event_files_merge.sample(frac=train_frac, random_state=42)
        else:
            size = len(event_files_merge)
            split = int(train_frac * size)
            df_event = event_files_merge[:split]
            df_dev_test = event_files_merge[split:]
            dev = df_dev_test[int(0.5*len(df_dev_test)):]
            test = df_dev_test[:int(0.5*len(df_dev_test))]

        train_cross_df.append(df_event)

    train = pd.concat(train_cross_df, axis=0)
    print("train data:", len(train))
    print("dev data:", len(dev))
    print("test data:", len(test))

    event_folder = "cross_event_" + test_event
    train.to_csv(os.path.join(damage_csv_path,event_folder,'train.csv'),index=False)
    dev.to_csv(os.path.join(damage_csv_path, event_folder, 'dev.csv'), index=False)
    test.to_csv(os.path.join(damage_csv_path, event_folder, 'test.csv'), index=False)


if __name__ == '__main__' :
    images_folder = 'ASONAM17_Damage_Image_Dataset'
    damage_folder = 'damage_csv'
    images_file_name = "ASONAM17_Damage_Image_Dataset.tar.gz"
    event_files = ['ecuador','nepal','matthew','ruby','gg']

    cwd = os.getcwd()
    download_data(cwd)

    eq_events = ['nepal','gg','ecuador']
    create_cross_event_dataset(eq_events,'ecuador')

    typhoon_events = ['ruby','gg','matthew']
    create_cross_event_dataset(typhoon_events,'matthew')

