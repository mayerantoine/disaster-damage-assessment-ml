import os
import pandas as  pd

damage_folder = 'damage_csv'
event_files = ['ecuador','nepal','matthew','ruby','gg']

cwd = os.getcwd()
print("working dir", cwd)
target_path = os.path.join(cwd, 'data')
damage_csv_path = os.path.join(cwd, 'data', damage_folder)



def create_cross_event_dataset(cross_events,test_event):
    cross_df =[]
    for event in cross_events:
        train_files = []
        for file in ['train.csv','dev.csv','test.csv']:
            df = pd.read_csv(os.path.join(damage_csv_path,event,file),names=['path','label'])
            # print(len(df.columns))
            print(f"{event} : {file} : {len(df)}")
            train_files.append(df)
        train_files_merge = pd.concat(train_files,axis=0)
        print(train_files_merge)

        if event == test_event:
            size = len(train_files_merge)
            split = int(0.5 * size)
            dev = train_files_merge[:split]
            test = train_files_merge[split:]

            print(f"Dev: {len(dev)}")
            print(f"test: {len(test)}")

        else:
            print(f"{event}:{len(train_files_merge)}")
            cross_df.append(train_files_merge)

    train = pd.concat(cross_df,axis=0)
    print("train data:",len(train))

    event_folder = "cross_event_"+test_event

    os.makedirs(os.path.join(damage_csv_path,event_folder),exist_ok=True)
    train.to_csv(os.path.join(damage_csv_path,event_folder,'train.csv'),index=False)
    dev.to_csv(os.path.join(damage_csv_path, event_folder, 'dev.csv'), index=False)
    test.to_csv(os.path.join(damage_csv_path, event_folder, 'test.csv'), index=False)


if __name__ == '__main__':
        
    eq_events = ['nepal','gg','ecuador']
    create_cross_event_dataset(eq_events,'ecuador')

    typhoon_events = ['ruby','gg','matthew']
    create_cross_event_dataset(typhoon_events,'matthew')