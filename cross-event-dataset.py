import os
import pandas as  pd

damage_folder = 'damage_csv'
event_files = ['ecuador','nepal','matthew','ruby','gg']

cwd = os.getcwd()
print("working dir", cwd)
target_path = os.path.join(cwd, 'data')
damage_csv_path = os.path.join(cwd, 'data', damage_folder)


def create_cross_event_dataset(cross_events,test_event,train_frac=0.6):
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


if __name__ == '__main__':
        
    eq_events = ['nepal','gg','ecuador']
    create_cross_event_dataset(eq_events,'ecuador')

    typhoon_events = ['ruby','gg','matthew']
    create_cross_event_dataset(typhoon_events,'matthew')
