import os
import pandas as pd
import math
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks,models,layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

images_folder = 'ASONAM17_Damage_Image_Dataset'
damage_folder = 'damage_csv'


def create_dataset(cwd, event, is_augment=False, batch_size=32, 
                                buffer_size=100,frac=0.2,IMG_SIZE=300):

    images_path = os.path.join(cwd,'data',images_folder)
    damage_path = os.path.join(cwd,'data',damage_folder)
    label_path = os.path.join(damage_path, event)

    data_augmentation_layer = tf.keras.Sequential([
                                  layers.RandomFlip("horizontal_and_vertical"),
                                  layers.RandomRotation(0.2),
                                  layers.RandomCrop(IMG_SIZE,IMG_SIZE),
                                  layers.RandomContrast(factor=0.8)
        ])

    print("images path:", images_path)
    print("damage path:", damage_path)
    print("label path:",label_path)


    img_gen = ImageDataGenerator(rescale=1 / 255.0, )

    train_df = pd.read_csv(os.path.join(label_path, 'train.csv'), header=None)
    train_df.columns = ['path', 'label']
    train_df_sample = train_df.sample(frac=frac,random_state=42)

    train_gen = img_gen.flow_from_dataframe(dataframe=train_df_sample,
                                            directory=images_path,
                                            x_col='path',
                                            y_col='label',
                                            class_mode='raw',
                                            batch_size=batch_size,
                                            target_size=(IMG_SIZE, IMG_SIZE))

    valid_df = pd.read_csv(os.path.join(label_path, 'dev.csv'), header=None)
    valid_df.columns = ['path', 'label']
    valid_df_sample = valid_df.sample(frac=frac,random_state=42)
    

    valid_gen = img_gen.flow_from_dataframe(dataframe=valid_df_sample,
                                            directory=images_path,
                                            x_col='path',
                                            y_col='label',
                                            class_mode='raw',
                                            shuffle=False,
                                            batch_size=batch_size,
                                            target_size=(IMG_SIZE, IMG_SIZE))

    test_df = pd.read_csv(os.path.join(label_path, 'test.csv'), header=None)
    test_df.columns = ['path', 'label']
    test_df_sample = test_df.sample(frac=frac,random_state=42)
    

    test_gen = img_gen.flow_from_dataframe(dataframe=test_df_sample,
                                           directory=images_path,
                                           x_col='path',
                                           y_col='label',
                                           class_mode='raw',
                                           shuffle=False,
                                           batch_size=batch_size,
                                           target_size=(IMG_SIZE, IMG_SIZE))

    # Now we're converting our ImageDataGenerator to Dataset

    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_gen,  # Our generator
        output_types=(tf.float32, tf.float32),  # How we're expecting our output dtype
        output_shapes=([None, IMG_SIZE, IMG_SIZE, 3], [None, ])  # How we're expecting our output shape
    )

    valid_dataset = tf.data.Dataset.from_generator(
        lambda: valid_gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, IMG_SIZE, IMG_SIZE, 3], [None, ])
    )

    test_dataset = tf.data.Dataset.from_generator(
        lambda: test_gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, IMG_SIZE, IMG_SIZE, 3], [None, ])
    )

    if is_augment:
        print("data augmentation....")
        train_dataset = train_dataset.map(lambda x, y: (data_augmentation_layer(x, training=True), y),
                                          num_parallel_calls=tf.data.AUTOTUNE)

    compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))
    steps_per_epoch = compute_steps_per_epoch(len(train_df_sample))
    validation_steps = compute_steps_per_epoch(len(valid_df_sample))

    print(f"steps_per_epochs: {steps_per_epoch}")
    print(f"validations_steps: {validation_steps}")

    train_dataset = train_dataset.prefetch(buffer_size=buffer_size)
    valid_dataset = valid_dataset.prefetch(buffer_size=buffer_size)

    return train_dataset, valid_dataset, test_dataset, steps_per_epoch, validation_steps

