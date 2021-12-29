import tensorflow as tf
from tensorflow.keras import optimizers, callbacks,models,layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

IMG_SIZE = 224
num_classes = 3


def get_vgg16_model(lr=0.001):
    tf.keras.backend.clear_session()

    pre_trained_model = VGG16(include_top=False,
                              weights='imagenet',
                              input_shape=(IMG_SIZE, IMG_SIZE, 3))

    pre_trained_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = pre_trained_model(x, training=False)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    vgg16_model = models.Model(inputs, outputs)
    vgg16_model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                        loss='sparse_categorical_crossentropy',
                        metrics=['acc'])

    vgg16_model.summary()

    return vgg16_model


def get_vgg16_fc2_model(lr=0.001):
    tf.keras.backend.clear_session()
    print(f"lr in model = {lr}")

    pre_trained_model = VGG16(include_top=True,
                              weights='imagenet',
                              input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # pre_trained_model.summary()
    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.layers[-2]
    print("output shape :", last_layer.output_shape)

    last_output = last_layer.output
    x = layers.Dropout(0.3)(last_output)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model2 = models.Model(pre_trained_model.input, outputs)
    model2.compile(optimizer=optimizers.Adam(learning_rate=lr),
                   loss='sparse_categorical_crossentropy',
                   metrics=['acc'])

    model2.summary()

    return model2



def get_efficient_model(lr=0.001):
    tf.keras.backend.clear_session()
    print(f"lr in model = {lr}")
    pre_trained_model = EfficientNetB0(include_top=False,
                                       weights='imagenet',
                                       input_shape=(IMG_SIZE, IMG_SIZE, 3))

    pre_trained_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs * 255.0)
    x = pre_trained_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    efficient_model = models.Model(inputs, outputs)
    efficient_model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                            loss='sparse_categorical_crossentropy',
                            metrics=['acc'])

    efficient_model.summary()

    return efficient_model



def get_mobilenet_model(lr=0.001):
    tf.keras.backend.clear_session()
    print(f"lr in model = {lr}")
    pre_trained_model = MobileNetV2(include_top=True,
                                    weights='imagenet',
                                    input_shape=(IMG_SIZE, IMG_SIZE, 3))

    pre_trained_model.trainable = False

    # inputs = layers.Input(shape=(IMG_SIZE,IMG_SIZE,3))
    # x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    # x = pre_trained_model(x,training= False)
    # x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Dropout(0.2)(x)
    # outputs = layers.Dense(num_classes,activation='softmax')(x)

    # mobilenet_model = models.Model(inputs,outputs)
    # mobilenet_model.compile(optimizer = optimizers.Adam(learning_rate=lr),
    # loss='sparse_categorical_crossentropy',
    # metrics=['acc'])
    # pre_trained_model.summary()

    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.layers[-2]
    print("output shape :", last_layer.output_shape)

    last_output = last_layer.output
    x = layers.Dropout(0.3)(last_output)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model2 = models.Model(pre_trained_model.input, outputs)
    model2.compile(optimizer=optimizers.Adam(learning_rate=lr),
                   loss='sparse_categorical_crossentropy',
                   metrics=['acc'])

    model2.summary()

    return model2
