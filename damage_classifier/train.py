import tensorflow as tf
import gc
import tensorflow_addons as tfa
from tensorflow.keras import optimizers, callbacks,models,layers
import matplotlib.pyplot as plt
from damage_classifier.models.models import get_mobilenet_model,get_efficient_model,get_vgg16_model,get_vgg16_fc2_model
from damage_classifier.preprocess import create_dataset

damage_path = '../data/damage_csv'


def subplot_learning_curve(model_name,history):
    #plt.clf()
    plt.figure(figsize=(10,5))
    for i,metric in enumerate(['acc','loss']):
        plt.subplot(1,2,i+1)
        plt.plot(history.history[metric])
        plt.plot(history.history['val_' + metric])
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.legend((metric, 'val_' + metric))
        plt.title(model_name + ": Learning curve " + metric + " vs " + 'val_' + metric)
    plt.show()


def finetune_model(lr ,model_name ,train_batches ,valid_batches ,initial_epoch,
                   epochs, steps_per_epoch ,validation_steps ,use_clr=False ,init_lr=1e-3 ,max_lr=1e-2 ,model=None):
    print(f"finetuning lr ={lr}")
    print(f"finetuning epochs ={epochs}")
    print(f"init LR epochs ={init_lr}")
    print(f"max LR epochs ={max_lr}")

    if use_clr:
        print("using cyclical LR for finetuning")
        clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=init_lr,
                                                  maximal_learning_rate=max_lr,
                                                  scale_fn=lambda x: 1/ (1. ** (x - 1)),
                                                  step_size=2 * steps_per_epoch)
        lr = clr

    if model:
        for layer in model.layers:
            layer.trainable = True

        check = callbacks.ModelCheckpoint(f'../outputs/model/{model_name}.h5', save_best_only=True)
        early_stop = callbacks.EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)

        model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc'])

        print()
        print("Training..................")
        history = model.fit(train_batches,
                            initial_epoch=initial_epoch,
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=valid_batches,
                            validation_steps=validation_steps,
                            callbacks=[early_stop])

    return history, model


def train(exp_name, event, model_name, is_augment=False, lr=0.001, batch_size=32, do_finetune=False,
                   use_clr=False, buffer_size=10, n_epochs=20, init_lr=1e-3, max_lr=1e-2):
    print(f"******************{exp_name}*********************************")
    print(f"model_name ={model_name}")
    print(f"data augmentation ={is_augment}")
    print(f"event ={event}")
    print(f"finetuning ={do_finetune}")
    print(f"lr ={lr}")
    print()

    gc.collect()

    print(f"Creating dataset.....")
    train_batches, valid_batches, test_batches, steps_per_epoch, validation_steps = create_dataset(damage_path,
                                                                                                   event,
                                                                                                   is_augment=is_augment,
                                                                                                   batch_size=batch_size)

    if use_clr:
        print("using cyclical LR for training")
        clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=init_lr,
                                                  maximal_learning_rate=max_lr,
                                                  scale_fn=lambda x: 1 / (1. ** (x - 1)),
                                                  step_size=2 * steps_per_epoch)
        lr = clr

    print("Model architecture...........")
    if model_name == 'vgg16':
        model = get_vgg16_model(lr=lr)
    if model_name == 'vgg16_fc2':
        model = get_vgg16_fc2_model(lr=lr)
    elif model_name == 'efficientnet':
        model = get_efficient_model(lr=lr)
    elif model_name == 'mobilenet':
        model = get_mobilenet_model(lr=lr)

    check = callbacks.ModelCheckpoint(f'../outputs/model/{model_name}.h5', save_best_only=True)
    early_stop = callbacks.EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)

    print()
    print("Training..................")
    history = model.fit(train_batches,
                        epochs=n_epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=valid_batches,
                        validation_steps=validation_steps,
                        callbacks=[check, early_stop])

    print()
    subplot_learning_curve(model_name, history)

    print()
    # print('loading best weights model')
    # model = models.load_model(f'./model/{model_name}.h5')

    print()
    print(f"Run evaluation.........")

    results_train = model.evaluate(train_batches, steps=steps_per_epoch, return_dict=True)
    results_test = model.evaluate(valid_batches, steps=validation_steps, return_dict=True)

    print()
    print(f"Training accuracy: {results_train['acc']}")
    print(f"Validation accuracy: {results_test['acc']}")

    if do_finetune:

        if not use_clr:
            # LR finetuning when not using CLR
            lr = lr * 1e-2
        print()
        print(f"******Fine tuning***********************")
        history, model = finetune_model(lr=lr, model_name=model_name, train_batches=train_batches,
                                        valid_batches=valid_batches, initial_epoch=n_epochs, epochs=2 * n_epochs,
                                        steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                                        use_clr=use_clr, init_lr=init_lr * 1e-2, max_lr=max_lr * 1e-2, model=model)
        print()
        subplot_learning_curve(model_name + "_fintuned", history)

        results_train = model.evaluate(train_batches, steps=steps_per_epoch, return_dict=True)
        results_test = model.evaluate(valid_batches, steps=validation_steps, return_dict=True)

        print()
        print(f"Training finetune accuracy: {results_train['acc']}")
        print(f"Validation finetune accuracy: {results_test['acc']}")

    return results_train['acc'], results_test['acc'], model