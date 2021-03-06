import tensorflow as tf
import gc
import numpy as np
import pandas as pd
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from damage_classifier.models.models import get_mobilenet_model,get_efficient_model,get_vgg16_model,get_vgg16_fc2_model
from damage_classifier.preprocess import create_dataset

import wandb
from wandb.keras import WandbCallback
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,balanced_accuracy_score
from sklearn.metrics import classification_report

damage_path = '../data/damage_csv'

# TODO experimentation with weights and biases
# TODO add inference on test dataset
# TODO add compute metrics on validation and test dataset
# TODO log confusion_matrix and precision-recall curve with wandb
# TODO add command line interface



def batch_predict(ds, model, validation_steps):
    y_true = np.concatenate([y for x,y in ds.take(validation_steps)])
    preds = np.concatenate([model.predict(image_batch) for image_batch,label in ds.take(validation_steps)])
    preds_labels = np.argmax(preds, axis=-1)

    results = pd.DataFrame({'y_true': y_true,
                            'y_pred': preds_labels})

    return results


def compute_metrics(pred):
    labels = pred.y_true
    preds = pred.y_pred
    acc = accuracy_score(labels, preds)
    # bal_acc = balanced_accuracy_score(labels, preds)
    precision = precision_score(labels,preds,average='weighted',zero_division=0)
    recall = recall_score(labels,preds,average='weighted',zero_division=0)
    f1 = f1_score(labels,preds,average='weighted',zero_division=0)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def subplot_learning_curve(model_name,history):
    #plt.clf()
    plt.figure(figsize=(15,5))
    for i,metric in enumerate(['acc','loss']):
        plt.subplot(1,2,i+1)
        plt.plot(history.history[metric])
        plt.plot(history.history['val_' + metric])
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.legend((metric, 'val_' + metric))
        plt.title(model_name + ": Learning curve " + metric + " vs " + 'val_' + metric)
    plt.show()

    return plt


def finetune_model(exp_name,lr ,model_name ,output_path,train_batches ,valid_batches ,initial_epoch,
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

        check = tf.keras.callbacks.ModelCheckpoint(f'{output_path}/{exp_name}.h5', save_best_only=True)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc'])
        ds = valid_batches.take(1)
        ds_filter = ds.map(lambda x, y: (x[:5], y[:5]))
        class_names = ['none', 'mild', 'severe']

        print()
        print("Training..................")
        history = model.fit(train_batches,
                            initial_epoch=initial_epoch,
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=valid_batches,
                            validation_steps=validation_steps,
                            callbacks=[check,WandbCallback(input_type='images',
                                                                training_data=ds_filter,
                                                                    labels=class_names)])
        print()
        #print('loading best weights model')
        #model = models.load_model(f'{output_path}/{exp_name}.h5')

    return history, model


def train(cwd,exp_name, event, model_name, output_path,is_augment=False, lr=0.001, batch_size=32, do_finetune=False,
                   use_clr=False, buffer_size=10, n_epochs=20, init_lr=1e-3, max_lr=1e-2,frac=0.2):
    print(f"******************{exp_name}*********************************")
    print(f"model_name ={model_name}")
    print(f"data augmentation ={is_augment}")
    print(f"event ={event}")
    print(f"finetuning ={do_finetune}")
    print(f"lr ={lr}")
    print()

    gc.collect()

    print(f"Creating dataset.....")
    train_batches, valid_batches, test_batches, steps_per_epoch, validation_steps = create_dataset(cwd,
                                                                                                   event,
                                                                                                   is_augment=is_augment,
                                                                                                   batch_size=batch_size,
                                                                                                   frac=frac,
                                                                                                   IMG_SIZE= 300)

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

    check = tf.keras.callbacks.ModelCheckpoint(f'{output_path}/{exp_name}.h5', save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)

    # data for wandb callback inference
    ds = valid_batches.take(1)
    ds_filter = ds.map(lambda x, y: (x[:5], y[:5]))
    class_names = ['none', 'mild', 'severe']

    print()
    print("Training..................")
    history = model.fit(train_batches,
                        epochs=n_epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=valid_batches,
                        validation_steps=validation_steps,
                        callbacks=[check,WandbCallback(input_type='images',
                                                                training_data=ds_filter,
                                                                    labels=class_names)])

    print()
    subplot_learning_curve(model_name, history)
    # wandb.log({"learning_curve": wandb.Image(plt_learning_curve)})

    # TODO Load model correctly
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

    # log and calculate metrics after training if not finetuning
    if not do_finetune:
        wandb.log({"train_acc": results_train['acc'],"valid_acc": results_test['acc']})

        #Predictions
        print()
        print(f"Run prediction and Log metrics.........")
        predictions = batch_predict(test_batches, model, validation_steps)
        metrics = compute_metrics(predictions)
        wandb.log(metrics)

        # Class report
        # print()
        # print(f"Class report metrics........")
        # classes_report = classification_report(predictions.y_true, predictions.y_pred,
        #                                        target_names=class_names, output_dict=True,zero_division=0)
        #wandb.log(classes_report)

    if do_finetune:

        if not use_clr:
            # LR finetuning when not using CLR
            lr = lr * 1e-2
        print()
        print(f"******Fine tuning***********************")
        history, model = finetune_model(exp_name=exp_name,lr=lr, model_name=model_name, output_path=output_path,
                                        train_batches=train_batches,valid_batches=valid_batches, initial_epoch=n_epochs,
                                        epochs=2 * n_epochs,steps_per_epoch=steps_per_epoch,
                                        validation_steps=validation_steps,
                                        use_clr=use_clr, init_lr=init_lr * 1e-2, max_lr=max_lr * 1e-2, model=model)
        print()
        # subplot_learning_curve(model_name + "_fintuned", history)

        results_train = model.evaluate(train_batches, steps=steps_per_epoch, return_dict=True)
        results_test = model.evaluate(valid_batches, steps=validation_steps, return_dict=True)

        print()
        print(f"Training finetune accuracy: {results_train['acc']}")
        print(f"Validation finetune accuracy: {results_test['acc']}")
        wandb.log({"train_acc": results_train['acc'], "valid_acc": results_test['acc']})

        # Predictions
        print()
        print(f"Run prediction and Log metrics.........")
        predictions = batch_predict(test_batches, model, validation_steps)
        metrics = compute_metrics(predictions)
        wandb.log(metrics)

        # Class report
        # print()
        # print(f"Class report metrics........")
        # classes_report = classification_report(predictions.y_true, predictions.y_pred,
        #                                        target_names=class_names, output_dict=True,zero_division=0)
        # wandb.log(classes_report)

    return results_train['acc'], results_test['acc'], model