import wandb
import time
import argparse
import os
from damage_classifier.train import train

# https://wandb.ai/maria_rodriguez/Surgical_instruments_models_/reports/Choosing-a-Model-for-Detecting-Surgical-Instruments--VmlldzoxMjI4NjQ0
# https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/blob/main/setup/readme.md


def _get_params(exp_name,model_name,tune,policy,event,lr=1e-3,batch_size=128,epochs=10,frac=0.2):
    return dict(
        exp_name=exp_name,
        event=event,
        model_name=model_name,
        is_augment=False,
        lr=lr,
        batch_size=batch_size,
        do_finetune=tune,
        use_clr=policy,
        buffer_size=10,
        n_epochs=epochs,
        init_lr=1e-3,
        max_lr=1e-2,
        frac =frac
    )


def get_timestamp():
    t = time.localtime()
    return time.strftime('%Y-%m-%d %H:%M:%S', t)


def run_experiment(hyper_params,lr,batch,epochs,frac,is_wandb):
    cwd = os.getcwd()
    print("working dir:",cwd)
    
    output_path = os.path.join(cwd,"outputs","model")
    os.makedirs(f"{output_path}/model/", exist_ok=True)
    print("model output path:",output_path)

    if not is_wandb:
        # If you don't want your script to sync to the cloud
        os.environ['WANDB_MODE'] = 'offline'
    wandb.login()
    tags = []
    for event in hyper_params["events"]:
        tags.append(event)
        for model in hyper_params["models"]:
            tags.append(model)
            for tune in hyper_params["tuning"]:
                for policy in hyper_params["clr"]:
                    timestamp = int(time.time())
                    time.sleep(2)
                    exp_name = event + "_" + model
                    if tune:
                        exp_name = exp_name + '_fine_tuned'
                        if policy:
                            exp_name = exp_name + '_CLR'
                    else:
                        if policy:
                            exp_name = exp_name + '_CLR'

                    exp_name = exp_name + "_" + str(timestamp)
                    params = _get_params(exp_name, model, tune, policy,event,lr,batch,epochs,frac)
                    print(exp_name)

                    run = wandb.init(
                        project='disaster-damage-test',
                        notes="events specific run",
                        tags=tags,
                        config=params,
                        reinit=True
                    )
          
                    if is_wandb:
                        run_name = run.name
                        config = wandb.config
                        run.name = config.exp_name+ "-" + run_name
                    
                    rs = train(cwd,params['exp_name'],params['event'], params['model_name'], output_path,
                                params['is_augment'],params['lr'],params['batch_size'],params['do_finetune'],
                                params['use_clr'],params['buffer_size'],params['n_epochs'],params['init_lr'], 
                                params['max_lr'],params['frac'])
                    
                    if is_wandb:
                        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event",type=str,default='cross_event_ecuador')
    parser.add_argument("--model", type=str, default='efficientnet')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch",type=int, default=32)
    parser.add_argument("--finetune",action='store_true')
    parser.add_argument("--clr",action='store_true')
    parser.add_argument("--frac",type=float, default=0.2)
    parser.add_argument("--wandb",action='store_true')

    args = parser.parse_args()

    all_events = ['ecuador','nepal','matthew','ruby','gg']
    all_models = ['vgg16_fc','mobilenet','efficientnet']

    hyper_params = {
        "events": ['ruby'],
        "models": ['mobilenet'],
        "tuning": [args.finetune],
        "clr": [args.clr]
    }
    events = []
    models = []

    if args.event == 'all':
        hyper_params['events'] = all_events
    else:
        events.append(args.event)
        hyper_params['events'] = events

    if args.model == 'all':
        hyper_params['models'] = all_models
    else:
        models.append(args.model)
        hyper_params['models'] = models

    print(hyper_params)


    # TODO download data if not available
    # TODO Create demo app notebooks
    # TODO Test tensorflow version - layer.RandomFlip


    # Run training  experiment
    run_experiment(hyper_params, args.lr, args.batch, args.epochs,args.frac,args.wandb)





