from damage_classifier.train import train
import wandb
import time


#events = ['ecuador','nepal','matthew','ruby','gg']
#models = ['vgg16_fc','mobilenet','efficientnet']
#tuning= [True,False]

events = ['ecuador','nepal']
models = ['mobilenet','efficientnet']
tuning = [False]

# TODO Create experiment list as a dict
# TODO how to tag and name the experiments
# TODO add command line interface
# TODO set wandb run name automotically


def _get_params(exp_name,model_name,tune,event):
    return dict(
        exp_name=exp_name,
        event=event,
        model_name=model_name,
        is_augment=False,
        lr=1e-3,
        batch_size=128,
        do_finetune=tune,
        use_clr=False,
        buffer_size=10,
        n_epochs=3,
        init_lr=1e-3,
        max_lr=1e-2
    )


wandb.login()

def get_timestamp():
    t = time.localtime()
    return time.strftime('%Y-%m-%d %H:%M:%S', t)


for event in events:
    for model in models:
        for tune in tuning:
            timestamp = int(time.time())
            time.sleep(2)
            if tune:
                exp_name = event + "_" + model + '_fine_tuned' + '_' + str(timestamp)
            else:
                exp_name = event + "_" + model + '_' + str(timestamp)
            params = _get_params(exp_name, model, tune, event)
            print(exp_name)

            run = wandb.init(
                project='disaster-damage-assessment',
                notes="events specific run",
                tags=["test"],
                config=params,
                reinit=True
            )
            run_name = run.name
            config = wandb.config
            run.name = config.exp_name + "-" + run.name

            rs = train(config.exp_name, config.event, config.model_name, config.is_augment, config.lr,
                       config.batch_size, config.do_finetune, config.use_clr, config.buffer_size, config.n_epochs,
                       config.init_lr, config.max_lr)
            run.finish()






