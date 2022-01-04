from damage_classifier.train import train
import wandb

events = ['ecuador','nepal','matthew','ruby','gg']
models = ['vgg16','vgg16_fc','mobilenet','efficientnet']

# TODO Create experiment list as a dict
# TODO how to tag and name the experiments
# TODO add command line interface
# TODO set wandb run name automotically


wandb.login()
params = dict(
    exp_name='efficientnet no augmentation',
    event='ecuador',
    model_name='efficientnet',
    is_augment=False,
    lr=5 * 1e-3,
    batch_size=128,
    do_finetune=False,
    use_clr=False,
    buffer_size=10,
    n_epochs=5,
    init_lr=1e-3,
    max_lr=1e-2
)

wandb.init(
    project='disaster-damage-assessment',
    notes="events specific run",
    tags=["test"],
    config=params
)

config = wandb.config

rs = train(config.exp_name,config.event,config.model_name,config.is_augment,config.lr,
           config.batch_size,config.do_finetune,config.use_clr,config.buffer_size,config.n_epochs,config.init_lr,config.max_lr)


#rs = train(exp_name,event,model_name,is_augment=False,lr=0.001,batch_size=128,do_finetune=False,
                #use_clr= False,buffer_size=10,n_epochs=1,init_lr=1e-3,max_lr=1e-2)


