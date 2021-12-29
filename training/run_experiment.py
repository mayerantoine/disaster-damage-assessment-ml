import os
from damage_classifier.data.download import download_images, extract_images
from damage_classifier.data.download import save_all_files
from damage_classifier.train import train

events = ['ecuador','nepal','matthew','ruby','gg']
models = ['vgg16','vgg16_fc','mobilenet','efficientnet']


exp_name = 'efficient no augmentation'
event= 'ruby'
model_name = 'mobilenet'

rs = train(exp_name,event,model_name,is_augment=False,lr=0.001,batch_size=128,do_finetune=False,
                use_clr= False,buffer_size=10,n_epochs=1,init_lr=1e-3,max_lr=1e-2)


