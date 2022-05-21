import os
import torch
import time

def load_model(model,checkpoint_path,device=torch.device('cuda'),**kwargs):
    optimizer=None
    checkpoint=torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    if 'optimizer' in kwargs:
        optimizer.load_state_dict(checkpoint['optimizer'])
    epoch=checkpoint['epoch']
    return model,optimizer,epoch

def make_train_dir():
    save_directory = os.path.join('saved_model')
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    now=time.gmtime()
    time_str=f"{now.tm_mon}-{now.tm_mday} {now.tm_hour+8}:{now.tm_min}"
    #experiment_ Dir is used to store models
    experiment_dir = os.path.join(save_directory, time_str)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    #tb_dir is the tensorboard log file
    tb_dir=os.path.join('runs')
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    tb_dir=os.path.join(tb_dir,f'Train_{time_str}')
    
    print(f'{time_str}')

    return experiment_dir,tb_dir

def set_grad(net,choice):
    for param in net.parameters():
        param.requires_grad = choice

def save_model(model,optimizer,epoch,model_name,experiment_dir):
    checkpoint_path= os.path.join(experiment_dir,f'{model_name}_{epoch}.pth')
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
    torch.save(state, checkpoint_path)
