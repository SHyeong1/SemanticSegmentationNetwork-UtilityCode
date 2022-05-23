import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .loss import SegmentationLosses
import sys
import time
from utils import metrics
import cv2
from tifffile import imread
from tifffile import imsave
import tqdm
import glob
import pprint

class TrainValidTool(object):
    def __init__(self,model,cfg,train_dataset=None,valid_dataset=None):
        self.model=model
        self.cfg=cfg
        self.train_dataset=train_dataset
        self.valid_dataset=valid_dataset
        
        self.device=torch.device(f'cuda')
        if train_dataset is not None:
            self.train_loader=DataLoader(dataset=self.train_dataset,batch_size=cfg.TRAIN.BatchSize,shuffle=True)
        if valid_dataset is not None:
            self.valid_loader=DataLoader(dataset=self.valid_dataset,batch_size=cfg.VALID.BatchSize,shuffle=True)
        
        
    
    #This function converts a batch obtained through the model into a labeled image 
    #with the type of numpy (the prediction graph is the confidence graph of nclass channel)
    def _get_batch_pre_label(self,batch_predict):
        batch_pred_labels=[]
        for predict in batch_predict:
            predict=F.softmax(predict)
            pred_label=predict.max(0).indices
            batch_pred_labels.append(pred_label.cpu().numpy())
        batch_pred_labels=np.array(batch_pred_labels)
        return batch_pred_labels

    #This function converts the label image into a binary gray image
    def _label2gray(self,label):
        gray=np.zeros((label.shape))
        for i in range(self.cfg.NumClass):
            gray[label==i]=i*(255//(self.cfg.NumClass-1))
        return gray.astype(np.uint8)

    #This function loads the model for the given path
    def _load_pretrained_model(self,model,optimizer,checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint=torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch=checkpoint['epoch']
        else:
            epoch=0
        return model,optimizer,epoch
    

    #This function writes the mask and predict of a batch in a same pic according to 
    #the path and prediction results stored in the prediction map
    #Write to the specified path
    def _write_batch_predict_mask(self,batch_predict,batch_mask,predict_dir,batch_name):
        if not os.path.exists(predict_dir):
            os.mkdir(predict_dir)
        compare_img=np.empty((batch_predict.shape[1],batch_predict.shape[2]*2+10),dtype=np.uint8)
        for i in range(len(batch_predict)):
            predict_tmp=batch_predict[i]
            gray_pre=self._label2gray(predict_tmp)
            single_name=batch_name[i]
            predict_path=os.path.join(predict_dir,f'{single_name}_mask_predict.tif')
            mask_tmp=self._label2gray(batch_mask[i])
            compare_img[0:predict_tmp.shape[0],0:predict_tmp.shape[1]]=mask_tmp
            compare_img[0:predict_tmp.shape[0],predict_tmp.shape[1]:predict_tmp.shape[1]+10]=255
            compare_img[0:predict_tmp.shape[0],predict_tmp.shape[1]+10:]=gray_pre
            cv2.imwrite(predict_path,compare_img)

    def _write_batch_predict(self,batch_predict,predict_dir,batch_name):
        if not os.path.exists(predict_dir):
            os.mkdir(predict_dir)
        compare_img=np.empty((batch_predict.shape[1],batch_predict.shape[2]*2+10),dtype=np.uint8)
        for i in range(len(batch_predict)):
            predict_tmp=batch_predict[i]
            gray_pre=self._label2gray(predict_tmp)
            single_name=batch_name[i]
            predict_path=os.path.join(predict_dir,f'{single_name}_mask.tif')
            cv2.imwrite(predict_path,gray_pre)

    def _build_lr_scheduler(self,optimizer):
        cfg=self.cfg
        if cfg.LearningRate_Scheduler.Method=='StepLR':#Adjust learning rate at equal intervals
            step=cfg.LearningRate_Scheduler.Step if cfg.LearningRate_Scheduler.Step!=None and cfg.LearningRate_Scheduler.Step!=0 else 50
            gamma=cfg.LearningRate_Scheduler.Gamma if cfg.LearningRate_Scheduler.Gamma!=None and cfg.LearningRate_Scheduler.Gamma!=0 else 0.5
            return  optim.lr_scheduler.StepLR(optimizer,step_size=step,gamma = gamma)
        elif cfg.LearningRate_Scheduler.Method=='ExponentialLR': #Exponential decay adjusted learning rate lr=lr×γ^epoch
            gamma=cfg.LearningRate_Scheduler.Gamma if cfg.LearningRate_Scheduler.Gamma!=None and cfg.LearningRate_Scheduler.Gamma!=0 else 0.9
            return  optim.lr_scheduler.ExponentialLR(optimizer,gamma)
        elif cfg.LearningRate_Scheduler.Method=='CosineAnnealingLR':
            T_max=cfg.LearningRate_Scheduler.T_Max if cfg.LearningRate_Scheduler.TMax!=None and cfg.LearningRate_Scheduler.T_Max!=0 else cfg.Max_Epoch
            eta_min=cfg.LearningRate_Scheduler.Eta_Min if cfg.LearningRate_Scheduler.EtaMin!=None else 0
            return torch.optim.lr_sheduler.CosineAnnealingLR(optimizer, T_max, eta_min, last_epoch=-1)
        
        else:
            return optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma =0.5)






    def train(self,pre_trained_path='',valid=False):
        cfg=self.cfg
        self.model=self.model.to(self.device)
        optimizer=optim.Adam(self.model.parameters(),lr=cfg.InitialLearningRate)
        self.model,optimizer,pre_epoch=self._load_pretrained_model(self.model,optimizer,pre_trained_path)
        self.model.train()
        loss_tool=SegmentationLosses(cuda=(self.device.type=='cuda')).build_loss(cfg.Loss)
        
        save_directory = os.path.join(sys.path[0],'saved_model')
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        now=time.gmtime()
        time_str=f"{now.tm_mon}-{now.tm_mday} {now.tm_hour+8}:{now.tm_min}"
        comment=f"{self.model.name}-{time_str};DataSet:{self.train_dataset.name};Nclass:{cfg.NumClass};lr:{cfg.InitialLearningRate};loss:{cfg.Loss}"
        pprint.pprint(f'{time_str}\n{comment}\n{cfg}')
        experiment_dir = os.path.join(save_directory, comment)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        tb_dir=os.path.join(sys.path[0],'runs')
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)
        tb_dir=os.path.join(tb_dir,f'Train_{comment}')
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)
        tb_writer=SummaryWriter(log_dir=tb_dir,comment=f'-{comment}_{time.ctime()}')
        scheduler =self._build_lr_scheduler(optimizer)
        start=time.perf_counter()
        beat_mIoU=-1
        with torch.set_grad_enabled(True):
            for epoch in range(pre_epoch,cfg.TRAIN.MaxEpoch+pre_epoch):
                self.model.train()
                train_loss=0.0
                print(f"===================epoch:{epoch}=====================")
                for batch in self.train_loader:
                    image=batch[0].to(self.device)
                    mask=batch[1].to(self.device)
                    optimizer.zero_grad()
                    predict=self.model(image)
                    loss=loss_tool(predict,mask)
                    loss.backward()
                    optimizer.step()
                    train_loss+=loss.item()
                scheduler.step()
                mid=time.perf_counter()
                cost=mid-start
                tb_writer.add_scalar('train_loss',train_loss.item())
                print('Loss of Train Data:%.4f'%(train_loss))
                print('Time Cost:%.4f'%(cost))
                if valid:
                    Acc, Acc_class, mIoU,val_loss=self.valid(epoch,tb_writer=tb_writer)
                    if mIoU>beat_mIoU:
                        print(f'This is the model with best mIoU, the epoch is {epoch}, Saving Model......')
                        checkpoint_path = os.path.join(experiment_dir,f'best model')
                        state={'model':self.model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'Config':f'{self.cfg}'}
                        torch.save(state, checkpoint_path)
                        print('Successfully Save!')
                if epoch%10==0 or epoch==cfg.TRAIN.MaxEpoch+pre_epoch-1:
                    print(f'This is the model with every certain step length, the epoch is {epoch}, Saving Model......')
                    checkpoint_path = os.path.join(experiment_dir,f'epoch:{epoch}')
                    state={'model':self.model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'config':f'{self.cfg}'}
                    torch.save(state, checkpoint_path)
                    print('Successfully Save!')
            print("*********************")
            print("Total time cost:%.4f"%(cost))

    def valid(self,pre_epoch=0,pre_trained_path='',pre_result_dir='',tb_writer=None):
        cfg=self.cfg
        self.model.eval()
        self.model=self.model.to(self.device)
        loss_tool=SegmentationLosses(cuda=(self.device.type=='cuda')).build_loss(cfg.Loss)
        comment=f"{self.model.name};DataSet:{self.valid_dataset.name};Nclass:{cfg.NumClass};lr:{cfg.InitialLearningRate};loss:{cfg.Loss}"
        evaluator=metrics.Evaluator(num_class=cfg.NumClass)
        if tb_writer==None:
            tb_dir=os.path.join(sys.path[0],'runs')
            if not os.path.exists(tb_dir):
                os.makedirs(tb_dir)
            tb_dir=os.path.join(tb_dir,f'valid_{comment}')
            if not os.path.exists(tb_dir):
                os.makedirs(tb_dir)
            tb_writer=SummaryWriter(log_dir=tb_dir,comment=f'-{comment}_{time.ctime()}')
        
        if pre_trained_path!='':
            optimizer=optim.Adam(self.model.parameters(),lr=cfg.InitialLearningRate)
            self.model,optimizer,pre_epoch=self._load_pretrained_model(self.model,optimizer,pre_trained_path)
        val_loss=0.0
        
        #If pre_ result_dir is not empty, a file storage prediction chart needs to be established
        #If it is not given, the representative does not need to output the prediction chart, so there is no need to create a folder
        if pre_result_dir!='':
            
            if not os.path.exists(pre_result_dir):
                os.mkdir(pre_result_dir)
            pre_result_dir=os.path.join(pre_result_dir,comment)
            if not os.path.exists(pre_result_dir):
                os.mkdir(pre_result_dir)
            now=time.gmtime()
            time_str=f"{now.tm_mon}-{now.tm_mday} {now.tm_hour+8}:{now.tm_min}"
            pre_result_dir=os.path.join(pre_result_dir,time_str)
            if not os.path.exists(pre_result_dir):
                os.mkdir(pre_result_dir)
        with torch.no_grad():
            for batch in self.valid_loader:
                image=batch[0].to(self.device)
                mask=batch[1].to(self.device)
                batch_name=batch[2]
                predict=self.model(image)
                loss=loss_tool(predict,mask)
                val_loss+=loss.item()
                #Since mask is a single channel and predict is a three channel one
                #softmax the predict before using evaluator
                #Function get_batch_pred_Label returns numpy
                predict=self._get_batch_pre_label(predict)
                #For ease of calculation, mask is converted into numpy array here
                mask=mask.cpu().numpy()
                #_write_batch_predict(self,batch_predict,batch_mask,predict_dir,batch_name):
                if pre_result_dir!='':
                    self._write_batch_predict_mask(predict,mask,pre_result_dir,batch_name)
                evaluator.add_batch(mask,predict)
            Acc=evaluator.Pixel_Accuracy()
            Acc_class=evaluator.Pixel_Accuracy_Class()
            mIoU=evaluator.Mean_Intersection_over_Union()
            FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
            ConfusionMatrix=evaluator.confusion_matrix
            tb_writer.add_scalar('val/loss', val_loss, pre_epoch)
            tb_writer.add_scalar('val/mIoU', mIoU, pre_epoch)
            tb_writer.add_scalar('val/Acc', Acc, pre_epoch)
            tb_writer.add_scalar('val/Acc_class', Acc_class, pre_epoch)
            tb_writer.add_scalar('val/fwIoU', FWIoU, pre_epoch)
            print(f'Test the performance of the {pre_epoch} model on the validation set......')
            print("Precision index of Valid Data:\nAcc:{}\tAcc_class:{}\nmIoU:{}\tVal Loss:{}".format(Acc, Acc_class, mIoU,val_loss))
            return Acc, Acc_class, mIoU,val_loss

    def test(self,pre_epoch=0,pre_trained_path='',pre_result_dir=''):
        cfg=self.cfg
        self.model.eval()
        self.model=self.model.to(self.device)
        comment=f"{self.model.name};DataSet:{self.train_dataset.name};Nclass:{cfg.NumClass};lr:{cfg.InitialLearningRate};loss:{cfg.Loss}"
        
        
        if pre_trained_path!='':
            optimizer=optim.Adam(self.model.parameters(),lr=cfg.InitialLearningRate)
            self.model,optimizer,pre_epoch=self._load_pretrained_model(self.model,optimizer,pre_trained_path)
        val_loss=0.0
        
        #pre_ result_ If dir is not empty, a file storage prediction chart needs to be established
        #If it is not given, the representative does not need to output the prediction chart, 
        #so there is no need to create a folder
        if pre_result_dir!='':
            if not os.path.exists(pre_result_dir):
                os.mkdir(pre_result_dir)
            pre_result_dir=os.path.join(pre_result_dir,comment)
            if not os.path.exists(pre_result_dir):
                os.mkdir(pre_result_dir)
            now=time.gmtime()
            time_str=f"{now.tm_mon}-{now.tm_mday} {now.tm_hour+8}:{now.tm_min}"
            pre_result_dir=os.path.join(pre_result_dir,time_str)
            if not os.path.exists(pre_result_dir):
                os.mkdir(pre_result_dir)
        with torch.no_grad():
            for batch in self.valid_loader:
                image=batch[0].to(self.device)
                batch_name=batch[2]
                predict=self.model(image)
                #Since mask is a single channel and predict is a three channel one, softmax the predict before using evaluator
                #function gey_batch_pred_Label returns numpy
                predict=self._get_batch_pre_label(predict)
                #For ease of calculation, mask is converted into numpy array here
                self._write_batch_predict(predict,pre_result_dir,batch_name)
                
            
            
