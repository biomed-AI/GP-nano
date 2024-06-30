import pickle
import numpy as np
import os, random, datetime
from tqdm import tqdm
import warnings
from time import time
warnings.simplefilter('ignore')
import dgl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import *
from evaluate import *
from dataset import *


gpus = list(range(torch.cuda.device_count()))
print("Available GPUs", gpus)

NN_config = {
    'model_class': DiffDockE3NN,
    'hidden_dim': 768,
    'feature_dim':116,
    'dropout': 0.2,
    'weight_decay': 1e-5,
    'lr': 2e-4,
    'obj_max': 1,   # optimization object: max is better
    'epochs': 30,
    'patience': 5,
    'batch_size': 20, 
    'model_strcutute':'',
    'config_s':''
}

def Seed_everything(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def Write_log(logFile,text,isPrint=True):
    if isPrint:
        print(text)
    logFile.write(text)
    logFile.write('\n')

def train_and_test(run_id=None, args=None, seed=None):
    if not run_id:
        run_id = 'run_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    output_path = './output/' + run_id + '/'
    os.makedirs(output_path, exist_ok = True)


    lr = NN_config['lr']
    epochs = NN_config['epochs']
    batch_size = NN_config['batch_size']
    weight_decay = NN_config['weight_decay']
    feature_dim = NN_config['feature_dim']
    hidden_dim = NN_config['hidden_dim']


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.system(f'cp ./*.py {output_path}')
    os.system(f'cp ./*.sh {output_path}')
    
    
    log = open(output_path + 'train_{}.log'.format(seed),'w', buffering=1)
    log.write("Task: " +  "\n" + str(NN_config) + '\n')

    #Dataset and dataloader
    train_dataset = NanobodyDataset(mode='train')
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,pin_memory=False,num_workers=8,collate_fn=dgl_picp_collate)
        #print('train ends')
    #print('train ends')
    valid_dataset = NanobodyDataset(mode='val')
    valid_dataloader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,pin_memory=False,num_workers=8,collate_fn=dgl_picp_collate)
    
    test_dataset = NanobodyDataset(mode='test')
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,pin_memory=False,num_workers=8,collate_fn=dgl_picp_collate)

    # #model
    model = Classifier(in_dim=feature_dim, hidden_dim=hidden_dim, n_classes=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=lr, weight_decay=weight_decay, eps=1e-5)

    # if len(gpus) > 1:
    #     model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

    loss_tr = nn.BCELoss(reduction='mean')
    #loss_tr =torch.nn.BCEWithLogitsLoss(reduction='mean')
    best_auc = 0
    not_improve_epochs = 0
    patience = 5

    
    for epoch in range(epochs):
        train_loss = 0
        model.train()

        bar = tqdm(train_dataloader)
        for batched_graph,labels in bar:
            optimizer.zero_grad()
            batched_graph =  batched_graph.to(device)
            labels = labels.to(device)
            nuv = batched_graph.ndata['nuv'][:,0].to(device)
            node = batched_graph.ndata['f'].to(device)
            feats = torch.concat((node,nuv),dim=1)
            pred = model(batched_graph,feats)
            # kl_loss *= 0

            del batched_graph,feats

            bce_loss = loss_tr(pred, labels)
            loss = bce_loss
            loss.backward()
            optimizer.step()

            train_loss +=  loss.item()
            bar.set_description('loss: %.4f' % (loss.item()))

            del labels, pred, loss # 节省显存

        valid_loss = 0
        valid_pred = []
        valid_label = []
        model.eval()
        bar = tqdm(valid_dataloader)
        for batched_graph, labels in bar:
            batched_graph =  batched_graph.to(device)
            labels = labels.to(device)
            nuv = batched_graph.ndata['nuv'][:,0].to(device)
            node = batched_graph.ndata['f'].to(device)
            feats = torch.concat((node,nuv),dim=1)
            with torch.no_grad():
                pred = model(batched_graph,feats)
                loss = loss_tr(pred, labels)
                valid_loss += loss.item()
                valid_pred.append(pred.detach().cpu().numpy())  
                valid_label.append(labels.detach().cpu().numpy())


        valid_auc,valid_acc,valid_f1,valid_pre,valid_re,valid_rmse,valid_r2score = cal_metrics(valid_label, valid_pred)

        test_loss = 0
        test_pred = []
        test_label = []
        model.eval()

        for batched_graph,labels in tqdm(test_dataloader):
            batched_graph =  batched_graph.to(device)
            labels = labels.to(device)
            nuv = batched_graph.ndata['nuv'][:,0].to(device)
            node = batched_graph.ndata['f'].to(device)
            feats = torch.concat((node,nuv),dim=1)
            pred = model(batched_graph,feats)
            with torch.no_grad():
                test_pred.append(pred.detach().cpu().numpy())
                test_label.append(labels.detach().cpu().numpy())

        test_auc,test_acc,test_f1,test_pre,test_rec,test_rmse,test_r2_score = cal_metrics(test_label, test_pred)
        
        #Write_log(log,'[epoch %s] lr: %.6f, train_loss: %.5f, valid_loss: %.5f, valid_auc: %.4f'%(epoch,lr,train_loss,valid_loss,valid_auc))
        if test_auc>best_auc: # use aupr to select best epoch
            if len(gpus) > 1:
                torch.save(model.module.state_dict(), output_path  + 'model_{}.ckpt'.format(seed))
            else:
                torch.save(model.state_dict(), output_path +  'model_{}.ckpt'.format(seed))
            best_auc = test_auc
            not_improve_epochs = 0
            Write_log(log,'[epoch %s] lr: %.6f, train_loss: %.5f, valid_loss: %.5f, valid_auc: %.4f,valid_acc:%.4f,valid_f1:%.4f,valid_f1_pre:%.4f,valid_f1_recall:%.4f,rmse:%.4f,r2:%.4f' \
            %(epoch,lr,train_loss,valid_loss,valid_auc,valid_acc,valid_f1,valid_pre,valid_re,valid_rmse,valid_r2score))
            mylist = []
            mylist.append(test_label)
            mylist.append(test_pred)
            with open(str(epoch)+'.pkl', 'wb') as file:
                pickle.dump(mylist, file)
        
        else:
            not_improve_epochs += 1
            Write_log(log,'[epoch %s] lr: %.6f, train_loss: %.5f, valid_loss: %.5f, NIE +1 ---> %s'\
            %(epoch,lr,train_loss,valid_loss,not_improve_epochs))
            if not_improve_epochs >= patience:
                break
        Write_log(log,'test_auc: %.5f,test_acc:%.5f,test_f1:%.5f,test_pre:%.5f,test_rec:%.5f,rmse:%.5f,r2:%.5f'%(test_auc,test_acc,test_f1,test_pre,test_rec,test_rmse,test_r2_score))
    Write_log(log, "Training Ends!")


    if args.test:
        if not args.train:
            log = open(output_path  + 'test_{}.log'.format(seed),'w', buffering=1)
            Write_log(log,"Task: "  + "\n" + str(NN_config) + '\n')


        test_dataset = NanobodyDataset(mode='test')
        test_dataloader = DataLoader(test_dataset,batch_size=40,shuffle=False,pin_memory=False,num_workers=8,collate_fn=dgl_picp_collate)
        print('test')
        state_dict = torch.load(output_path  + 'model_{}.ckpt'.format(seed), device)
        #state_dict = torch.load('/data/user/zhouxl/Workspace/antibody/best/model_2027.ckpt'.format(seed), device)
        model.load_state_dict(state_dict)

        model.eval()
        test_loss = 0
        test_pred = []
        test_label = []
        for batched_graph,labels in tqdm(test_dataloader):
            batched_graph =  batched_graph.to(device)
            labels = labels.to(device)
            nuv = batched_graph.ndata['nuv'][:,0].to(device)
            node = batched_graph.ndata['f'].to(device)
            feats = torch.concat((node,nuv),dim=1)
            with torch.no_grad():
                pred = model(batched_graph,feats)
                loss = loss_tr(pred, labels)
                test_loss += loss.item()
                test_pred.append(pred.detach().cpu().numpy())
                test_label.append(labels.detach().cpu().numpy())

        test_auc,test_acc,test_f1,test_pre,test_rec,test_rmse,test_r2_score = cal_metrics(test_label, test_pred)
        epoch = 101
        Write_log(log,'test_auc: %.5f,test_acc:%.5f,test_f1:%.5f,test_pre:%.5f,test_rec:%.5f,rmse:%.5f,r2:%.5f'%(test_auc,test_acc,test_f1,test_pre,test_rec,test_rmse,test_r2_score))
        # drawroc(test_label, test_pred,output_path,epoch,seed)
        # drawprc(test_label, test_pred,output_path,epoch,seed)