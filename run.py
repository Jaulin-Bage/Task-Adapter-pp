import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from utils import *
from dataset import SetDataManager
from models import TaskAdapter

def train(data_loader_list, model, optimization, start_epoch, stop_epoch, params):    

    [base_loader, val_loader, test_loader] = data_loader_list
    if optimization == 'SGD':
        lr = params.lr
        print('lr=', lr)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[12,24,36,48,60,72], gamma=0.8)
        optimizer.zero_grad()
    elif optimization == 'Adam':
        lr = params.lr
        print('lr=', lr)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        optimizer.zero_grad()
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(start_epoch,stop_epoch):

        model.train()

        avg_loss=0
        
        acc_all = []
        iter_num = len(base_loader)

        for i, (x, label) in enumerate(base_loader):

            x = x.cuda() # [n_way, support+query, T, C, H, W] [5,2,8,3,224,224]
            nway, sq, t, c, h, w = x.shape
            x = x.reshape(nway*sq*t, c, h, w) # images
            # with autocast(device_type='cuda'):
            if True:
                vis_dis, sem_dis = model(x,label=label) 

                # compute loss
                y_query = torch.from_numpy(np.repeat(range( nway ), n_query ))
                y_query = Variable(y_query.cuda())
                loss = loss_fn(vis_dis*sem_dis*64, y_query)#*sem_dis*64

                scores = vis_dis*sem_dis
                y_query = y_query.cpu().numpy()
                topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
                topk_ind = topk_labels.cpu().numpy()
                top1_correct = np.sum(topk_ind[:,0] == y_query)
                correct_this, count_this = float(top1_correct), len(y_query)
                acc_all.append(correct_this/ count_this*100  )

                avg_loss = avg_loss+loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
            
        print('Epoch {:d} | Loss {:f} Acc = {:.2f}% +- {:.2f}% | '.format(epoch, avg_loss/float(i+1), acc_mean, 1.96* acc_std/np.sqrt(iter_num)),end="")  
        # writer.add_scalar(tag='loss/train',scalar_value=avg_loss/float(i+1),global_step=epoch)
        # writer.add_scalar(tag='acc/train',scalar_value=acc_mean,global_step=epoch)
        params.logfile.write(time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))+'\n')
        params.logfile.write('Epoch {:d} | Loss {:f} Acc = {:.2f}% +- {:.2f}% | '.format(epoch, avg_loss/float(i+1), acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        
        if epoch > params.warm_up_epoch:
            model.eval()
            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)
            acc_all = []
        
            iter_num = len(val_loader)
            with torch.no_grad():
                for i, (x, label) in enumerate(val_loader):
                    x = x.cuda()

                    nway, sq, t, c, h, w = x.shape
                    x = x.reshape(nway*sq*t, c, h, w) # 80 images

                    vis_dis, sem_dis = model(x,label=label) # 80 x 2048

                    scores = vis_dis*sem_dis
                    y_query = np.repeat(range( nway ), n_query )
                    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
                    topk_ind = topk_labels.cpu().numpy()
                    top1_correct = np.sum(topk_ind[:,0] == y_query)
                    correct_this, count_this = float(top1_correct), len(y_query)
                    acc_all.append(correct_this/ count_this*100  )

            acc_all  = np.asarray(acc_all)
            acc_mean = np.mean(acc_all)
            acc_std  = np.std(acc_all)
            print('%d Val Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        # writer.add_scalar(tag='acc/val',scalar_value=acc_mean,global_step=epoch)
        
            params.logfile.write('%d Val Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num))+'\n')

            acc = acc_mean
            if acc > max_acc:
                print("best model! save...")
                params.logfile.write("best model! save..."+'\n')
                if max_acc > 0:
                    os.remove(os.path.join(params.checkpoint_dir, str(max_acc)+'.tar'))
                max_acc = acc
                outfile = os.path.join(params.checkpoint_dir, str(acc)+'.tar')
                torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        else:
            print("Wait...")
    return model,max_acc

def test(test_loader, model, params):
    
    model.eval()
    acc_all = []
    iter_num = len(test_loader)
    with torch.no_grad():
        for i, (x, label) in enumerate(test_loader):
            x = x.cuda()
            
            nway, sq, t, c, h, w = x.shape
            x = x.reshape(nway*sq*t, c, h, w) # 80 images

            vis_dis, sem_dis = model(x,label=label) # 80 x 2048

            scores = vis_dis*sem_dis
            y_query = np.repeat(range( nway ), n_query )
            topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:,0] == y_query)
            correct_this, count_this = float(top1_correct), len(y_query)
            acc_all.append(correct_this/ count_this*100  )
            if (i+1) % 1000 ==0:
                print('total tasks:{}\tacc_mean:{}\tacc_std:{}'.format(i+1, np.mean(np.asarray(acc_all)), 1.96*np.std(np.asarray(acc_all))/np.sqrt(i+1)))
                if not params.test_model :
                    params.logfile.write('total tasks:{}\tacc_mean:{}\tacc_std:{}'.format(i+1, np.mean(np.asarray(acc_all)), 1.96*np.std(np.asarray(acc_all))/np.sqrt(i+1))+'\n')

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
    if not params.test_model :
        params.logfile.write('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num))+'\n')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子


if __name__=='__main__':

    setup_seed(916) #随机种子
    # params = parse_args('train')
    params = argparse.Namespace(**read_yaml())
    
    base_path=params.dataset_base_path
    if params.dataset == 'kinetics':
        base_file = base_path+params.dataset+'/annotations/train.txt'
        val_file =base_path+params.dataset+'/annotations/test.txt'
        test_file = base_path+params.dataset+'/annotations/test.txt'
        adapter_depth=2
        text_depth=8
    elif params.dataset == 'somethingcmn':
        base_file = base_path+'smsm_cmn/annotations/train.txt'
        val_file =base_path+'smsm_cmn/annotations/test.txt'
        test_file = base_path+'smsm_cmn/annotations/test.txt'
        adapter_depth=6
        text_depth=2
    elif params.dataset == 'somethingotam':
        base_file = base_path+'smsm_otam/annotations/train.txt'
        val_file = base_path+'smsm_otam/annotations/test.txt'
        test_file = base_path+'smsm_otam/annotations/test.txt'
        adapter_depth=6
        text_depth=2
    elif params.dataset == 'hmdb51':
        base_file = base_path+params.dataset+'/annotations/train.txt'
        val_file = base_path+params.dataset+'/annotations/test.txt'
        test_file = base_path+params.dataset+'/annotations/test.txt'
        adapter_depth=2
        text_depth=8
    elif params.dataset == 'ucf101':
        base_file = base_path+params.dataset+'/annotations/train.txt'
        val_file = base_path+params.dataset+'/annotations/test.txt'
        test_file = base_path+params.dataset+'/annotations/test.txt'
        adapter_depth=2
        text_depth=8
    else:
        raise ValueError('Unknown dataset')
    
    
    image_size = 224
    optimization = 'SGD' 

     
    if params.method in ['taskadapter']:
        n_query = params.n_query
        n_shot = params.n_shot


        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        base_datamgr            = SetDataManager(image_size, n_query = n_query,  num_segments = params.num_segments, n_eposide=params.train_episode, **train_few_shot_params)
        base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
         
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 
        val_datamgr             = SetDataManager(image_size, n_query = n_query,  num_segments = params.num_segments, n_eposide = params.eval_episode, **test_few_shot_params)
        val_loader              = val_datamgr.get_data_loader( val_file, aug = False) 

        test_datamgr             = SetDataManager(image_size, n_query = n_query,  num_segments = params.num_segments, n_eposide = params.test_episode, **test_few_shot_params)
        test_loader              = test_datamgr.get_data_loader( test_file, aug = False) 

        model = TaskAdapter(params.train_n_way, params.n_shot, params.n_query, dataset=params.dataset,adapter_depth=adapter_depth,text_depth=text_depth)

    else:
        raise ValueError('Unknown method')

    model = model.cuda()
    model.distribute_backbone([i for i in range(params.num_gpus)])

    if params.test_model:

        checkpoint = torch.load(params.checkpoint,map_location='cpu',weights_only=True)
        checkpoint = checkpoint['state']
        model.load_state_dict(checkpoint)
        # parameters = sum(p.numel() for p in model.text.parameters() if p.requires_grad)
        test(test_loader, model, params)

    else:
        params.checkpoint_dir = '%s/checkpoints/%s/%dway_%dshot' %(os.path.join(params.work_dir,params.method), params.dataset, params.train_n_way, params.n_shot)
        if params.train_aug:
            params.checkpoint_dir += '_aug'
        params.checkpoint_dir = os.path.join(params.checkpoint_dir, time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time())))
        
        params.logfile = os.path.join(params.checkpoint_dir, 'log.txt')
        # writer = SummaryWriter(comment='_'+params.method+'_'+params.dataset+'_%dway_%dshot'%(params.train_n_way, params.n_shot), flush_secs=30)

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        if os.path.isfile(params.logfile):
            params.logfile = open(params.logfile, 'a', buffering=1)
        else:
            params.logfile = open(params.logfile, 'w', buffering=1)

        params.logfile.write('[Message]'+params.message+'\n')
        params.logfile.write('%s\n'%dict(dataset    = params.dataset, 
                                         method     = params.method, 
                                         stop_epoch = params.stop_epoch, 
                                         n_query    = n_query, 
                                         n_shot     = n_shot,
                                         lr         = params.lr, 
                                         ))

        start_epoch = params.start_epoch
        stop_epoch = params.stop_epoch
        data_loader_list = [base_loader, val_loader, test_loader]

        if params.load_weights:
            checkpoint = torch.load(params.checkpoint,map_location='cpu',weights_only=True)
            checkpoint = checkpoint['state']
            model.load_state_dict(checkpoint,strict=True)
        

        model,acc_record = train(data_loader_list,  model, optimization, start_epoch, stop_epoch, params)

        checkpoint = torch.load(os.path.join(params.checkpoint_dir, str(acc_record)+'.tar'),map_location=lambda storage, loc: storage.cuda(0),weights_only=True)
        checkpoint = checkpoint['state']
        model.load_state_dict(checkpoint)
        test(test_loader, model, params)


