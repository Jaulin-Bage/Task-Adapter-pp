# import mmcv
import decord
from PIL import Image, ImageEnhance
import torch
import numpy as np
import torchvision.transforms as transforms
import json
import os

identity = lambda x:x
transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)


class VideoDataset:
    def __init__(self, data_file, image_size, train_aug=False, num_segments=None):
        self.video_list = [x.strip().split(' ') for x in open(data_file)]
        print('video number:%d' % (len(self.video_list)))
        self.num_segments = num_segments
        self.train_aug = train_aug 
        self.trans_loader = TransformLoader(image_size)
        self.transform = self.trans_loader.get_composed_transform(train_aug)
        self.label_transform = transforms.ToTensor()
        self.image_tmpl = 'img_{:05d}.jpg'

    def __getitem__(self, i):
        assert len(self.video_list[i]) == 3
        full_path = self.video_list[i][0]
        num_frames = int(self.video_list[i][1])
        label = int(self.video_list[i][2])

        num_segments = self.num_segments
        if self.train_aug and num_frames>8 : # random sample
            # frame_id = np.random.randint(num_frames)
            average_duration = num_frames // num_segments
            frame_id = np.multiply(list(range(num_segments)), average_duration)
            frame_id = frame_id + np.random.randint(average_duration, size=num_segments)
        else:
            # frame_id = num_frames//2
            tick = num_frames / float(num_segments)
            frame_id = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
        frame_id = frame_id + 1 # idx >= 1

        img_group = []
        for k in range(self.num_segments):
            img_path = os.path.join(full_path,self.image_tmpl.format(frame_id[k]))
            img = Image.open(img_path)
            img = self.transform(img)
            img_group.append(img)
        img_group = torch.stack(img_group,0)
        label = torch.tensor(label)
        # print('ok',image_path)
        return img_group, label

    def __len__(self):
        return len(self.video_list)


class SubVideoDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, random_select=False, num_segments=8):
        self.sub_meta = sub_meta
        # self.video_list = [x.strip().split(' ') for x in open(sub_meta)]
        if True:
            self.image_tmpl = 'img_{:05d}.jpg'
        else:
            self.image_tmpl = 'img_{:05d}.png'
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform
        self.random_select = random_select
        self.num_segments = num_segments
    
    def __getitem__(self,i):
        # image_path = os.path.join( self.sub_meta[i])
        assert len(self.sub_meta[i]) == 2
        full_path = self.sub_meta[i][0]
        num_frames = self.sub_meta[i][1]
        num_segments = self.num_segments
        if self.random_select and num_frames>8 : # random sample
            # frame_id = np.random.randint(num_frames)
            average_duration = num_frames // num_segments
            frame_id = np.multiply(list(range(num_segments)), average_duration)
            frame_id = frame_id + np.random.randint(average_duration, size=num_segments)
        else:
            # frame_id = num_frames//2
            tick = num_frames / float(num_segments)
            frame_id = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
        frame_id = frame_id + 1 # idx >= 1

        img_group = []
        for k in range(self.num_segments):
            img_path = os.path.join(full_path,self.image_tmpl.format(frame_id[k]))
            img = Image.open(img_path)
            img = self.transform(img)
            img_group.append(img)
        img_group = torch.stack(img_group,0)
        target = self.target_transform(self.cl)
        # print('ok',image_path)
        return img_group, target

    def __len__(self):
        return len(self.sub_meta)
        
class SetDataManager:
    def __init__(self, image_size, n_way, n_support, n_query, num_segments, n_eposide =200):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)
        self.num_segments = num_segments

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset( data_file , self.batch_size, transform, random_select=aug, num_segments=self.num_segments) # video
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 48, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

class TransformLoader:
    def __init__(self, image_size, 
                    #normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),#imagenet归一化参数
                    #normalize_param    = dict(mean= [0.376, 0.401, 0.431] , std=[0.224, 0.229, 0.235]),#本来的参数
                 normalize_param    =dict(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),#CLIP归一化参数
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomResizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Resize':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        elif transform_type=='RandomCrop':
            return method([self.image_size, self.image_size])
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            #transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
            # transform_list = ['Resize', 'RandomHorizontalFlip', 'RandomCrop', 'ToTensor', 'Normalize']#
            transform_list = ['Resize', 'RandomCrop', 'ToTensor', 'Normalize']#
        else:
            transform_list = ['Resize','CenterCrop', 'ToTensor','Normalize']#'Normalize'

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class SetDataset: # frames
    def __init__(self, data_file, batch_size, transform, random_select=False, num_segments=None):
        # with open(data_file, 'r') as f:
            # self.meta = json.load(f)
        self.video_list = [x.strip().split(' ') for x in open(data_file)]

        # self.cl_list = np.unique(self.meta['image_labels']).tolist()
        self.cl_list = np.zeros(len(self.video_list),dtype=int)
        for i in range(len(self.video_list)):
            self.cl_list[i] = self.video_list[i][2]
        self.cl_list = np.unique(self.cl_list).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        # for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            # self.sub_meta[y].append(x)
        for x in range(len(self.video_list)):
            root_path = self.video_list[x][0]
            num_frames = int(self.video_list[x][1])
            label = int(self.video_list[x][2])
            self.sub_meta[label].append([root_path,num_frames])

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubVideoDataset(self.sub_meta[cl], cl, transform = transform ,random_select = random_select, num_segments=num_segments)
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]#torch.tensor([74, 74, 48,48,48])

class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]


    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out

# if __name__ == '__main__':
#     base_file = '/mnt/zhangyueran/Datasets/smsm_otam/annotations/train.txt'
#     val_file = '/mnt/zhangyueran/Datasets/smsm_otam/annotations/val.txt'
#     test_file = '/mnt/zhangyueran/Datasets/smsm_otam/annotations/test.txt'

#     train_few_shot_params    = dict(n_way = 5, n_support = 5) 
#     base_datamgr            = SetDataManager(224, n_query = 1,  num_segments =8, **train_few_shot_params)
#     base_loader             = base_datamgr.get_data_loader( base_file , aug = True )

    # train_dataset = VideoDataset(data_file=base_file,img_prefix='/home/zzx/workspace/data/kinetics_100',img_norm_cfg=img_norm_cfg,num_segments=8)
    # print(train_dataset.__len__())
    # data_loader_params = dict(batch_size = 2, shuffle = True, num_workers = 1, pin_memory = True)       
    # train_loader = torch.utils.data.DataLoader(train_dataset, **data_loader_params)

    # for i, (x,y) in enumerate(train_loader):
        # print(x.shape,y)
    # l = []
    # for i, (x,y) in enumerate(base_loader):
    #     l.append(x)
