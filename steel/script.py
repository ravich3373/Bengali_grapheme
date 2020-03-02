import fastai
from fastai.vision import *
from pathlib import Path
from skimage.io import imread,imshow,imsave
import warnings
from collections import defaultdict
import pandas as pd

from tqdm import tqdm_notebook  as tqdm
import os
import gc
import torch
from fastai.vision.interpret import SegmentationInterpretation
import pickle

data_dir = Path('../train_rsz')
lbl_dir_fish = Path('../fish/lbl_fish_rsz')

tst_dir = Path('../test_rsz')

df = pd.read_csv('../train.csv')

with open('fish_val_fls.pkl','rb') as fl:
    val_fls_fish = pickle.load(fl)
    
def valid_func(fn):
    return fn.name in val_fls_fish

data1 = (SegmentationItemList.from_folder(data_dir)
        #.filter_by_rand(0.01)
       .split_by_valid_func(valid_func)
       .label_from_func(lambda l:lbl_dir_fish/(l.stem+".png"),classes=['bg','fish'])
        #.add_test(get_image_files(tst_dir))
       .transform(get_transforms(), tfm_y=True,size=(350,525))
       .databunch(bs=2)
       .normalize(stats=imagenet_stats))

from fastai.metrics import dice

from fastai.callbacks import *
from torchvision.models import *
from mxresnet import *
from functools import partial
from ranger import *

def fit_with_annealing(learn:Learner, num_epoch:int, lr:float=defaults.lr, annealing_start:float=0.7)->None:
    n = len(learn.data.train_dl)
    anneal_start = int(n*num_epoch*annealing_start)
    phase0 = TrainingPhase(anneal_start).schedule_hp('lr', lr)
    phase1 = TrainingPhase(n*num_epoch - anneal_start).schedule_hp('lr', lr, anneal=annealing_cos)
    phases = [phase0, phase1]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    learn.fit(num_epoch)
    
opt_func = partial(Ranger,  betas=(0.9,0.99), eps=1e-6)

learn = unet_learner(data1, resnet18,pretrained=True,metrics=dice, wd=0.01, bottle=True,opt_func=opt_func).to_fp16()

fit_with_annealing(learn, 1, 0.0001)