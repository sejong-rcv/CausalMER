import argparse
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from src.utils import *
from torch.utils.data import DataLoader
from src import train
import numpy as np
import random

from torch import nn
import sys
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import os
import pickle
from src.eval_metrics import *
from tqdm import tqdm
from src import causal_mer_model

### Default arguments are for unaligned version of MOSEI
### For other datasets, please refer to Supplmentary

parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MulT',
                    help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosei_senti',
                    help='dataset to use (default: mosei_senti)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=5,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='mult',
                    help='name of the trial (default: "mult")')

# for custom
parser.add_argument('--mod', type=str, default='tav',
                    help='define using modality')

# for mosei_senti
parser.add_argument('--mosei_task', default='regression', 
                    choices=['regression', 'cls'])

# for branch lr
parser.add_argument('--m_lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')

parser.add_argument('--t_lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')

parser.add_argument('--a_lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')

parser.add_argument('--v_lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')

parser.add_argument('--c_lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--c_t_lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--c_a_lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--c_v_lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')

# for loss weight
parser.add_argument('--alpha', type=float, default=0.1, help='weight for cls loss')
parser.add_argument('--beta', type=float, default=1.0, help='weight for kl loss')

# for constant init
parser.add_argument('--constant', type=float, default=0.0,
                    help='initial constant (default: 0.0)')

# for fusion mode
parser.add_argument('--fusion_mode', type=str, default='mask',
                    help='define fusion mode')

# for NDE
parser.add_argument('--nde_t', type=float, default=1.0,
                    help='weight for nde_t')
parser.add_argument('--nde_a', type=float, default=1.0,
                    help='weight for nde_a')
parser.add_argument('--nde_v', type=float, default=1.0,
                    help='weight for nde_v')

# for checking imbalance
parser.add_argument('--remove', type=str, default='',
                    help='checking imbalance')

args = parser.parse_args()

random_seed = args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
torch.multiprocessing.set_start_method('spawn')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ['PYTHONHASHSEED'] = str(random_seed)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 필수
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"

dataset = str.lower(args.dataset.strip())
valid_partial_mode = args.lonly + args.vonly + args.aonly

if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

use_cuda = False

output_dim_dict = {
    'mosi': 7 if args.mosei_task == 'cls' else 1,
    'mosei_senti': 7 if args.mosei_task == 'cls' else 1,
    'iemocap': 8
}

criterion_dict = {
    'iemocap': 'CrossEntropyLoss',
    'mosei_senti': 'CrossEntropyLoss' if args.mosei_task == 'cls' else 'L1Loss',
    'mosi': 'CrossEntropyLoss' if args.mosei_task == 'cls' else 'L1Loss'
}

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        # torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True

####################################################################
#
# Load the dataset (aligned or non-aligned)
#
####################################################################

print("Start loading the data....")

train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator(device=device))
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator(device=device))
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, generator=torch.Generator(device=device))

print('Finish loading the data....')
if not args.aligned:
    print("### Note: You are running in unaligned mode.")

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')
# 추가
hyp_params.modality = args.mod
hyp_params.device = device

def caculate_kl_loss(z_te, z_nde, z_nde_t, z_nde_a, z_nde_v):
    p_te = torch.nn.functional.softmax(z_te, -1)
    p_nde = torch.nn.functional.softmax(z_nde, -1)
    kl_loss_main = - p_te * p_nde.log()
    kl_loss_main = kl_loss_main.sum(1).mean()

    p_nde_t = torch.nn.functional.softmax(z_nde_t, -1)
    p_nde_a = torch.nn.functional.softmax(z_nde_a, -1)
    p_nde_v = torch.nn.functional.softmax(z_nde_v, -1)

    kl_loss_ta = - p_nde_t * p_nde_a.log()
    kl_loss_ta = kl_loss_ta.sum(1).mean()

    kl_loss_av = - p_nde_a * p_nde_v.log()
    kl_loss_av = kl_loss_av.sum(1).mean()

    kl_loss_vt = - p_nde_v * p_nde_t.log()
    kl_loss_vt = kl_loss_vt.sum(1).mean()

    kl_loss_t = - p_te * p_nde_t.log()
    kl_loss_t = kl_loss_t.sum(1).mean()

    kl_loss_a = - p_te * p_nde_a.log()
    kl_loss_a = kl_loss_a.sum(1).mean()

    kl_loss_v = - p_te * p_nde_v.log()
    kl_loss_v = kl_loss_v.sum(1).mean()
    
    kl_loss = kl_loss_main + (kl_loss_ta + kl_loss_av + kl_loss_vt)/3 + (kl_loss_t + kl_loss_a + kl_loss_v)

    return kl_loss

def calculate_loss(preds, eval_attr):
    alpha = hyp_params.alpha
    beta = hyp_params.beta
    
    te_loss = criterion(preds['logits_te'], eval_attr)
    t_loss = criterion(preds['logits_t'], eval_attr)
    a_loss = criterion(preds['logits_a'], eval_attr)
    v_loss = criterion(preds['logits_v'], eval_attr)
    
    cls_loss =  te_loss + t_loss + a_loss + v_loss
    kl_loss = caculate_kl_loss(preds['z_te'], preds['z_nde'], preds['z_nde_t'], preds['z_nde_a'], preds['z_nde_v'])
    
    cls_loss = alpha * cls_loss
    kl_loss = beta * kl_loss
    loss = cls_loss + kl_loss
    return loss

def evaluate(model, criterion, test=False):
    model.eval()
    loader = test_loader if test else valid_loader
    total_loss = 0.0

    results = []
    truths = []

    with torch.no_grad():
        for batch_X, batch_Y, batch_Y_cls, batch_META in tqdm(loader):
            sample_ind, text, audio, vision = batch_X
            if hyp_params.mosei_task =='cls':
                eval_attr = batch_Y_cls
            else:
                eval_attr = batch_Y.squeeze(dim=-1) # if num of labels is 1
        
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    if 't' in hyp_params.remove:
                        text = torch.zeros_like(text)
                    if 'a' in hyp_params.remove:
                        audio = torch.zeros_like(audio)
                    if 'v' in hyp_params.remove:
                        vision = torch.zeros_like(vision)
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                    if hyp_params.dataset == 'iemocap' or hyp_params.mosei_task == 'cls':
                        eval_attr = eval_attr.long()
                    
            batch_size = text.size(0)
            
            net = model
            preds = net(text, audio, vision)
            if hyp_params.dataset == 'iemocap':
                preds['logits_te'] = preds['logits_te'].view(-1, 2)
                if hyp_params.modality == 'tav':
                    preds['logits_t'] = preds['logits_t'].view(-1, 2)
                    preds['logits_a'] = preds['logits_a'].view(-1, 2)
                    preds['logits_v'] = preds['logits_v'].view(-1, 2)
                else:
                    print("error")
                eval_attr = eval_attr.view(-1)
            if hyp_params.mosei_task == 'cls':
                eval_attr = eval_attr.view(-1)
                
            combined_loss = calculate_loss(preds, eval_attr)
            total_loss += combined_loss.item() * batch_size

            # Collect the results into dictionary
            results.append(preds['logits_causal_mer'])
            if hyp_params.dataset == 'iemocap':
                truths.append(eval_attr.detach().cpu())
            else:
                truths.append(batch_Y.detach().cpu())
                
    avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
    results = torch.cat(results)
    truths = torch.cat(truths)
    return avg_loss, results, truths

if __name__ == '__main__':
    model = getattr(causal_mer_model, 'CAUSALMER')(hyp_params)
    model.load_state_dict(load_model(hyp_params, name=hyp_params.name).state_dict())
    model = model.cuda()
    criterion = getattr(nn, hyp_params.criterion)()
    _, results, truths = evaluate(model, criterion, test=True)
    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'iemocap':
        eval_iemocap(results, truths)

