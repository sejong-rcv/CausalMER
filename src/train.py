import torch
from torch import nn
import sys
from src import models
from src import causal_mer_model
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, CosineAnnealingLR
import os
import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *

from tqdm import tqdm

####################################################################
#
# Construct the model and the CTC module (which may not be needed)
#
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader):   
    model = getattr(causal_mer_model, 'CAUSALMER')(hyp_params)
    print(hyp_params)
    if hyp_params.use_cuda:
        model = model.cuda()
    optimizer = getattr(optim, hyp_params.optim)([
        {'params': model.tav_branch.parameters(), 'lr': hyp_params.m_lr},
        {'params': model.t_branch.parameters(), 'lr': hyp_params.t_lr},
        {'params': model.a_branch.parameters(), 'lr': hyp_params.a_lr},
        {'params': model.v_branch.parameters(), 'lr': hyp_params.v_lr},
        {'params': model.constant_t, 'lr':hyp_params.c_t_lr},
        {'params': model.constant_a, 'lr':hyp_params.c_a_lr},
        {'params': model.constant_v, 'lr':hyp_params.c_v_lr},
    ], lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)

####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']    
    scheduler = settings['scheduler']    
    
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
        
        # cls_loss =  (alpha * te_loss) + t_loss + a_loss + v_loss
        cls_loss =  te_loss + t_loss + a_loss + v_loss
        kl_loss = caculate_kl_loss(preds['z_te'], preds['z_nde'], preds['z_nde_t'], preds['z_nde_a'], preds['z_nde_v'])
        
        cls_loss = alpha * cls_loss
        kl_loss = beta * kl_loss
        loss = cls_loss + kl_loss
        return loss

    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        start_time = time.time()
        results = list()
        truths = list()
        for batch_X, batch_Y, batch_Y_cls, batch_META in tqdm(train_loader):
            sample_ind, text, audio, vision = batch_X
            if hyp_params.mosei_task =='cls':
                eval_attr = batch_Y_cls
            else:
                eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
            
            model.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                    if hyp_params.dataset == 'iemocap' or hyp_params.mosei_task == 'cls':
                        eval_attr = eval_attr.long()
            
            batch_size = text.size(0)
            batch_chunk = hyp_params.batch_chunk
            # net = nn.DataParallel(model) if batch_size > 10 else model
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
            combined_loss.backward()
            
            results.append(preds['logits_causal_mer'].detach().cpu())
            if hyp_params.dataset == 'iemocap':
                truths.append(eval_attr.detach().cpu())
            else:
                truths.append(batch_Y.detach().cpu())
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)

            optimizer.step()
            epoch_loss += combined_loss.item() * batch_size

        elapsed_time = time.time() - start_time
        print('Epoch {:2d} | Batch {:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                format(epoch, num_batches, elapsed_time, epoch_loss / hyp_params.n_train))
        
        results = torch.cat(results)
        truths = torch.cat(truths)
        
        return epoch_loss / hyp_params.n_train, results, truths

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
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        if hyp_params.dataset == 'iemocap' or hyp_params.mosei_task == 'cls':
                            eval_attr = eval_attr.long()
                        
                batch_size = text.size(0)
                
                # net = nn.DataParallel(model) if batch_size > 10 else model
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
                    truths.append(eval_attr)
                else:
                    truths.append(batch_Y)
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths
    
    # -------------------------------------------------------------------
    best_valid_loss = 1e8
    best_valid_acc = 1e-8
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()

        print("train ACC-----------------------")
        train_loss, train_results, train_truths = train(model, optimizer, criterion)
        if hyp_params.dataset == "mosei_senti":
            train_avg_acc, train_acc2_nonzero, train_acc2 = eval_mosei_senti(train_results, train_truths)
        elif hyp_params.dataset == "mosi":
            train_avg_acc, train_acc2_nonzero, train_acc2 = eval_mosi(train_results, train_truths)
        else:
            train_avg_acc = eval_iemocap(train_results, train_truths)
        
        print("validation ACC-----------------------")
        val_loss, val_results, val_truths = evaluate(model, criterion, test=False)
        if hyp_params.dataset == "mosei_senti":
            val_avg_acc, val_acc2_nonzero, val_acc2 = eval_mosei_senti(val_results, val_truths)
        elif hyp_params.dataset == "mosi":
            val_avg_acc, val_acc2_nonzero, val_acc2 = eval_mosi(val_results, val_truths)
        else:
            val_avg_acc = eval_iemocap(val_results, val_truths)
        
        print("test ACC-----------------------")
        test_loss, test_results, test_truths = evaluate(model, criterion, test=True)
        if hyp_params.dataset == "mosei_senti":
            test_avg_acc, test_acc2_nonzero, test_acc2 = eval_mosei_senti(test_results, test_truths)
        elif hyp_params.dataset == "mosi":
            test_avg_acc, test_acc2_nonzero, test_acc2 = eval_mosi(test_results, test_truths)
        else:
            test_avg_acc = eval_iemocap(test_results, test_truths)

        end = time.time()
        duration = end-start
        scheduler.step(val_loss)

        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
        print("-"*50)
        
        if val_loss < best_valid_loss:
            print(f"Saved model at checkpoint/{hyp_params.name}.pt! for best_valid_loss")
            save_model(hyp_params, model, name=hyp_params.name + "_best_valid_loss")
            best_valid_loss = val_loss
        
        if val_avg_acc > best_valid_acc:
            print(f"Saved model at checkpoint/{hyp_params.name}.pt! for best_valid_acc")
            save_model(hyp_params, model, name=hyp_params.name + "_best_valid_acc")
            best_valid_acc = val_avg_acc

    print("best valid loss -----------")
    model = load_model(hyp_params, name=hyp_params.name + "_best_valid_loss")
    _, results, truths = evaluate(model, criterion, test=True)

    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi':
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == 'iemocap':
        eval_iemocap(results, truths)

    print("best valid acc -----------")
    model = load_model(hyp_params, name=hyp_params.name + "_best_valid_acc")
    _, results, truths = evaluate(model, criterion, test=True)

    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi':
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == 'iemocap':
        eval_iemocap(results, truths)

    sys.stdout.flush()