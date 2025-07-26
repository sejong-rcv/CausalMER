import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score


def multiclass_acc(preds, truths, is_cls=False):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    if is_cls:
        return np.sum((np.round(preds) - 3) == np.round(truths)) / float(len(truths))
    else:
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

# using ce loss code ----------------------------------------------------------------
def eval_mosei_senti(results, truths, exclude_zero=False):
    # results => classification (dim=7)
    # truths => original gt ([-3, 3] real num)
    is_cls = False
    if results.shape[-1] == 7: # cls
        results = torch.argmax(results, dim=1)    
        is_cls = True
        
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()
    
    # 1. acc7
    if is_cls:
        test_preds_a7 = test_preds
    else:
        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    acc7 = multiclass_acc(test_preds_a7, test_truth_a7, is_cls)
    
    # 2. acc2 non-neg/neg
    binary_truth = (test_truth >= 0)
    if is_cls: 
        binary_preds = (test_preds >= 3)
    else: 
        binary_preds = (test_preds >= 0)
    acc2 = accuracy_score(binary_truth, binary_preds)
    f1 = f1_score(binary_truth, binary_preds)
    
    # 3. acc2 pos/neg 
    # for pos/neg acc
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    binary_truth = (test_truth[non_zeros] >= 0)
    if is_cls:
        binary_preds = (test_preds[non_zeros] >= 3)
    else:
        binary_preds = (test_preds[non_zeros] > 0)
    acc2_nonzero = accuracy_score(binary_truth, binary_preds)
    f1_nonzero = f1_score(binary_truth, binary_preds)
    
    ## for regression task
    # mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    # corr = np.corrcoef(test_preds, test_truth)[0][1]
    # f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    # print("MAE: ", mae)
    # print("Correlation Coefficient: ", corr)
    
    print("mult_acc_7: ", acc7)
    print("acc2 (pos/neg): ", acc2_nonzero)
    print("F1 score (pos/neg)", f1_nonzero)
    print("acc2 (neg/non-neg): ", acc2)
    print("F1 score (neg/non-neg)", f1)

    print("-" * 50)
    
    return acc7, acc2_nonzero, acc2
# ------------------------------------------------------------------------------

def eval_mosi(results, truths, exclude_zero=False):
    return eval_mosei_senti(results, truths, exclude_zero)

def eval_iemocap(results, truths, single=-1):
    emos = ["Neutral", "Happy", "Sad", "Angry"]
    if single < 0:
        acc_list = []
        f1_list = []

        test_preds = results.view(-1, 4, 2).cpu().detach().numpy()
        test_truth = truths.view(-1, 4).cpu().detach().numpy()
        
        for emo_ind in range(4):
            print(f"{emos[emo_ind]}: ")
            test_preds_i = np.argmax(test_preds[:,emo_ind],axis=1)
            test_truth_i = test_truth[:,emo_ind]
            f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
            acc = accuracy_score(test_truth_i, test_preds_i)
            print("  - F1 Score: ", f1)
            print("  - Accuracy: ", acc)
            acc_list.append(acc)
            f1_list.append(f1)
        print("Average Accuracy: ", sum(acc_list)/4)
        print("Average F1: ", sum(f1_list)/4)
        return sum(acc_list)/4
    else:
        test_preds = results.view(-1, 2).cpu().detach().numpy()
        test_truth = truths.view(-1).cpu().detach().numpy()
        
        print(f"{emos[single]}: ")
        test_preds_i = np.argmax(test_preds,axis=1)
        test_truth_i = test_truth
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        print("  - F1 Score: ", f1)
        print("  - Accuracy: ", acc)



