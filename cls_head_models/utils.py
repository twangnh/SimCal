import numpy as np
import matplotlib.pyplot as plt
import torch
# from sklearn.metrics import f1_score
import importlib
# import pdb
import pickle

def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def batch_show(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(20,20))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

def print_write(print_str, log_file):
    print(*print_str)
    with open(log_file, 'a') as f:
        print(*print_str, file=f)

def init_weights(model, weights_path, caffe=False, classifier=False):  
    """Initialize weights"""
    print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',
                                              weights_path))    
    weights = torch.load(weights_path)   
    if not classifier:
        if caffe:
            weights = {k: weights[k] if k in weights else model.state_dict()[k] 
                       for k in model.state_dict()}
        else:
            weights = weights['state_dict_best']['feat_model']
            weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k] 
                       for k in model.state_dict()}
    else:      
        weights = weights['state_dict_best']['classifier']
        weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k] 
                   for k in model.state_dict()}
    model.load_state_dict(weights)   
    return model

def shot_acc (preds, labels, many_shot_thr1=1000, many_shot_thr2=100, low_shot_thr=10):

    train_info = pickle.load(open('./lvis_train_cats_info.pt', 'rb'))
    val_info = pickle.load(open('./lvis_val_cats_info.pt', 'rb'))
    train_on_val_cls = [train_item for idx, (train_item, val_item) in enumerate(zip(train_info, val_info)) if
                        val_item['instance_count'] > 0]
    train_on_val_cls_label_to_info = {item['id']: item['instance_count'] for item in train_on_val_cls}
    train_on_val_cls_label_to_info[0] = 10000## bg, fake num

    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    train_class_count = []
    test_class_count = []
    class_correct = []
    class_preds = []
    for l in np.unique(labels):
        train_class_count.append(train_on_val_cls_label_to_info[l])
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())
        class_preds.append((preds == l).sum())
    many_shot1 = []
    many_shot2 = []
    median_shot = []
    low_shot = []

    many_shot1_precision = []
    many_shot2_precision = []
    median_shot_precision = []
    low_shot_precision = []

    # for i in range(len(train_class_count)):
    #     if train_class_count[i] >= many_shot_thr:
    #         many_shot.append((class_correct[i] / test_class_count[i]))
    #     elif train_class_count[i] <= low_shot_thr:
    #         low_shot.append((class_correct[i] / test_class_count[i]))
    #     else:
    #         median_shot.append((class_correct[i] / test_class_count[i]))
    # return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)

    for i in range(len(train_class_count)):
        if train_class_count[i] >= many_shot_thr1:
            many_shot1.append((class_correct[i] / test_class_count[i]))
            many_shot1_precision.append((class_correct[i] / class_preds[i]))
        elif train_class_count[i] < many_shot_thr1 and train_class_count[i] >= many_shot_thr2:
            many_shot2.append((class_correct[i] / test_class_count[i]))
            many_shot2_precision.append((class_correct[i] / class_preds[i]))
        elif train_class_count[i] < many_shot_thr2 and train_class_count[i] >= low_shot_thr:
            median_shot.append((class_correct[i] / test_class_count[i]))
            median_shot_precision.append((class_correct[i] / class_preds[i]))
        else:
            low_shot.append((class_correct[i] / test_class_count[i]))
            low_shot_precision.append((class_correct[i] / class_preds[i]))
    return many_shot1[0], np.mean(many_shot1),np.mean(many_shot2), np.mean(median_shot), np.mean(low_shot), \
           many_shot1_precision,many_shot2_precision, median_shot_precision, low_shot_precision
def F_measure(preds, labels, openset=False, theta=None):
    
    if openset:
        # f1 score for openset evaluation
        true_pos = 0.
        false_pos = 0.
        false_neg = 0.
        
        for i in range(len(labels)):
            true_pos += 1 if preds[i] == labels[i] and labels[i] != -1 else 0
            false_pos += 1 if preds[i] != labels[i] and labels[i] != -1 and preds[i] != -1 else 0
            false_neg += 1 if preds[i] != labels[i] and labels[i] == -1 else 0

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        return 2 * ((precision * recall) / (precision + recall + 1e-12))
    else:
        # Regular f1 score
        # return f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')
        return f1_score(labels, preds, average='macro')

def mic_acc_cal(preds, labels):
    bg_acc =(preds[labels==0]==0).sum()/(labels==0).sum().float()
    fg_acc = torch.eq(preds, labels).float()[labels != 0].mean()
    acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1, bg_acc, fg_acc

def class_count (data):
    labels = np.array(data.dataset.labels)
    class_data_num = []
    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
    return class_data_num

# def dataset_dist (in_loader):

#     """Example, dataset_dist(data['train'][0])"""
    
#     label_list = np.array([x[1] for x in in_loader.dataset.samples])
#     total_num = len(data_list)

#     distribution = []
#     for l in np.unique(label_list):
#         distribution.append((l, len(label_list[label_list == l])/total_num))
        
#     return distribution

