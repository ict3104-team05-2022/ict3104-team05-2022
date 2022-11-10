import argparse
import json
import numpy as np
import pickle
import torch
import warnings
import torch._utils
import wandb

from apmeter import APMeter

warnings.filterwarnings("ignore")
def make_gt(gt_file, logits, num_classes=51):
    gt_new = {}
    with open(gt_file, 'r') as f:
        gt = json.load(f)

    i = 0
    for vid in gt.keys():
        if gt[vid]['subset'] != "testing":
            continue
        # print(logits[vid].shape)
        num_pred = logits[vid].shape[1]
        label = np.zeros((num_pred, num_classes), np.float32)

        fps = float(num_pred / float(gt[vid]['duration']))
        for ann in gt[vid]['actions']:
            for fr in range(0, num_pred, 1):
                if fr / fps > ann[1] and fr / fps < ann[2]:
                    label[fr, ann[0]] = 1
        gt_new[vid]=label
        i += 1
    return gt_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-split', type=str)
    parser.add_argument('-pkl_path', type=str)  # './test.pkl'
    
    parser.add_argument('-batch_size', type=str) 
    parser.add_argument('-learning_rate', type=str)  
    parser.add_argument('-epochs', type=str)
    args = parser.parse_args()
    
    wandb.init(project="testing-visualisation",
    config={
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
    })

    split = args.split
    pkl_path = args.pkl_path

    if split == 'CS':
        gt_file = './data/smarthome_CS_51_v2.json'
        classes = 51

    pkl = open(pkl_path, 'rb')
    logits = pickle.load(pkl, encoding='latin1')

    gt_new=make_gt(gt_file,logits,classes)

    apm = APMeter()
    for vid in gt_new.keys():
        logit = np.transpose(logits[vid], (1, 0))
        apm.add(logit, gt_new[vid])

        val_map = torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
        
        wandb.log({"val_map": val_map})
        wandb.log({"avg_class_accuracy": sum(apm.value()) / len(apm.value())})
        
    val_map = torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
    print ("Test Frame-based map", val_map.mean())
