from __future__ import division
import time
import os
import argparse
import csv
import sys
from datetime import date, datetime

import torch
import wandb
import pickle
import pandas as pd
import re

import warnings

warnings.filterwarnings("ignore")
os.environ["WANDB_SILENT"] = "True"


# Convert input variable to true or false
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Add parameter arguments to the parser
parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow (or joint for eval)')
parser.add_argument('-train', type=str2bool, default='True', help='train or eval')
parser.add_argument('-comp_info', type=str)
parser.add_argument('-rgb_model_file', type=str)
parser.add_argument('-flow_model_file', type=str)
parser.add_argument('-gpu', type=str, default='4')
parser.add_argument('-dataset', type=str, default='charades')
parser.add_argument('-rgb_root', type=str, default='../data/dataset/v_iashin_i3d')
parser.add_argument('-flow_root', type=str, default='no_root')
parser.add_argument('-type', type=str, default='original')
parser.add_argument('-lr', type=str, default='0.1')
parser.add_argument('-epoch', type=str, default='50')
parser.add_argument('-model', type=str, default='')
parser.add_argument('-APtype', type=str, default='wap')
parser.add_argument('-randomseed', type=str, default='False')
parser.add_argument('-load_model', type=str, default='False')
parser.add_argument('-num_channel', type=str, default='False')
parser.add_argument('-batch_size', type=str, default='False')
parser.add_argument('-kernelsize', type=str, default='False')
parser.add_argument('-feat', type=str, default='False')
parser.add_argument('-split_setting', type=str, default='CS')
parser.add_argument('-json_file', type=str, help='path to json file')
args = parser.parse_args()

# Import neccessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

# set random seed
if args.randomseed == "False":
    SEED = 0
elif args.randomseed == "True":
    SEED = random.randint(1, 100000)
else:
    SEED = int(args.randomseed)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# print('Random_SEED!!!:', SEED)

from torch.optim import lr_scheduler
from torch.autograd import Variable

import json

import pickle
import math

from tqdm.notebook import trange
from tqdm.notebook import tqdm
from time import sleep

if str(args.APtype) == 'map':
    from apmeter import APMeter

batch_size = int(args.batch_size)

# If it is TSU dataset, load the data to train_spilt and test_split
split_setting = str(args.split_setting)

from smarthome_i3d_per_video import TSU as Dataset
from smarthome_i3d_per_video import TSU_collate_fn as collate_fn

classes = 51

if split_setting == 'CS':
    train_split = args.json_file
    test_split = args.json_file


elif split_setting == 'CV':
    train_split = './data/smarthome_CV_51.json'
    test_split = './data/smarthome_CV_51.json'

# Unknown ?
rgb_root = args.rgb_root
skeleton_root = './data/Skeleton'
flow_root = './data/Flow'


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Load RGB skeleton data
def load_data_rgb_skeleton(train_split, val_split, root_skeleton, root_rgb):
    # Load training data to Pytorch dataloader
    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root_skeleton, root_rgb, batch_size, classes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                 pin_memory=True, collate_fn=collate_fn)  # 8
    # Load validation data to Pytorch dataloader
    else:
        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, 'testing', root_skeleton, root_rgb, batch_size, classes)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0,
                                                 pin_memory=True, collate_fn=collate_fn)  # 2

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    return dataloaders, datasets


def load_data(train_split, val_split, root):
    # Load Data
    # Change from 8 to 2
    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root, batch_size, classes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                                 pin_memory=True, collate_fn=collate_fn)
        dataloader.root = root
    else:

        dataset = None
        dataloader = None
    # Change from 2 to 1
    val_dataset = Dataset(val_split, 'testing', root, batch_size, classes)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1,
                                                 pin_memory=True, collate_fn=collate_fn)
    val_dataloader.root = root

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    return dataloaders, datasets


# train the model
def run(models, criterion, num_epochs=50):
    wandb.init(project="training-visualisation",
               config={
                   "batch_size": int(args.batch_size),
                   "learning_rate": float(args.lr),
                   "epochs": int(args.epoch),
               })

    since = time.time()
    best_model = None
    best_map = 0.0
    pbar = trange(num_epochs, desc='All epoch')
    for idx, epoch in enumerate(pbar):
        # for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        probs = []
        for model, gpu, dataloader, optimizer, sched, model_file in models:
            with tqdm(dataloader['train']) as tepoch:
                tepoch.set_description('Epoch {}/{} train'.format(epoch+1, num_epochs))
                train_map, train_loss = train_step(model, gpu, optimizer, tepoch, epoch)

            with tqdm(dataloader['val']) as tepoch:
                tepoch.set_description('Epoch {}/{} val'.format(epoch+1, num_epochs))
                prob_val, val_loss, val_map = val_step(model, gpu, tepoch, epoch)
                probs.append(prob_val)
                sched.step(val_loss)

                wandb.log({"loss": val_loss.item(), "accuracy": val_map.item()})

                if best_map < val_map:
                    best_map = val_map
                    pbar.set_postfix({'loss': val_loss.item(), 'best accuracy': best_map.item()})
                    wandb.log({"best accuracy": best_map.item()})
                    best_model = model
                    pickle.dump(prob_val, open('./models/' + str(epoch) + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                    # torch.save(model.state_dict(),
                    #            './results/' + str(args.model) + '/weight_epoch_' + str(args.lr) + '_' + str(epoch))
                    # torch.save(model, './results/' + str(args.model) + '/model_epoch_' + str(args.lr) + '_' + str(epoch))
                    # print('save here for model: ',
                    #      './results/' + str(args.model) + '/model_epoch_' + str(args.lr) + '_' + str(epoch))
                    # print('save here for weight:',
                    #      './results/' + str(args.model) + '/weight_epoch_' + str(args.lr) + '_' + str(epoch))

    # Create results folder if it doesn't exist
    if not os.path.exists("results"):

        # if the results directory is not present
        # then create it.
        os.makedirs("results")

    if not os.path.exists("results/" + 'training'):

        # if the results_folder_name directory is not present
        # then create it.
        os.makedirs("results/" + 'training')

    # Save the best model
    timestr = time.strftime("%Y%m%d-%H%M%S")
    torch.save(best_model, f'./models/PDAN_TSU_RGB_Train_{timestr}')
    print(f"Trained model saved in ./models/PDAN_TSU_RGB_Train_{timestr}")

    # Creating 'Overall Accuracy (Training)' CSV file
    # Column names: Trained On | Train m-AP | Train Loss | Train Epochs
    cleaned_train_map = (str(train_map))[7:-1]  # Remove tensor() and trailing comma
    cleaned_val_map = (str(best_map))[7:-1]  # Remove tensor() and trailing comma
    cleaned_train_loss = (str(train_loss))[7:-18]  # Remove tensor() and trailing comma
    cleaned_val_loss = (str(val_loss ))[7:-18]  # Remove tensor() and trailing comma
    # Load the JSON file to dict
    json_file = open(args.json_file, 'r')
    video_dict = json.load(json_file)
    noOfTrainingVideo = 0
    noOfTestingVideo = 0
    for values in video_dict.values():
        # Get training videos
        if values['subset'] == 'training':
            noOfTrainingVideo += 1
        # Get testing videos
        elif values['subset'] == 'testing':
            noOfTestingVideo += 1

    df = pd.DataFrame({
        'Train On': f'{noOfTrainingVideo} Training Videos',
        'Train m-AP': cleaned_train_map,
        'Train Loss': cleaned_train_loss,
        'Tested On': f'{noOfTestingVideo} Testing Videos',
        'Prediction m-AP': cleaned_val_map,
        'Validation loss': cleaned_val_loss,
        'Epochs': str(int(args.epoch)),
    }, index=[0])

    cwd = os.getcwd()
    csv_file_name = f'{timestr}_Overall_Training_Accuracy.csv'
    file_path = os.path.join(cwd, 'results', 'training', csv_file_name)
    df.to_csv(file_path, index=False)

    # Add in title Overall Training Accuracy
    title = ['Overall Training Accuracy']

    with open(file_path, 'r') as readFile:
        rd = csv.reader(readFile)
        lines = list(rd)
        lines.insert(0, title)

    with open(file_path, 'w', newline='') as writeFile:
        wt = csv.writer(writeFile)
        wt.writerows(lines)

    readFile.close()
    writeFile.close()

    print(f"Training results saved in {file_path}")


# Eval the model
def eval_model(model, dataloader, baseline=False):
    results = {}
    for data in dataloader:
        other = data[3]
        outputs, loss, probs, _ = run_network(model, data, 0, baseline)
        fps = outputs.size()[1] / other[1][0]

        results[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], data[2].numpy()[0], fps)
    return results


# Run the model through the network
def run_network(model, data, gpu, epoch=0, baseline=False):
    inputs, mask, labels, other = data
    # wrap them in Variable
    inputs = Variable(inputs.cuda(gpu))
    mask = Variable(mask.cuda(gpu))
    labels = Variable(labels.cuda(gpu))

    mask_list = torch.sum(mask, 1)
    mask_new = np.zeros((mask.size()[0], classes, mask.size()[1]))
    for i in range(mask.size()[0]):
        mask_new[i, :, :int(mask_list[i])] = np.ones((classes, int(mask_list[i])))
    mask_new = torch.from_numpy(mask_new).float()
    mask_new = Variable(mask_new.cuda(gpu))

    inputs = inputs.squeeze(3).squeeze(3)
    activation = model(inputs, mask_new)

    outputs_final = activation

    if args.model == "PDAN_TSU_RGB":
        # print('outputs_final1', outputs_final.size())
        # outputs_final = outputs_final[:, 0, :, :]  # Original
        outputs_final = outputs_final[0, :, :, :] # Modified
    # print('outputs_final',outputs_final.size())
    outputs_final = outputs_final.permute(0, 2, 1)
    probs_f = F.sigmoid(outputs_final) * mask.unsqueeze(2)
    loss_f = F.binary_cross_entropy_with_logits(outputs_final, labels, size_average=False)
    loss_f = torch.sum(loss_f) / torch.sum(mask)

    loss = loss_f

    corr = torch.sum(mask)
    tot = torch.sum(mask)

    return outputs_final, loss, probs_f, corr / tot


# Train the model
def train_step(model, gpu, optimizer, dataloader, epoch):
    model.train(True)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    apm = APMeter()
    for data in dataloader:
        optimizer.zero_grad()
        num_iter += 1

        outputs, loss, probs, err = run_network(model, data, gpu, epoch)
        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        error += err.data
        tot_loss += loss.data
        loss.backward()
        optimizer.step()

        dataloader.set_postfix({'loss': (tot_loss / num_iter).item(), 'accuracy': (100 * apm.value().mean()).item()})
        sleep(0.1)

    if args.APtype == 'wap':
        train_map = 100 * apm.value()
    else:
        train_map = 100 * apm.value().mean()
    # print('train-map:', train_map)
    apm.reset()

    epoch_loss = tot_loss / num_iter
    # print('epoch-loss:', epoch_loss)
    return train_map, epoch_loss


# validate the model
def val_step(model, gpu, dataloader, epoch):
    model.train(False)
    apm = APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    num_preds = 0

    full_probs = {}

    # Iterate over data.
    for data in dataloader:
        num_iter += 1
        other = data[3]

        outputs, loss, probs, err = run_network(model, data, gpu, epoch)

        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])

        error += err.data
        tot_loss += loss.data

        probs = probs.squeeze()

        full_probs[other[0][0]] = probs.data.cpu().numpy().T

        dataloader.set_postfix({'loss': (tot_loss / num_iter).item(), 'accuracy': (
                    torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]).item()})
        sleep(0.1)

    epoch_loss = tot_loss / num_iter

    val_map = torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
    # print('Training accuracy:', val_map)
    # print(100 * apm.value())
    apm.reset()
    return full_probs, epoch_loss, val_map


if __name__ == '__main__':
    print(str(args.model))
    print('batch_size:', batch_size)
    print('cuda_avail', torch.cuda.is_available())
    print('CS Json File: ', args.json_file)

    if args.mode == 'flow':
        print('flow mode', flow_root)
        dataloaders, datasets = load_data(train_split, test_split, flow_root)
    elif args.mode == 'skeleton':
        print('Pose mode', skeleton_root)
        dataloaders, datasets = load_data(train_split, test_split, skeleton_root)
    elif args.mode == 'rgb':
        print('RGB mode', rgb_root)
        dataloaders, datasets = load_data(train_split, test_split, rgb_root)

    if args.train:
        num_channel = args.num_channel
        if args.mode == 'skeleton':
            input_channnel = 256
        else:
            input_channnel = 1024

        num_classes = classes
        mid_channel = int(args.num_channel)

        if args.model == "PDAN_TSU_RGB":
            print("you are processing PDAN_TSU_RGB")
            from models import PDAN as Net

            model = Net(num_stages=1, num_layers=5, num_f_maps=mid_channel, dim=input_channnel, num_classes=classes)

        model = torch.nn.DataParallel(model)

        if args.load_model != "False":
            # entire model
            model = torch.load(args.load_model, map_location=torch.device('cpu'))
            # weight
            # model.load_state_dict(torch.load(str(args.load_model)))
            print("loaded", args.load_model)

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('pytorch_total_params', pytorch_total_params)
        print('num_channel:', num_channel, 'input_channnel:', input_channnel, 'num_classes:', num_classes)
        model.cuda()

        criterion = nn.NLLLoss(reduce=False)
        lr = float(args.lr)
        # print(lr)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, verbose=True)
        run([(model, 0, dataloaders, optimizer, lr_sched, args.comp_info)], criterion, num_epochs=int(args.epoch))

        if wandb.run is not None:
            wandb.finish()
