from __future__ import division
import time
import os
import argparse
import sys


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow (or joint for eval)', default='rgb')  # added default parameter
parser.add_argument('-train', type=str2bool, default='False', help='train or eval')
parser.add_argument('-comp_info', type=str, default='PDAN_TSU_RGB') #change default from "" to "PDAN_TSU_RGB"
parser.add_argument('-rgb_model_file', type=str, default='')
parser.add_argument('-flow_model_file', type=str)
parser.add_argument('-gpu', type=str, default='4')
parser.add_argument('-dataset', type=str, default='TSU')  # change default from "charades" to "TSU"
parser.add_argument('-rgb_root', type=str, default='no_root')
parser.add_argument('-flow_root', type=str, default='no_root')
parser.add_argument('-type', type=str, default='original')
parser.add_argument('-lr', type=str, default='0.1')
parser.add_argument('-epoch', type=str, default='5')  # change default from "50" to "5"
parser.add_argument('-model', type=str, default='PDAN_TSU_RGB')  # change default from "" to "PDAN_TSU_RGB"
parser.add_argument('-APtype', type=str, default='map')  # change default from "wap" to "map"
parser.add_argument('-randomseed', type=str, default='False')
parser.add_argument('-load_model', type=str,
                    default='./models/PDAN_TSU_RGB')  # change default from "False" to "./models/PDAN_TSU_RGB"
parser.add_argument('-num_channel', type=str,
                    default='512')  # change default from "False" to "512" (just random no idea why 3)
parser.add_argument('-batch_size', type=str, default='2')  # change default from "False" to "1"
parser.add_argument('-kernelsize', type=str, default='3') # change default from "False" to "3"
parser.add_argument('-feat', type=str, default='False')
parser.add_argument('-split_setting', type=str, default='CS')
parser.add_argument('-input_video_file', type=str, default='P02T04C05',  help='input video file name')
parser.add_argument('-input_video_full_path', type=str, default='../data/P02T04C05.mp4', help='input video file path')
parser.add_argument('-test', type=str2bool, default='False', help='train or eval')
args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import cv2

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
print('Random_SEED!!!:', SEED)

from torch.optim import lr_scheduler
from torch.autograd import Variable

import json

import pickle
import math

if str(args.APtype) == 'map':
    from apmeter import APMeter

batch_size = int(args.batch_size)

if args.dataset == 'TSU':
    split_setting = str(args.split_setting)

    from smarthome_i3d_per_video import TSU as Dataset
    from smarthome_i3d_per_video import TSU_collate_fn as collate_fn

    classes = 51

    if split_setting == 'CS':
        train_split = './data/smarthome_CS_51.json'
        test_split = './data/smarthome_CS_51.json'

    elif split_setting == 'CV':
        train_split = './data/smarthome_CV_51.json'
        test_split = './data/smarthome_CV_51.json'

    rgb_root = './data/RGB'
    skeleton_root = '/skeleton/feat/Path/'

fileName = ""


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_data_rgb_skeleton(train_split, val_split, root_skeleton, root_rgb):
    # Load Data

    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root_skeleton, root_rgb, batch_size, classes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                 pin_memory=True, collate_fn=collate_fn)  # 8
    else:
        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, 'testing', root_skeleton, root_rgb, batch_size, classes)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0,
                                                 pin_memory=True, collate_fn=collate_fn)  # 2

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    return dataloaders, datasets


activityList = ["Enter", "Walk", "Make_coffee.Get_water", "Make_tea/put something in sink", "unknown class 4",
                "Use_Drawer", "unknown class 6", "Use_telephone",
                "Leave", "Put_something_on_table", "Drink.From_glass", "unknown class 11", "unknown class 12",
                "Drink.From_cup", "Dump_in_trash", "unknown class 15",
                "unknown class 16", "Use_cupboard", "unknown class 18", "Read", "Drink.From_bottle", "Use_fridge",
                "Wipe_table/clean dish with water", "unknown class 23",
                "Eat_snack", "Sit_down", "Watch_TV", "Use_laptop", "Get_up", "Drink.From_bottle", "unknown class 30",
                "unknown class 31",
                "Lay_down", "unknown class 33", "Write", "Breakfast.Eat_at_table", "unknown class 36",
                "unknown class 37", "unknown class 38", "Breakfast.Cut_bread",
                "Clean_dishes.Dry_up", "unknown class 41", "Cook.Use_stove", "Cook.Cut", "unknown class 44",
                "Cook.Stir", "Cook.Use_oven", "like uselaptop",
                "unknown class 48", "unknown class 49", "unknown class 50"]


# self declared (essentially works same as run() method)
def val_file(models, num_epochs=50):
    probs = []
    for model, gpu, dataloader, optimizer, sched, model_file in models:
        prob_val, val_loss, val_map = val_step(model, gpu, dataloader['val'], 0)
        probs.append(prob_val)
        sched.step(val_loss)

        arrayForMaxAndIndex = []
        for index in range(len(prob_val.get(fileName)[1])):
            # get the highest prob class at each frame from 51 class
            activityAtEachFrameArray = []
            for index1 in range(len(prob_val.get(fileName))):
                activityAtEachFrameArray.append(prob_val.get(fileName)[index1][index])
            maxValue = max(activityAtEachFrameArray)
            indexOfMaxValue = activityAtEachFrameArray.index(maxValue)
            arrayForMaxAndIndex.append([activityList[indexOfMaxValue], maxValue])
        create_caption_video(arrayForMaxAndIndex)
        #print("array for both max and index: ", arrayForMaxAndIndex)




def load_data(train_split, val_split, root):
    # Load Data

    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root, batch_size, classes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                                 pin_memory=True, collate_fn=collate_fn)
        dataloader.root = root
    else:

        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, 'testing', root, batch_size, classes)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2,
                                                 pin_memory=True, collate_fn=collate_fn)
    val_dataloader.root = root

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    return dataloaders, datasets


# train the model
def run(models, criterion, num_epochs=50):
    since = time.time()

    best_map = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        probs = []
        for model, gpu, dataloader, optimizer, sched, model_file in models:
            # Comment out the following line to train the model since using pre train model.
            # train_map, train_loss = train_step(model, gpu, optimizer, dataloader['train'], epoch)
            prob_val, val_loss, val_map = val_step(model, gpu, dataloader['val'], epoch)
            probs.append(prob_val)
            sched.step(val_loss)

            # print("prob_val is: ", prob_val)
            print("number of array is: ", len(prob_val.get('P02T01C07')))
            np.set_printoptions(threshold=sys.maxsize)
            # print("content in first array: ", prob_val.get('P02T01C07')[0])
            print("number of item in first array: ", len(prob_val.get('P18T15C03')[0]))

            if best_map < val_map:
                best_map = val_map
                torch.save(model.state_dict(),
                           './results/' + str(args.model) + '/weight_epoch_' + str(args.lr) + '_' + str(epoch))
                torch.save(model, './results/' + str(args.model) + '/model_epoch_' + str(args.lr) + '_' + str(epoch))
                print('save here:', './results/' + str(args.model) + '/weight_epoch_' + str(args.lr) + '_' + str(epoch))


def eval_model(model, dataloader, baseline=False):
    results = {}
    for data in dataloader:
        other = data[3]
        outputs, loss, probs, _ = run_network(model, data, 0, baseline)
        fps = outputs.size()[1] / other[1][0]

        results[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], data[2].numpy()[0], fps)
    return results


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
        outputs_final = outputs_final[:, 0, :, :]
    # print('outputs_final',outputs_final.size())
    outputs_final = outputs_final.permute(0, 2, 1)
    probs_f = F.sigmoid(outputs_final) * mask.unsqueeze(2)
    loss_f = F.binary_cross_entropy_with_logits(outputs_final, labels, size_average=False)
    loss_f = torch.sum(loss_f) / torch.sum(mask)

    loss = loss_f

    corr = torch.sum(mask)
    tot = torch.sum(mask)

    return outputs_final, loss, probs_f, corr / tot


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
    if args.APtype == 'wap':
        train_map = 100 * apm.value()
    else:
        train_map = 100 * apm.value().mean()
    print('train-map:', train_map)
    apm.reset()

    epoch_loss = tot_loss / num_iter

    return train_map, epoch_loss


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

    epoch_loss = tot_loss / num_iter

    val_map = torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
    print('val-map:', val_map)
    print(100 * apm.value())
    apm.reset()

    return full_probs, epoch_loss, val_map


def create_caption_video(arrayWithCaptions):
    video = args.input_video_full_path
    cap = cv2.VideoCapture(video)
    print("No: ", cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    numberOfFramePerCaption = math.ceil(length / len(arrayWithCaptions))
    # Get video metadata
    video_fps = cap.get(cv2.CAP_PROP_FPS),
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # we are using x264 codec for mp4
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    print(os.getcwd())
    writer = cv2.VideoWriter('./video/output/OUTPUT_VIDEO.mp4', apiPreference=0, fourcc=fourcc,
                             fps=video_fps[0], frameSize=(int(width), int(height)))

    i = 1  # frame counter
    counter = 0  # counter for arrayWithCaptions
    while True:
        # Capture frames in the video
        ret, frame = cap.read()
        # describe the type of font
        # to be used.
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Use putText() method for
        # inserting text on video
        caption = arrayWithCaptions[counter][0]
        if i % numberOfFramePerCaption == 0:
            counter += 1
            caption = arrayWithCaptions[counter][0]
        cv2.putText(frame,
                    caption,
                    (50, 50),
                    font, 1,
                    (0, 0, 0),
                    2,
                    cv2.LINE_4)

        i += 1

        # Display the resulting frame
        # TODO: Get it to run in within the cell as it runs

        # Uncomment to display the external video player frame
        cv2.imshow('video', frame)

        writer.write(frame)

        # creating 'q' as the quit
        # button for the video

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        if cv2.getWindowProperty('video', cv2.WND_PROP_VISIBLE) < 1:
            print("ALL WINDOWS ARE CLOSED")
            cv2.destroyAllWindows()
            break
        if not ret:
            break

    writer.release()
    # release the cap object
    cap.release()
    # close all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import torch
    __spec__ = None
    print(str(args.model))
    print('batch_size:', batch_size)
    print('cuda_avail', torch.cuda.is_available())
    # fileName = input("Type file name: ")
    fileName = args.input_video_file
    # Remove .mp4 from fileName
    fileName = fileName[:-4]
    if args.mode == 'flow':
        print('flow mode', flow_root) #ownself commented
        dataloaders, datasets = load_data(train_split, test_split, flow_root) #ownself commented
    elif args.mode == 'skeleton':
        print('Pose mode', skeleton_root)
        dataloaders, datasets = load_data(train_split, test_split, skeleton_root)
    elif args.mode == 'rgb':
        print('RGB mode', rgb_root)
        dataloaders, datasets = load_data(train_split, test_split, rgb_root)

    if not args.train:
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
            model = torch.load(args.load_model)
            # weight
            # model.load_state_dict(torch.load(str(args.load_model)))
            print("loaded", args.load_model)

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('pytorch_total_params', pytorch_total_params)
        print('num_channel:', num_channel, 'input_channnel:', input_channnel, 'num_classes:', num_classes)
        model.cuda()

        criterion = nn.NLLLoss(reduce=False)
        lr = float(args.lr)
        print(lr)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, verbose=True)
        # run([(model, 0, dataloaders, optimizer, lr_sched, args.comp_info)], criterion, num_epochs=int(args.epoch))
        print(args.test)
        if args.test:
            run([(model, 0, dataloaders, optimizer, lr_sched, args.comp_info)], criterion, num_epochs=int(args.epoch))
        else:
            val_file([(model, 0, dataloaders, optimizer, lr_sched, args.comp_info)], num_epochs=int(args.epoch))
