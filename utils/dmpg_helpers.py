import os
import os.path as osp
import json
import random
import copy
import numpy as np

import torch
from torch.utils.data import ConcatDataset, DataLoader
from dataset import iTrackerDataset, AFFNetDataset

def get_calibration_points(calibration_point_num):
    """
    sample from fixed gaze points for calibrating
    [0, 1, 2, 
     3, 4, 5,
     6, 7, 8,
     9, 10, 11]
    1: [4]
    2: [4, 7]
    4: [0, 2, 9, 11]
    6: [0, 2, 9, 11, 4, 7]
    """
    if calibration_point_num == 1:
        point_indices = [4]
    elif calibration_point_num == 2:
        point_indices = [4, 7]
    elif calibration_point_num == 4:
        point_indices = [0, 2, 9, 11]
    elif calibration_point_num == 6:
        point_indices = [0, 2, 9, 11, 4, 7]
    return point_indices

def get_test_subjects(dataset_path, model_name):
    eval_datasets = {}
    for recording in sorted(os.listdir(dataset_path)):
        recording_path = osp.join(dataset_path, recording)
        with open(osp.join(recording_path, 'info.json'), 'r') as json_file:
            info = json.load(json_file)
        if info['dataset'] == 'train':
            continue
        if model_name == 'iTracker':
            dataset = iTrackerDataset(recording_path)
        elif model_name == 'AFFNet':
            dataset = AFFNetDataset(recording_path)
        with open(osp.join(recording_path, 'face.json'), 'r') as json_file:
            face = json.load(json_file)
            frame_indices = list(face.keys())
        point_data = {'dataset': dataset, 'indices': frame_indices, 
                     'dataset_path': dataset_path, 'recording': recording}
        if info['subject'] not in eval_datasets.keys():
            eval_datasets[info['subject']] = {info['point']: point_data}
        else:
            eval_datasets[info['subject']][info['point']] = point_data

    return eval_datasets

def divide_dataloaders(config, subject_dataset, sample_method='Uniform'):
    eval_datasets = []
    calibrate_datasets = []
    same_point_eval_datasets = []
    calibration_point_indices = get_calibration_points(config['calibrate']['sample_point_num'])
    for point in range(12):
        if point not in calibration_point_indices:
            eval_datasets.append(subject_dataset[str(point)]['dataset'])
        else:
            frame_indices = subject_dataset[str(point)]['indices']
            if sample_method == 'Uniform':
                step = (len(frame_indices) - 1) / (config['calibrate']['sample_num'] - 1)
                uniform_indices = [round(i * step) for i in range(config['calibrate']['sample_num'])]
                select_indices = [frame_indices[i] for i in uniform_indices]
            elif sample_method == 'Continuous':
                # start_index = random.randint(0, len(frame_indices) - config['calibrate']['sample_num'])
                start_index = 0
                select_indices = frame_indices[start_index:start_index + config['calibrate']['sample_num']]
            
            not_select_indices = [item for item in frame_indices if item not in select_indices]
            dataset = copy.deepcopy(subject_dataset[str(point)]['dataset'])
            dataset.select_indices(select_indices)
            calibrate_datasets.append(dataset)
            dataset = copy.deepcopy(subject_dataset[str(point)]['dataset'])
            dataset.select_indices(not_select_indices)
            same_point_eval_datasets.append(dataset)
    
    # print(len(eval_datasets), len(calibrate_datasets), len(same_point_eval_datasets))
    # print(calibrate_datasets[0].count)
    other_eval_dataloader = DataLoader(ConcatDataset(eval_datasets), batch_size=config['calibrate']['params']['batch_size'], \
                                 num_workers=config['calibrate']['params']['num_workers'], drop_last=False, pin_memory=False)
    cal_dataloader = DataLoader(ConcatDataset(calibrate_datasets), batch_size=config['calibrate']['params']['batch_size'], \
                                 num_workers=config['calibrate']['params']['num_workers'], drop_last=False, pin_memory=False)
    same_eval_dataloader = DataLoader(ConcatDataset(same_point_eval_datasets), batch_size=config['calibrate']['params']['batch_size'], \
                                 num_workers=config['calibrate']['params']['num_workers'], drop_last=False, pin_memory=False)

    return cal_dataloader, same_eval_dataloader, other_eval_dataloader