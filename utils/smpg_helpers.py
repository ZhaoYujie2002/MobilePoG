import os
import os.path as osp
import json
import copy
import numpy as np
import random
from collections import defaultdict

import torch
from torch.utils.data import ConcatDataset, DataLoader
from dataset import AFFNetDataset, iTrackerDataset

def sample_calibration_pose(sample_pose_num):
    POSE = ['1_1', '1_2', '1_3',
            '2_1', '2_2', '2_3',
            '3_1', '3_2', '3_3',
            '4_1', '4_2', '4_3']
    pose_groups = defaultdict(list)
    for pose in POSE:
        row = pose.split('_')[0]
        pose_groups[row].append(pose)
    rows = list(pose_groups.keys())
    num_rows = len(rows)
    base = sample_pose_num // num_rows
    remainder = sample_pose_num % num_rows

    selected = []
    for row in rows:
        selected.extend(random.sample(pose_groups[row], base))
    
    if remainder > 0:
        leftover_rows = random.sample(rows, remainder)
        for row in leftover_rows:
            remaining_candidates = list(set(pose_groups[row]) - set(selected))
            if remaining_candidates:
                selected.append(random.choice(remaining_candidates))
    
    return selected

def sample_from_calibration_points(dataset_path, recording, sample_point_num):
    """
    sample from fixed gaze points for calibrating
    screen direction: 1, 3  [10, 4]
    1: (5, 2)
    3: ()
    5: (0, 0), (0, 4), (10, 0), (10, 4), (5, 2)
    9: (0, 0), (0, 2), (0, 4), (5, 0), (5, 2), (5, 4), (10, 0), (10, 2), (10, 4)
    screen direction: 2, 4 [4, 10]
    1: (2, 5)
    3: ()
    5: (0, 0), (0, 10), (4, 0), (4, 10), (2, 5)
    9: (0, 0), (0, 5), (0, 10), (2, 0), (2, 5), (2, 10), (4, 0), (4, 5), (4, 10)
    """
    screen_direction = int(recording[-3])
    if screen_direction == 1 or screen_direction == 3:
        if sample_point_num == 1:
            sample_points = [(5, 2)]
        elif sample_point_num == 5:
            sample_points = [(0, 0), (0, 4), (10, 0), (10, 4), (5, 2)]
        elif sample_point_num == 9:
            sample_points = [(0, 0), (0, 2), (0, 4), (5, 0), (5, 2), (5, 4), (10, 0), (10, 2), (10, 4)]
        elif sample_point_num == 13:
            sample_points = [(0, 0), (0, 2), (0, 4), (5, 0), (5, 2), (5, 4), (10, 0), (10, 2), (10, 4), 
                             (3, 1), (3, 3), (7, 1), (7, 3)]
    
    elif screen_direction == 2 or screen_direction == 4:
        if sample_point_num == 1:
            sample_points = [(2, 5)]
        elif sample_point_num == 5:
            sample_points = [(0, 0), (0, 10), (4, 0), (4, 10), (2, 5)]
        elif sample_point_num == 9:
            sample_points = [(0, 0), (0, 5), (0, 10), (2, 0), (2, 5), (2, 10), (4, 0), (4, 5), (4, 10)]
        elif sample_point_num == 13:
            sample_points = [(0, 0), (0, 5), (0, 10), (2, 0), (2, 5), (2, 10), (4, 0), (4, 5), (4, 10), 
                             (1, 3), (1, 7), (3, 3), (3, 7)]
    
    with open(osp.join(dataset_path, recording, 'label.json'), 'r') as json_file:
        label = json.load(json_file)
        frame_indices = list(label.keys())
    select_indices = []
    for index in frame_indices:
        if (label[index]['row'], label[index]['col']) in sample_points:
            select_indices.append(index)
    not_select_indices = [item for item in frame_indices if item not in select_indices]
    return select_indices, not_select_indices

POSE = ['1_1', '1_2', '1_3',
        '2_1', '2_2', '2_3',
        '3_1', '3_2', '3_3',
        '4_1', '4_2', '4_3']

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
        pose_data = {'dataset': dataset, 'indices': frame_indices, 
                     'dataset_path': dataset_path, 'recording': recording}
        if info['subject'] not in eval_datasets.keys():
            eval_datasets[info['subject']] = {info['pose']: pose_data}
        else:
            eval_datasets[info['subject']][info['pose']] = pose_data

    return eval_datasets

def divide_dataloaders(config, subject_dataset, random_sample=False):
    eval_datasets = []
    calibrate_datasets = []
    same_pose_eval_datasets = []
    # select_pose_list = random.sample(POSE, config['calibrate']['pose_num'])
    select_pose_list = sample_calibration_pose(config['calibrate']['pose_num'])
    for pose in POSE:
        if pose not in select_pose_list:
            eval_datasets.append(subject_dataset[pose]['dataset'])
        else:
            if random_sample:
                frame_indices = subject_dataset[pose]['indices']
                select_indices = random.sample(frame_indices, config['calibrate']['sample_num'])
                not_select_indices = [item for item in frame_indices if item not in select_indices]
            else:
                dataset_path = subject_dataset[pose]['dataset_path']
                recording = subject_dataset[pose]['recording']
                select_indices, not_select_indices = sample_from_calibration_points(dataset_path, recording, config['calibrate']['sample_point_num'])
            dataset = copy.deepcopy(subject_dataset[pose]['dataset'])
            dataset.select_indices(select_indices)
            calibrate_datasets.append(dataset)
            dataset = copy.deepcopy(subject_dataset[pose]['dataset'])
            dataset.select_indices(not_select_indices)
            same_pose_eval_datasets.append(dataset)
    
    # print(len(eval_datasets), len(calibrate_datasets), len(same_pose_eval_datasets))
    # print(calibrate_datasets[0].count)
    other_eval_dataloader = DataLoader(ConcatDataset(eval_datasets), batch_size=config['calibrate']['params']['batch_size'], \
                                 num_workers=config['calibrate']['params']['num_workers'], drop_last=False, pin_memory=False)
    cal_dataloader = DataLoader(ConcatDataset(calibrate_datasets), batch_size=config['calibrate']['params']['batch_size'], \
                                 num_workers=config['calibrate']['params']['num_workers'], drop_last=False, pin_memory=False)
    same_eval_dataloader = DataLoader(ConcatDataset(same_pose_eval_datasets), batch_size=config['calibrate']['params']['batch_size'], \
                                 num_workers=config['calibrate']['params']['num_workers'], drop_last=False, pin_memory=False)

    return cal_dataloader, same_eval_dataloader, other_eval_dataloader
