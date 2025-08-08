import yaml
from argparse import ArgumentParser
import numpy as np
import os
import os.path as osp
import random
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import utils.misc as misc
from model import init_model
from engine import train_one_epoch, evaluate_model
from utils.smpg_helpers import get_test_subjects, divide_dataloaders
from utils.cal import frozen_model

def calibrate(config, subject_dataset, device, logger):
    cal_dataloader, same_eval_dataloader, other_eval_dataloader = divide_dataloaders(config, subject_dataset)
    model = init_model(config['model'], config['calibrate']['load'])
    frozen_model(model)
    model = nn.Sequential(
        model,
        nn.Linear(2, 2)
    )
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['calibrate']['params']['lr'], betas=(0.9, 0.95))
    logger.info('start calibrating with fintuning linear layers')
    
    same_error_list = []
    same_sample_num_list = []
    other_error_list = []
    other_sample_num_list = []
    same_error, same_sample_num = evaluate_model(same_eval_dataloader, model, device, logger)
    other_error, other_sample_num =  evaluate_model(other_eval_dataloader, model, device, logger)
    same_error_list.append(same_error)
    same_sample_num_list.append(same_sample_num)
    other_error_list.append(other_error)
    other_sample_num_list.append(other_sample_num)
    for epoch_idx in range(config['calibrate']['params']['epoch']):
        epoch = epoch_idx + 1
        train_one_epoch(optimizer, cal_dataloader, model, epoch, device, logger)
        same_error, same_sample_num = evaluate_model(same_eval_dataloader, model, device, logger)
        other_error, other_sample_num =  evaluate_model(other_eval_dataloader, model, device, logger)
        same_error_list.append(same_error)
        same_sample_num_list.append(same_sample_num)
        other_error_list.append(other_error)
        other_sample_num_list.append(other_sample_num)

    return same_error_list, same_sample_num_list, other_error_list, other_sample_num_list


def main(args):
    config = args.config
    device = torch.device(f'cuda:{args.gpu}')
    # set seed
    seed = config['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # set logger
    os.makedirs(config['calibrate']['save']['save_path'], exist_ok=True)
    logger_path = osp.join(config['calibrate']['save']['save_path'], 'calibration.log')
    logger = misc.set_logger('CalibrationLog', logger_path)
    same_total_error_list = []
    same_num_list = []
    other_total_error_list = []
    other_num_list = []

    eval_datasets = get_test_subjects(config['dataset'], config['model'])
    for subject in eval_datasets.keys():
        subject_dataset = eval_datasets[subject]
        same_error, same_sample_num, other_error, other_sample_num = calibrate(config, subject_dataset, device, logger)
        same_total_error_list.append(same_error)
        same_num_list.append(same_sample_num)
        other_total_error_list.append(other_error)
        other_num_list.append(other_sample_num)
    
    eval_result_list = []
    avg_error_list = []
    same_error_list = []
    other_error_list = []
    for i in range(config['calibrate']['params']['epoch'] + 1):
        same_total_error = 0
        same_num = 0
        other_total_error = 0
        other_num = 0
        for j in range(len(same_total_error_list)):
            same_total_error += same_total_error_list[j][i] * same_num_list[j][i]
            same_num += same_num_list[j][i]
            other_total_error += other_total_error_list[j][i] * other_num_list[j][i]
            other_num += other_num_list[j][i]
        
        eval_result = {'avg error': (same_total_error + other_total_error) / (same_num + other_num),
                    'same error': same_total_error / same_num,
                    'other error': other_total_error / other_num}
        eval_result_list.append(eval_result)
        avg_error_list.append(eval_result['avg error'])
        same_error_list.append(eval_result['same error'])
        other_error_list.append(eval_result['other error'])
    
    sample_point_num = config['calibrate']['sample_point_num']
    pose_num = config['calibrate']['pose_num']
    with open(osp.join(config['calibrate']['save']['save_path'], f'calibrate_result_{pose_num}_{sample_point_num}.json'), 'w') as json_file:
        json.dump(eval_result_list, json_file, indent=4)
    
    epoch_list = range(config['calibrate']['params']['epoch'] + 1)
    plt.figure()
    plt.plot(epoch_list, avg_error_list, label='average error', marker='o')
    plt.plot(epoch_list, same_error_list, label='same error', marker='s')
    plt.plot(epoch_list, other_error_list, label='other error', marker='^')
    plt.title('calibration result')
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.legend()
    plt.savefig(osp.join(config['calibrate']['save']['save_path'], f'calibrate_result_{pose_num}_{sample_point_num}.jpg'))


if __name__ == '__main__':
    parser = ArgumentParser(description='training code for base model')
    # Config parameters
    parser.add_argument('--config', type=str, required=True, help='Path of the YAML config file')
    parser.add_argument('--gpu', default=0, type=int, help='GPU to use')
    parser.add_argument('--point_num', default=1, type=int, help='The number of calibration points')
    parser.add_argument('--pose_num', default=1, type=int, help='The number of calibration poses')

    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
        args.config = config
    args.config['calibrate']['sample_point_num'] = args.point_num
    args.config['calibrate']['pose_num'] = args.pose_num
    main(args)
