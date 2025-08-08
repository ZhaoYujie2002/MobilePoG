import yaml
from argparse import ArgumentParser
import numpy as np
import os
import os.path as osp
from tqdm import tqdm

import torch
import utils.misc as misc
from dataset import *
from utils.dmpg_helpers import get_test_subjects, divide_dataloaders
from model import init_model
from utils.cal import frozen_model

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from utils.loss import euclidean_dist


def calibrate(config, subject_dataset, device, logger):
    cal_dataloader, same_eval_dataloader, other_eval_dataloader = divide_dataloaders(config, subject_dataset, sample_method=config['calibrate']['sample_method'])
    model = init_model(config['model'], config['calibrate']['load'])
    model = model.to(device)
    frozen_model(model)

    train_feature = []
    train_output = []
    for data_iter, data in enumerate(tqdm(cal_dataloader, desc=f'making training data')):
        face_image = data['face_image'].to(device)
        leye_image = data['leye_image'].to(device)
        reye_image = data['reye_image'].to(device)
        bbox = data['bbox'].to(device)
        label = data['label'].to(device)
        model_input = [leye_image, reye_image, face_image, bbox]
        feature = model.feature_embedding(model_input)
        train_feature.append(feature.cpu())
        train_output.append(label.cpu())
    train_feature = torch.cat(train_feature, dim=0).numpy()
    train_output = torch.cat(train_output, dim=0).numpy()

    test_same_feature = []
    test_same_output = []
    with torch.no_grad():
        for data_iter, data in enumerate(tqdm(same_eval_dataloader, desc=f'making test-same data')):
            face_image = data['face_image'].to(device)
            leye_image = data['leye_image'].to(device)
            reye_image = data['reye_image'].to(device)
            bbox = data['bbox'].to(device)
            label = data['label'].to(device)
            model_input = [leye_image, reye_image, face_image, bbox]
            feature = model.feature_embedding(model_input)
            test_same_feature.append(feature.cpu())
            test_same_output.append(label.cpu())
    test_same_feature = torch.cat(test_same_feature, dim=0).numpy()
    test_same_output = torch.cat(test_same_output, dim=0).numpy()

    test_other_feature = []
    test_other_output = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data_iter, data in enumerate(tqdm(other_eval_dataloader, desc=f'making test-other data')):
            face_image = data['face_image'].to(device)
            leye_image = data['leye_image'].to(device)
            reye_image = data['reye_image'].to(device)
            bbox = data['bbox'].to(device)
            label = data['label'].to(device)
            model_input = [leye_image, reye_image, face_image, bbox]
            feature = model.feature_embedding(model_input)
            test_other_feature.append(feature.cpu())
            test_other_output.append(label.cpu())
    test_other_feature = torch.cat(test_other_feature, dim=0).numpy()
    test_other_output = torch.cat(test_other_output, dim=0).numpy()

    svr = SVR(kernel='rbf', C=1.0, epsilon=0.2)
    multi_output_svr = MultiOutputRegressor(svr)

    multi_output_svr.fit(train_feature, train_output)

    same_error = euclidean_dist(torch.from_numpy(multi_output_svr.predict(test_same_feature)), torch.from_numpy(test_same_output))
    other_error = euclidean_dist(torch.from_numpy(multi_output_svr.predict(test_other_feature)), torch.from_numpy(test_other_output))

    return same_error.item(), test_same_feature.shape[0], other_error.item(), test_other_feature.shape[0]


def main(args):
    config = args.config
    device = torch.device(f'cuda:{args.gpu}')
    # set seed
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    # set logger
    os.makedirs(config['calibrate']['save']['save_path'], exist_ok=True)
    logger_path = osp.join(config['calibrate']['save']['save_path'], 'calibration.log')
    logger = misc.set_logger('CalibrationLog', logger_path)
    same_total_error = 0
    same_num = 0
    other_total_error = 0
    other_num = 0

    eval_datasets = get_test_subjects(config['dataset'], config['model'])
    for subject in eval_datasets.keys():
        subject_dataset = eval_datasets[subject]
        same_error, same_sample_num, other_error, other_sample_num = calibrate(config, subject_dataset, device, logger)
        same_total_error += same_error * same_sample_num
        same_num += same_sample_num
        other_total_error += other_error * other_sample_num
        other_num += other_sample_num
    
    eval_result = {'avg error': (same_total_error + other_total_error) / (same_num + other_num),
                   'same error': same_total_error / same_num,
                   'other error': other_total_error / other_num}
    
    sample_point_num = config['calibrate']['sample_point_num']
    sample_num = config['calibrate']['sample_num']
    sample_method = config['calibrate']['sample_method']
    with open(osp.join(config['calibrate']['save']['save_path'], f'calibrate_result_{sample_num}_{sample_point_num}_{sample_method}.json'), 'w') as json_file:
        json.dump(eval_result, json_file, indent=4)


if __name__ == '__main__':
    parser = ArgumentParser(description='training code for base model')
    # Config parameters
    parser.add_argument('--config', type=str, required=True, help='Path of the YAML config file')
    parser.add_argument('--gpu', default=0, type=int, help='GPU to use')
    parser.add_argument('--point_num', default=1, type=int, help='The number of calibration points')
    parser.add_argument('--sample_method', default="Continuous", type=str, help='Continuous or Uniform')

    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
        args.config = config
    args.config['calibrate']['sample_point_num'] = args.point_num
    args.config['calibrate']['sample_method'] = args.sample_method
    main(args)