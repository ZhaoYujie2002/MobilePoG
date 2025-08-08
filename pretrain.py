import yaml
from argparse import ArgumentParser
import numpy as np
import os
import os.path as osp
from tqdm import tqdm

import torch
from torch.utils.data import ConcatDataset, DataLoader

from torch.utils.data import distributed, RandomSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import utils.misc as misc
from dataset import *
from model import init_model
from engine import pretrain_one_epoch, evaluate_pretrain_model

"""
Code for training base model on multi-gpu
"""

def make_dataloader(model_name, datasets_path, batchsize, num_workers, args):
    dataset = {'train': [], 'test': []}
    for dataset_path in datasets_path:
        for recording in os.listdir(dataset_path):
            recording_path = osp.join(dataset_path, recording)
            info = get_data_info(recording_path)
            if model_name == 'iTracker':
                dataset[info['dataset']].append(iTrackerDataset(recording_path))
            elif model_name == 'AFFNet':
                dataset[info['dataset']].append(AFFNetDataset(recording_path))
    train_dataset = ConcatDataset(dataset['train'])
    test_dataset = ConcatDataset(dataset['test'])
    #------------------------------------------------------
    if args.distributed:
        train_sampler = distributed.DistributedSampler(
            dataset = train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True
        )
        test_sampler = distributed.DistributedSampler(
            dataset = test_dataset,
            num_replicas=args.world_size,
            rank = args.rank,
            shuffle=False
        )
    else:
        train_sampler = RandomSampler(train_dataset)
        test_dataset = RandomSampler(test_dataset)
    #------------------------------------------------------
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batchsize, sampler=train_sampler, 
                                  num_workers=num_workers, drop_last=True, pin_memory=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batchsize, sampler=test_sampler, 
                                 num_workers=num_workers, drop_last=False, pin_memory=False)

    return (train_dataloader, test_dataloader)

def make_test_dataloader(model_name, dataset_path, batchsize, num_workers, args):
    dataset = []
    for recording in os.listdir(dataset_path):
        recording_path = osp.join(dataset_path, recording)
        info = get_data_info(recording_path)
        if info['dataset'] == 'train':
            continue
        if model_name == 'iTracker':
            dataset.append(iTrackerDataset(recording_path))
        elif model_name == 'AFFNet':
            dataset.append(AFFNetDataset(recording_path))
    test_dataset = ConcatDataset(dataset)
    #------------------------------------------------------
    if args.distributed:
        test_sampler = distributed.DistributedSampler(
            dataset = test_dataset,
            num_replicas=args.world_size,
            rank = args.rank,
            shuffle=False
        )
    else:
        test_dataset = RandomSampler(test_dataset)
    #------------------------------------------------------
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batchsize, sampler=test_sampler, 
                                 num_workers=num_workers, drop_last=False, pin_memory=False)

    return test_dataloader

def main(args):
    #------------------------------------------------------
    # Setting for distributed data parallel
    misc.init_distributed_mode(args)
    #------------------------------------------------------
    config = args.config
    # Fix the seed for reproducibilty
    seed = 2024 + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # make dataloaders
    dataloaders = make_dataloader(config['model'], config['dataset'], config['train']['params']['batch_size'], 
                                  config['train']['params']['num_workers'], args)
    train_dataloader, test_dataloader = dataloaders
    test_dataloader = make_test_dataloader(config['model'], config['dataset'][0], config['train']['params']['batch_size'], 
                                  config['train']['params']['num_workers'], args)
    
    if misc.is_main_process():
        os.makedirs(config['train']['save']['save_path'], exist_ok=True)
        logger_path = osp.join(config['train']['save']['save_path'], 'train.log')
        logger = misc.set_logger('TrainingLog', logger_path)
    else:
        logger = None

    model = init_model(config['model'])
    model.cuda(args.gpu)
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=config['train']['params']['lr'], betas=(0.9, 0.95))
    if config['train']['load'] is not None:
        model_without_ddp.load_state_dict(torch.load(config['train']['load']))
        if misc.is_main_process():
            logger.info(f"load model weight from {config['train']['load']}")
    # Train model
    if misc.is_main_process():
        print('start training ' + str(config['model']))
        logger.info('start training ' + str(config['model']))
        eval_result = {}
    for epoch_idx in range(config['train']['params']['epoch']):
        epoch = epoch_idx + 1
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch_idx)
        pretrain_one_epoch(optimizer, train_dataloader, model, epoch, config, args, logger)

        if epoch >= config['train']['save']['save_epoch']:
            if args.distributed:
                test_dataloader.sampler.set_epoch(epoch_idx)
            error = evaluate_pretrain_model(test_dataloader, model, args, logger)
            if misc.is_main_process():
                logger.info(f'epoch {epoch} evaluation error: {error}')
                eval_result[f'epoch {epoch}'] = error
                torch.save(model_without_ddp.state_dict(), osp.join(config['train']['save']['save_path'], f'model_epoch{epoch}.pth'))
    
    if misc.is_main_process():
        with open(osp.join(config['train']['save']['save_path'], 'eval_result.json'), 'w') as json_file:
            json.dump(eval_result, json_file, indent=4)
    misc.cleanup_process()
    
    return


if __name__ == '__main__':
    parser = ArgumentParser(description='training code for base model')
    # Config parameters
    parser.add_argument('--config', type=str, required=True, help='Path of the YAML config file')
    parser.add_argument('--gpu', default=0, type=int, help='GPU to use')
    parser.add_argument('--local_rank', default=-1, type=int)

    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
        args.config = config

    main(args)
    
    
