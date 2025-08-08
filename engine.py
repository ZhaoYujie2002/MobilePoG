from tqdm import tqdm
import time
import logging
import torch

import utils.misc as misc
import utils.lr_sched as lr_sched
from utils.loss import euclidean_dist

def pretrain_one_epoch(optimizer, train_dataloader, model, epoch, config, args, logger:logging.Logger):
    device = torch.device(args.gpu)
    model.train()
    lr  = lr_sched.adjust_learning_rate(config['train']['params']['lr'], optimizer,
                                        epoch, config['train']['params']['warmup_epoch'],
                                        config['train']['params']['decay_epoch'], config['train']['params']['decay'])
    mse = torch.nn.MSELoss()
    mse.to(device)
    if misc.is_main_process():
        print(f'start training for epoch {epoch}, learning rate {lr}')
        logger.info(f'start training for epoch {epoch}, learning rate {lr}')
    for data_iter, data in enumerate(tqdm(train_dataloader, desc=f'training epoch {epoch}')):
        face_image = data['face_image'].to(device)
        leye_image = data['leye_image'].to(device)
        reye_image = data['reye_image'].to(device)
        bbox = data['bbox'].to(device)
        label = data['label'].to(device)
        model_input = [leye_image, reye_image, face_image, bbox]
        output = model(model_input)
        loss = mse(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if misc.is_main_process():
            loss_scalar = loss.item()
            logger.info(f'epoch {epoch}:[{data_iter + 1 }/{len(train_dataloader)}] training loss: {loss_scalar}')

    return

def evaluate_pretrain_model(test_dataloader, model, args, logger:logging.Logger):
    device = torch.device(args.gpu)
    model.eval()
    error_meter = misc.MultiProcessMeter()
    if misc.is_main_process():
        print(f'start testing')
        logger.info(f'start testing')
    with torch.no_grad():
        for data_iter, data in enumerate(tqdm(test_dataloader, desc=f'testing')):
            face_image = data['face_image'].to(device)
            leye_image = data['leye_image'].to(device)
            reye_image = data['reye_image'].to(device)
            bbox = data['bbox'].to(device)
            label = data['label'].to(device)
            model_input = [leye_image, reye_image, face_image, bbox]
            output = model(model_input)
            dist = euclidean_dist(output, label)
            error_value = dist.item()
            error_meter.update(error_value)

        error_meter.synchronize_between_processes()
        error = error_meter.get_avg()
        if misc.is_main_process():
            print(f'evaluation error: {error}')
            logger.info(f'evaluation error: {error}')

    return error

def train_one_epoch(optimizer, train_dataloader, model, epoch, device, logger:logging.Logger):
    model.to(device)
    l1loss = torch.nn.L1Loss()
    l1loss.to(device)
    print(f'start training for epoch {epoch}')
    logger.info(f'start training for epoch {epoch}')
    for data_iter, data in enumerate(tqdm(train_dataloader, desc=f'training epoch {epoch}')):
        face_image = data['face_image'].to(device)
        leye_image = data['leye_image'].to(device)
        reye_image = data['reye_image'].to(device)
        bbox = data['bbox'].to(device)
        label = data['label'].to(device)
        model_input = [leye_image, reye_image, face_image, bbox]
        output = model(model_input)
        loss = l1loss(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return

def evaluate_model(test_dataloader, model, device, logger:logging.Logger):
    model.eval()
    model.to(device)
    print(f'start testing')
    logger.info(f'start testing')
    total_error = 0
    total_num = 0
    with torch.no_grad():
        for data_iter, data in enumerate(tqdm(test_dataloader, desc=f'testing')):
            face_image = data['face_image'].to(device)
            leye_image = data['leye_image'].to(device)
            reye_image = data['reye_image'].to(device)
            bbox = data['bbox'].to(device)
            label = data['label'].to(device)
            model_input = [leye_image, reye_image, face_image, bbox]
            output = model(model_input)
            dist = euclidean_dist(output, label)
            error_value = dist.item()
            total_error += error_value * face_image.shape[0]
            total_num += face_image.shape[0]

        error = total_error / total_num
        print(f'evaluation error: {error}')
        logger.info(f'evaluation error: {error}')

    return error, total_num
