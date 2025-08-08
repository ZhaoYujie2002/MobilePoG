import os.path as osp
import random
import json
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def get_data_info(recording_path):
    """
    Get recording's info from info.json
    """
    with open(osp.join(recording_path, 'info.json'), 'r') as json_file:
        info = json.load(json_file)
    return info

def is_bbox_complete(nbbox):
    x_min, y_min, x_max, y_max = nbbox
    return x_min >= 0 and y_min >= 0 and x_max <= 1 and y_max <= 1

def augment_bbox(image, face_bbox, leye_bbox, reye_bbox):
    h, w, _ = image.shape
    bbox = np.array([face_bbox, leye_bbox, reye_bbox])
    left_limit = np.min(bbox[:, 0])
    right_limit = w - np.max(bbox[:, 2])
    top_limit = np.min(bbox[:, 1])
    bottom_limit = h - np.max(bbox[:, 3])
    bias = round(30 * random.uniform(-1, 1))
    bias = min(bias, left_limit, right_limit, top_limit, bottom_limit, key=abs)

    face_bbox = list(face_bbox)
    leye_bbox = list(leye_bbox)
    reye_bbox = list(reye_bbox)

    for i in range(4):
        face_bbox[i] = face_bbox[i] + int(round(bias))
        leye_bbox[i] = leye_bbox[i] + int(round(bias * 0.5))
        reye_bbox[i] = reye_bbox[i] + int(round(bias * 0.5))
    
    return face_bbox, leye_bbox, reye_bbox

def crop_and_resize(image, bbox, expand_ratio=1.2, target_size=224):
    h, w, _ = image.shape
    
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    
    bbox_size = int(max(bbox_width, bbox_height) * expand_ratio)
    
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    new_x_min = max(0, center_x - bbox_size // 2)
    new_y_min = max(0, center_y - bbox_size // 2)
    new_x_max = min(w, center_x + bbox_size // 2)
    new_y_max = min(h, center_y + bbox_size // 2)
    
    cropped_image = image[new_y_min:new_y_max, new_x_min:new_x_max]
    
    resized_image = cv2.resize(cropped_image, (target_size, target_size))
    
    return resized_image, (new_x_min, new_y_min, new_x_max, new_y_max)

def make_grid(image, bbox, grid_size=25):
    h, w, _ = image.shape
    
    mask = np.zeros((h, w), dtype=np.uint8)
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
    
    mask[y_min:y_max, x_min:x_max] = 1
    
    resized_mask = cv2.resize(mask, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST)
    grid = resized_mask[:, :, np.newaxis]
    
    return grid

class iTrackerDataset(Dataset):
    def __init__(self, recording_path):
        self.recording_path = osp.abspath(recording_path)
        with open(osp.join(recording_path, 'label.json'), 'r') as json_file:
            self.labels = json.load(json_file)
        with open(osp.join(recording_path, 'face.json'), 'r') as json_file:
            self.face_bbox = json.load(json_file)
        self.count = len(list(self.labels.keys()))
        self.frame_indices = []
        for index in range(self.count):
            frame_index = str(index).zfill(5)
            if is_bbox_complete(self.face_bbox[frame_index]['face_nbbox']) and \
                is_bbox_complete(self.face_bbox[frame_index]['leye_nbbox']) and \
                is_bbox_complete(self.face_bbox[frame_index]['reye_nbbox']):
                self.frame_indices.append(frame_index)
        self.count = len(self.frame_indices)
        self.dataset = get_data_info(recording_path)['dataset']
        
    def select_indices(self, indices):
        self.frame_indices = indices
        self.count = len(self.frame_indices)

    def __len__(self):
        return self.count
    
    def __getitem__(self, index):
        frame_index = self.frame_indices[index]
        frame_path = osp.join(self.recording_path, 'frames', frame_index + '.jpg')

        if not osp.isfile(frame_path):
            raise RuntimeError(f'frame_path: {frame_path} error')
        
        image = cv2.imread(frame_path)
        face_bbox = self.face_bbox[frame_index]['face_bbox']
        leye_bbox = self.face_bbox[frame_index]['leye_bbox']
        reye_bbox = self.face_bbox[frame_index]['reye_bbox']
        # do augmentation
        if self.dataset == 'train':
            face_bbox, leye_bbox, reye_bbox = augment_bbox(image, face_bbox, leye_bbox, reye_bbox)

        face_image, _ = crop_and_resize(image, face_bbox, expand_ratio=1.0, target_size=224)
        leye_image, _ = crop_and_resize(image, leye_bbox, expand_ratio=1.2,target_size=224)
        reye_image, _ = crop_and_resize(image, reye_bbox, expand_ratio=1.2,target_size=224)
        grid = make_grid(image, self.face_bbox[frame_index]['face_bbox'], grid_size=25)

        label = [self.labels[frame_index]['label_x'], self.labels[frame_index]['label_y']]

        with open(osp.join(self.recording_path, 'headpose.json'), 'r') as json_file:
            headposes = json.load(json_file)
            headpose = [headposes[frame_index]['pitch'], headposes[frame_index]['yaw'], headposes[frame_index]['roll']]

        transform_to_tensor = transforms.ToTensor()
        data = {
            'face_image': transform_to_tensor(face_image),
            'leye_image': transform_to_tensor(leye_image),
            'reye_image': transform_to_tensor(reye_image),
            'bbox': transform_to_tensor(grid),
            'label': torch.tensor(label), 
            'headpose': torch.tensor(headpose), 
        }

        return data

class AFFNetDataset(Dataset):
    def __init__(self, recording_path):
        self.recording_path = osp.abspath(recording_path)
        with open(osp.join(recording_path, 'label.json'), 'r') as json_file:
            self.labels = json.load(json_file)
        with open(osp.join(recording_path, 'face.json'), 'r') as json_file:
            self.face_bbox = json.load(json_file)
        self.count = len(list(self.labels.keys()))
        self.frame_indices = []
        for index in range(self.count):
            frame_index = str(index).zfill(5)
            if is_bbox_complete(self.face_bbox[frame_index]['face_nbbox']) and \
                is_bbox_complete(self.face_bbox[frame_index]['leye_nbbox']) and \
                is_bbox_complete(self.face_bbox[frame_index]['reye_nbbox']):
                self.frame_indices.append(frame_index)
        self.count = len(self.frame_indices)
        self.dataset = get_data_info(recording_path)['dataset']
    
    def select_indices(self, indices):
        self.frame_indices = indices
        self.count = len(self.frame_indices)
    
    def __len__(self):
        return self.count
    
    def __getitem__(self, index):
        frame_index = self.frame_indices[index]
        frame_path = osp.join(self.recording_path, 'frames', frame_index + '.jpg')

        if not osp.isfile(frame_path):
            raise RuntimeError(f'frame_path: {frame_path} error')
        
        image = cv2.imread(frame_path)
        face_bbox = self.face_bbox[frame_index]['face_bbox']
        leye_bbox = self.face_bbox[frame_index]['leye_bbox']
        reye_bbox = self.face_bbox[frame_index]['reye_bbox']
        # do augmentation
        if self.dataset == 'train':
            face_bbox, leye_bbox, reye_bbox = augment_bbox(image, face_bbox, leye_bbox, reye_bbox)

        face_image, _ = crop_and_resize(image, face_bbox, expand_ratio=1.0, target_size=224)
        leye_image, _ = crop_and_resize(image, leye_bbox, expand_ratio=1.2,target_size=112)
        reye_image, _ = crop_and_resize(image, reye_bbox, expand_ratio=1.2,target_size=112)
        reye_image = cv2.flip(reye_image, 1)
        rect = face_bbox + leye_bbox + reye_bbox

        label = [self.labels[frame_index]['label_x'], self.labels[frame_index]['label_y']]

        with open(osp.join(self.recording_path, 'headpose.json'), 'r') as json_file:
            headposes = json.load(json_file)
            headpose = [headposes[frame_index]['pitch'], headposes[frame_index]['yaw'], headposes[frame_index]['roll']]

        transform_to_tensor = transforms.ToTensor()
        data = {
            'face_image': transform_to_tensor(face_image),
            'leye_image': transform_to_tensor(leye_image),
            'reye_image': transform_to_tensor(reye_image),
            'bbox': torch.tensor(rect).float(), 
            'label': torch.tensor(label),
            'headpose': torch.tensor(headpose), 
        }

        return data


