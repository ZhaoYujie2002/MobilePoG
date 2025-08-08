import torch
import torch.nn as nn

def frozen_model(model):
    for n, p in model.named_parameters():
        if 'prompt_embeddings_lr' not in n and 'prompt_embeddings_tb' not in n:
            p.requires_grad = False
    model.eval()

def frozen_model_backbone(model):
    for n, p in model.named_parameters():
        p.requires_grad = False
    model.eval()
    for n, p in model.get_regression_head().named_parameters():
        p.requires_grad = True