import torch

def euclidean_dist(tensor_a: torch.Tensor, tensor_b: torch.Tensor):
    """
    compute mean Euclidean Distance
    params:
        tensor_a(Tensor): [N, 2]
        tensor_b(Tensor): [N, 2]
    """
    squared_error = torch.sum((tensor_a - tensor_b) ** 2, dim=1)
    distance = torch.mean(torch.sqrt(squared_error))
    return distance