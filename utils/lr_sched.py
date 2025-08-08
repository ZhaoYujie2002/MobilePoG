"""
code for schedule learning rate
"""
def adjust_learning_rate(lr, optimizer, epoch, warmup_epochs, decay_epochs, decay):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs 
    elif epoch >= decay_epochs:
        lr = lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr