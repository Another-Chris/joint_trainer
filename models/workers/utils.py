import torch

def make_samples(x):

        x_pos = torch.cat((x[0], x[1]), dim=1)
        x_neg = torch.cat((x[0], x[2]), dim=1)

        return  x_pos, x_neg

def make_labels(y):
    bsz = y.size(0) // 2
    slen = y.size(1)
    label = torch.cat((torch.ones(bsz, slen, requires_grad=False), torch.zeros(bsz, slen, requires_grad=False)), dim=0)
    return label