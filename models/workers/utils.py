import torch

def make_samples(x,augment = False):

        x_pos = torch.cat((x[0], x[1]), dim=1)
        x_neg = torch.cat((x[0], x[2]), dim=1)

        if augment:
            x_pos2 = torch.cat((x[1], x[0]), dim=1)
            x_neg2 = torch.cat((x[1], x[2]), dim=1)

            x_pos=torch.cat((x_pos,x_pos2),dim=0)
            x_neg=torch.cat((x_neg,x_neg2),dim=0)

        return  x_pos, x_neg

def make_labels(y):
    bsz = y.size(0) // 2
    slen = y.size(2)
    label = torch.cat((torch.ones(bsz, 1, slen, requires_grad=False), torch.zeros(bsz, 1, slen, requires_grad=False)), dim=0)
    return label