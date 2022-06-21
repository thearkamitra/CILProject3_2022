import torch
import torch.nn as nn

class GeneralizedDiceLoss(nn.Module):
    def __init__(self, epsilon= 1e-9):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
    def forward(self, inp, targ):
        inp = inp.contiguous().permute(0, 2, 3, 1)
        targ = targ.contiguous().permute(0, 2, 3, 1)

        w = torch.zeros((targ.shape[-1],))
        w = 1. / (torch.sum(targ, (0, 1, 2))**2 + self.epsilon)

        numerator = targ * inp
        numerator = w * torch.sum(numerator, (0, 1, 2))
        numerator = torch.sum(numerator)

        denominator = targ + inp
        denominator = w * torch.sum(denominator, (0, 1, 2))
        denominator = torch.sum(denominator)

        dice = 2. * (numerator + self.epsilon) / (denominator + self.epsilon)

        return 1. - dice

def WeightedBCE(input, mask, weight):
    """
        Considers a weight to the roads.
        Slight changes can incorporate it for any loss actually. Not sure about the local dice focal loss
    """
    loss_init = nn.functional.binary_cross_entropy(input, mask, reduction="none")
    loss_weighted = loss_init*(1- mask) + mask*weight* loss_init
    return torch.sum(loss_weighted)

def BorderLossBCE(input, mask, weight):
    """
    Consider a weight to the borders. Same as WeightedBCE for modularity
    """
    loss_init = nn.functional.binary_cross_entropy(input, mask, reduction="none")
    is_border = torch.ones_like(mask)
    for i in range(-1,2):
        for j in range(-1,2):
            is_border *= torch.roll(mask, shifts=(i,j), dims=(1,2))

    loss_weighted = loss_init*(1- is_border) + is_border*weight* loss_init
    return torch.sum(loss_weighted)
