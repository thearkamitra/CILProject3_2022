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