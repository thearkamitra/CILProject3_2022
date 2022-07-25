import torch
import torch.nn as nn
import pdb


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-9):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, inp, targ):
        inp = inp.contiguous().permute(0, 2, 3, 1)
        targ = targ.contiguous().permute(0, 2, 3, 1)

        w = torch.zeros((targ.shape[-1],))
        w = 1.0 / (torch.sum(targ, (0, 1, 2)) ** 2 + self.epsilon)

        numerator = targ * inp
        numerator = w * torch.sum(numerator, (0, 1, 2))
        numerator = torch.sum(numerator)

        denominator = targ + inp
        denominator = w * torch.sum(denominator, (0, 1, 2))
        denominator = torch.sum(denominator)

        dice = 2.0 * (numerator + self.epsilon) / (denominator + self.epsilon)

        return 1.0 - dice


def WeightedBCE(input, mask, weight=2):
    """
    Considers a weight to the roads.
    Slight changes can incorporate it for any loss actually. Not sure about the local dice focal loss
    """
    loss_init = nn.functional.binary_cross_entropy(input, mask, reduction="none")
    loss_weighted = loss_init * (1 - mask) + mask * weight * loss_init
    return torch.sum(loss_weighted)


def WeightedBCE2(prediction, label):

    label = label.long()
    mask = label.float()
    num_road = torch.sum((mask == 1).float()).float()
    num_bg = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_bg / (num_road + num_bg)
    mask[mask == 0] = 1.1 * num_road / (num_road + num_bg)
    cost = torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask
    )

    if label.numel() == 0:
        return None  # maybe throw error instead
    else:
        return cost


def BorderLossBCE(input, mask, weight=2):
    """
    Consider a weight to the borders. Same as WeightedBCE for modularity
    """
    loss_init = nn.functional.binary_cross_entropy(input, mask, reduction="none")
    is_border = torch.zeros_like(mask)
    for i in range(-1, 2):
        for j in range(-1, 2):
            try:
                is_border += torch.roll(mask, shifts=(i, j), dims=(2, 3))
            except:
                is_border += torch.roll(mask, shifts=(i, j), dims=(1, 2))
    is_border = torch.where(is_border < 9, 1, 0)
    is_border = is_border * mask
    loss_weighted = loss_init * (1 - is_border) + is_border * weight * loss_init
    return torch.sum(loss_weighted)


def FocalLoss(input, mask, weight=2, gamma=2):
    p_t = input * mask + (1 - input) * (1 - mask)
    ce_loss = nn.functional.binary_cross_entropy(input, mask, reduction="none")
    loss_init = ce_loss * ((1 - p_t) ** gamma)
    loss_weighted = loss_init * (1 - mask) + mask * weight * loss_init
    return torch.sum(loss_weighted)


def DiceLoss(input, mask):
    nr = 1 + torch.sum(2 * input * mask, (1, 2, 3))
    dr = 1 + torch.sum(input + mask, (1, 2, 3))
    dice_coeff = nr / dr
    dice_loss = 1 - torch.mean(dice_coeff)
    return dice_loss


def TverskyLoss(input, mask, weight=0.75):
    nr = 1 + torch.sum(input * mask, (1, 2, 3))
    dr = 1 + torch.sum(
        input * mask + weight * (1 - input) * mask + (1 - weight) * input * (1 - mask),
        (1, 2, 3),
    )
    tv_index = nr / dr
    tvloss = 1 - torch.mean(tv_index)
    return tvloss
