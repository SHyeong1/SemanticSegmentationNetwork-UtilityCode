import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'bdice':
            return self.BinaryDiceLoss
        elif mode == 'dice':
            return self.DiceLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        target[target >= c] = self.ignore_index
        target[target < 0] = self.ignore_index
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def  BinaryDiceLoss(self,predict,target,smooth=1, p=2, reduction='mean'):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + smooth
        den = torch.sum(predict.pow(p) + target.pow(p), dim=1) + smooth

        loss = 1 - num / den

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        
    def DiceLoss(self,predict, target,smooth=1, p=2, reduction='mean'):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = self.BinaryDiceLoss(predict[:, i], target[:, i],smooth, p, reduction)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]
    
class MSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(MSELoss, self).__init__(**kwargs)
        self.mse = torch.nn.MSELoss()
        
    def forward(self, preds, target):
        return dict(loss=self.mse(preds,target))
class CriterionKlDivergence(nn.Module):
    '''
    '''
    def __init__(self):
        super(CriterionKlDivergence,self).__init__()
        self.criterion_kd = nn.KLDivLoss()
    def forward(self,s_feature,t_feature):
        #assert not t_feature.requires_grad
        assert s_feature.dim() == 4
        assert t_feature.dim() == 4
        assert s_feature.size(0) == t_feature.size(0),'{0} vs {1}'.format(s_feature.size(0),t_feature.size(0))
        assert s_feature.size(2) == t_feature.size(2),'{0} vs {1}'.format(s_feature.size(2),t_feature.size(2))
        assert s_feature.size(3) == t_feature.size(3),'{0} vs {1}'.format(s_feature.size(3),t_feature.size(3))
        return self.criterion_kd(F.log_softmax(s_feature),F.softmax(t_feature))

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




