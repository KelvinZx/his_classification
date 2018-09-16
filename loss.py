from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class WeightCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
    - num_classes (int): number of classes.
    - epsilon (float): weight.
    """

    def __init__(self, num_classes, weight, epsilon=0.1):
        super(WeightCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.weight = weight
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.epsilon = 0.0000001
    def forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (num_classes)
        """

        log_probs = self.logsoftmax(inputs)
        #log_probs = log_probs.clamp(min=self.epsilon, max=1.0 - self.epsilon)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.cuda()
        smooth = torch.pow(log_probs, 0.5)
        loss = (- targets * log_probs * self.weight).mean(0).sum()
        return loss


class FocalWeightCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
    - num_classes (int): number of classes.
    - epsilon (float): weight.
    """

    def __init__(self, num_classes, weight, epsilon=0.1):
        super(FocalWeightCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.weight = weight
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        smooth = torch.pow(inputs, 0.5)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.cuda()
        targets = (1 - self.epsilon) * targets
        loss = (- targets * log_probs * self.weight * smooth).mean(0).sum()
        return loss