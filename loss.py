"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

__all__ = ['MixSoftmaxCrossEntropyLoss', 'MixSoftmaxCrossEntropyOHEMLoss',
           'EncNetLoss', 'ICNetLoss', 'get_segmentation_loss']


# TODO: optim function

'''
class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=0, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs))
'''

'''
class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=0,
                 num_classes=3, alpha=None, gamma=2.0):
        super().__init__()
        self.aux = aux
        self.aux_weight = aux_weight
        self.num_classes = num_classes
        self.gamma = gamma
        self.ignore_index = ignore_index

        # alpha: [bg, others, shadow]; bg=0 disables loss on it
        if alpha is None:
            alpha = torch.tensor([0.0, 0.2, 10], dtype=torch.float32)
        self.register_buffer('alpha', alpha)

    def _focal_loss(self, pred, target):
        ce_loss = F.cross_entropy(
            pred, target, weight=self.alpha, ignore_index=self.ignore_index, reduction='none'
        )
        pt = torch.exp(-ce_loss)  # Estimate p_t
        focal = ((1 - pt) ** self.gamma) * ce_loss

        # Mask out background (class 0) and ignore_index
        #valid_mask = (target != 0)
        #focal = focal[valid_mask]
        return focal.mean()

    def forward(self, preds, target):
        loss = self._focal_loss(preds[0], target)
        if self.aux:
            for i in range(1, len(preds)):
                loss += self.aux_weight * self._focal_loss(preds[i], target)
        return dict(loss=loss)
'''










class  MixSoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=0, 
                 class_weights=None, recall_alpha=0.3, epsilon=1e-6):
        """
        Args:
            aux: Whether to use auxiliary outputs
            aux_weight: Weight for auxiliary losses
            ignore_index: Class index to ignore (background)
            class_weights: Per-class weights [background, shadow, others]
            recall_alpha: Weight for recall-focused term (0-1)
            epsilon: Small value to prevent division by zero
        """
        super().__init__()
        self.aux = aux
        self.aux_weight = aux_weight
        self.recall_alpha = recall_alpha
        self.epsilon = epsilon
        
        # Default weights if not provided
        
        if class_weights is None:
            class_weights = torch.tensor([0, 0.2, 1, 0.03,0.06])  # [bg, shadow, others]
        self.register_buffer('class_weights', class_weights)
        
        
        # Base CE loss with class weights
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=ignore_index
        )
        
    def _recall_term(self, pred, target):
        """Component that directly optimizes shadow recall"""
        pred_shadow = pred[:, 2]  # Shadow channel logits
        target_shadow = (target == 2).float()
        
        # Shadow probability (sigmoid for binary case)
        shadow_probs = torch.sigmoid(pred_shadow)
        
        # Calculate recall components
        tp = (shadow_probs * target_shadow).sum()
        fn = ((1 - shadow_probs) * target_shadow).sum()
        
        # Recall = TP / (TP + FN)
        shadow_recall = tp / (tp + fn + self.epsilon)
        
        # We minimize (1 - recall) to maximize recall
        return 1.0 - shadow_recall
    
    def _aux_forward(self, *inputs):
        *preds, target = tuple(inputs)
        
        # Main loss
        main_loss = self.ce_loss(preds[0], target)
        recall_loss = self._recall_term(preds[0], target)
        total_loss = (1 - self.recall_alpha) * main_loss + self.recall_alpha * recall_loss
        
        # Auxiliary losses
        for i in range(1, len(preds)):
            aux_ce = self.ce_loss(preds[i], target)
            aux_recall = self._recall_term(preds[i], target)
            total_loss += self.aux_weight * (
                  aux_ce  +  aux_recall)
            
        return total_loss
    
    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            main_loss = self.ce_loss(preds[0], target)
            recall_loss = self._recall_term(preds[0], target)
            return dict(loss= main_loss + recall_loss)



#To balance class weights











'''
class  MixSoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=0,
                 class_weights=None, pr_alpha=0.5, epsilon=1e-6):
        super().__init__()
        self.aux = aux
        self.aux_weight = aux_weight
        self.pr_alpha = pr_alpha  # Precision-recall balance
        self.epsilon = epsilon
        
        if class_weights is None:
            class_weights = torch.tensor([0, 0.4, 10])
        self.register_buffer('class_weights', class_weights)
        
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=ignore_index
        )
    
    def _precision_recall_term(self, pred, target):
        pred_shadow = pred[:, 2]
        target_shadow = (target == 2).float()
        shadow_probs = torch.sigmoid(pred_shadow)
        
        tp = (shadow_probs * target_shadow).sum()
        fp = (shadow_probs * (1 - target_shadow)).sum()
        fn = ((1 - shadow_probs) * target_shadow).sum()
        
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        
        return   (1 - precision)+  (1 - recall)
    
    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        
        # Main loss
        ce_loss = self.ce_loss(preds[0], target)
        pr_loss = self._precision_recall_term(preds[0], target)
        total_loss =  ce_loss + pr_loss  # 70% CE, 30% PR
        
        # Auxiliary losses
        if self.aux:
            for i in range(1, len(preds)):
                aux_ce = self.ce_loss(preds[i], target)
                aux_pr = self._precision_recall_term(preds[i], target)
                total_loss += self.aux_weight * (0.4 * aux_ce + 0.6 * aux_pr)
        
        return dict(loss=total_loss)

'''










'''

# reference: https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/nn/loss.py
class EncNetLoss(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with SE Loss"""

    def __init__(self, se_loss=True, se_weight=0.2, nclass=19, aux=False,
                 aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(EncNetLoss, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if not self.se_loss and not self.aux:
            return super(EncNetLoss, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            return dict(loss=loss1 + self.aux_weight * loss2)
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(EncNetLoss, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.se_weight * loss2)
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.aux_weight * loss2 + self.se_weight * loss3)

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect


# TODO: optim function
class ICNetLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for ICNet"""

    def __init__(self, nclass, aux_weight=0.4, ignore_index=0, **kwargs):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index)
        self.nclass = nclass
        self.aux_weight = aux_weight

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])

        pred, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)
        # [batch, W, H] -> [batch, 1, W, H]
        target = target.unsqueeze(1).float()
        target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode='bilinear', align_corners=True).squeeze(
            1).long()
        loss1 = super(ICNetLoss, self).forward(pred_sub4, target_sub4)
        loss2 = super(ICNetLoss, self).forward(pred_sub8, target_sub8)
        loss3 = super(ICNetLoss, self).forward(pred_sub16, target_sub16)
        return dict(loss=loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight)


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000, use_weight=True, **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                        1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        n, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)

        if self.min_kept > num_valid:
            print("Lables: {}".format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()

        target = target.masked_fill_(1 - valid_mask, self.ignore_index)
        target = target.view(n, h, w)

        return self.criterion(pred, target)


class MixSoftmaxCrossEntropyOHEMLoss(OhemCrossEntropy2d):
    def __init__(self, aux=False, aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs))
'''

def get_segmentation_loss(model, use_ohem=False, **kwargs):
    if use_ohem:
        return MixSoftmaxCrossEntropyOHEMLoss(**kwargs)

    model = model.lower()
    if model == 'encnet':
        return EncNetLoss(**kwargs)
    elif model == 'icnet':
        return ICNetLoss(**kwargs)
    else:
        return MixSoftmaxCrossEntropyLoss(**kwargs)
