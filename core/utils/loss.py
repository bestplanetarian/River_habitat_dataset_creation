"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

__all__ = ['MixSoftmaxCrossEntropyLoss', 'MixSoftmaxCrossEntropyOHEMLoss',
           'EncNetLoss', 'ICNetLoss', 'get_segmentation_loss']


# TODO: optim function





class MixSoftmaxFocalLoss(nn.Module):
    """
    Cross-entropy + focal loss for semantic segmentation, with optional aux heads.
    - Background pixels are excluded via ignore_index (default=0).
    - Works with logits of shape (N, C, H, W) and targets (N, H, W).
    - Supports multiple predictions (main + aux). Use aux=True to include them.
    """
    def __init__(
        self,
        aux: bool = True,
        aux_weight: float = 0.2,
        ignore_index: int = 0,          # you said 0 is ignore/bg
        gamma: float = 2.0,
        alpha = [0.0, 2.5, 1.0, 3.5, 1.2],          # (C=5) shadow up, boulder up
        ce_class_weight = [0.0, 2.0, 1.0, 3.0, 1.0],  # CE weights (C,)
        ce_weight: float = 1.0,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        dice_smooth: float = 1.0
    ):
        super().__init__()
        self.aux = aux
        self.aux_weight = aux_weight
        self.ignore_index = int(ignore_index)
        self.gamma = float(gamma)

        self.ce_weight = float(ce_weight)
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.dice_smooth = float(dice_smooth)

        # Focal alpha
        if alpha is None or isinstance(alpha, (float, int)):
            self.register_buffer("alpha", None if alpha is None else torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

        # CE weights
        if ce_class_weight is None:
            self.register_buffer("ce_class_weight", None)
        else:
            self.register_buffer("ce_class_weight", torch.as_tensor(ce_class_weight, dtype=torch.float32))

    def _unpack_preds(self, preds):
        if isinstance(preds, torch.Tensor):
            return [preds]
        if isinstance(preds, (list, tuple)):
            return list(preds)
        if isinstance(preds, dict):
            out = preds["out"] if "out" in preds else next(v for v in preds.values() if isinstance(v, torch.Tensor))
            lst = [out]
            if self.aux and ("aux" in preds) and (preds["aux"] is not None):
                lst.append(preds["aux"])
            return lst
        raise TypeError(f"Unsupported preds type: {type(preds)}")

    def _ce_focal(self, logits: torch.Tensor, target: torch.Tensor):
        valid = (target != self.ignore_index)

        ce = F.cross_entropy(
            logits, target,
            weight=self.ce_class_weight,
            ignore_index=self.ignore_index,
            reduction="none"
        )
        ce = ce * valid

        if self.focal_weight != 0.0:
            log_probs = F.log_softmax(logits, dim=1)
            probs = log_probs.exp()

            safe_target = target.clone()
            safe_target[~valid] = 0
            idx = safe_target.unsqueeze(1)

            p_t = probs.gather(1, idx).squeeze(1)
            log_p_t = log_probs.gather(1, idx).squeeze(1)

            if self.alpha is None:
                alpha_factor = 1.0
            else:
                if self.alpha.ndim == 0:
                    alpha_factor = torch.full_like(p_t, float(self.alpha))
                else:
                    alpha_factor = self.alpha.view(1, -1, 1, 1).gather(1, idx).squeeze(1)

            focal = -alpha_factor * ((1.0 - p_t).clamp(min=1e-8) ** self.gamma) * log_p_t
            focal = focal * valid
        else:
            focal = torch.zeros_like(ce)

        denom = valid.sum().clamp(min=1)
        return (self.ce_weight * ce + self.focal_weight * focal).sum() / denom

    def _dice(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Soft Dice over non-ignored pixels. Computes mean Dice over classes excluding ignore_index label.
        """
        N, C, H, W = logits.shape
        valid = (target != self.ignore_index)

        # probabilities
        probs = F.softmax(logits, dim=1)

        # safe target for one-hot
        safe_target = target.clone()
        safe_target[~valid] = 0
        target_1h = F.one_hot(safe_target, num_classes=C).permute(0, 3, 1, 2).float()  # (N,C,H,W)

        valid_f = valid.unsqueeze(1).float()
        probs = probs * valid_f
        target_1h = target_1h * valid_f

        # exclude ignored label channel if ignore_index is within [0,C-1]
        class_mask = torch.ones(C, device=logits.device, dtype=torch.float32)
        if 0 <= self.ignore_index < C:
            class_mask[self.ignore_index] = 0.0

        intersect = (probs * target_1h).sum(dim=(0, 2, 3))
        denom = probs.sum(dim=(0, 2, 3)) + target_1h.sum(dim=(0, 2, 3))

        dice = (2.0 * intersect + self.dice_smooth) / (denom + self.dice_smooth)
        dice = dice * class_mask

        # average over non-ignored classes
        return 1.0 - (dice.sum() / class_mask.sum().clamp(min=1.0))

    def _loss_once(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self._ce_focal(logits, target)
        if self.dice_weight != 0.0:
            loss = loss + self.dice_weight * self._dice(logits, target)
        return loss

    def forward(self, preds, target: torch.Tensor):
        pred_list = self._unpack_preds(preds)
        loss = self._loss_once(pred_list[0], target)
        if self.aux and len(pred_list) > 1:
            for aux_pred in pred_list[1:]:
                loss = loss + self.aux_weight * self._loss_once(aux_pred, target)
        return {"loss": loss}





'''
class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=0, **kwargs):
        # Set weights: [background, shadow, others]
        class_weights = torch.tensor([0.0, 1.0, 0.1], dtype=torch.float32)
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index, weight=class_weights)
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




class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    '''
    def __init__(
        self,
        aux=True,
        aux_weight=0.2,
        ignore_index=0,
        weight=None,          # <-- NEW: class weights
        **kwargs
    ):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(
            weight= torch.tensor([0.0, 3.0, 1.0, 2.5, 1.0],dtype=torch.float32).cuda(),
            ignore_index=ignore_index
        )
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs):
        *preds, target = tuple(inputs)

        loss = super().forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super().forward(preds[i], target)
            loss = loss + self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])

        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super().forward(*inputs))
    '''
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
'''







def get_segmentation_loss(model, use_ohem=False, loss_type='cross_entropy', **kwargs):
    if loss_type == 'focal':
        return MixSoftmaxFocalLoss(**kwargs)
    if use_ohem:
        return MixSoftmaxCrossEntropyOHEMLoss(**kwargs)

    model = model.lower()
    if model == 'encnet':
        return EncNetLoss(**kwargs)
    elif model == 'icnet':
        return ICNetLoss(**kwargs)
    else:
        return MixSoftmaxCrossEntropyLoss(**kwargs)