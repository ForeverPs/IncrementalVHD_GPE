import torch
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.metrics import average_precision_score, precision_score, recall_score


def eval(pred, pred_cls, gt):
    if torch.sum(gt) > 0:
        try:
            pred = pred.detach().cpu().squeeze().numpy()
            pred_cls = pred_cls.detach().cpu().squeeze().numpy()
            gt = gt.detach().cpu().squeeze().numpy()
            acc = np.sum(pred_cls == gt) / len(gt) if len(gt) else 0
            precision, recall, thresholds = precision_recall_curve(gt, pred)
            pr_auc = auc(recall, precision)
            roc_auc = roc_auc_score(gt, pred)
            p_score, r_score = precision_score(gt, pred_cls, zero_division=0), recall_score(gt, pred_cls, zero_division=0)
            ap = average_precision_score(gt, pred_cls)
            return pr_auc, roc_auc, p_score, r_score, ap, acc
        except:
            return 0, 0, 0, 0, 0, 0
    else:
        return 0, 0, 0, 0, 0, 0


if __name__ == '__main__':
    pred = torch.rand(100,)

    pred_cls = pred.clone()
    pred_cls[pred_cls < 0.5] = 0
    pred_cls[pred_cls >= 0.5] = 1

    gt = torch.rand(100,)
    gt[gt < 0.5] = 0
    gt[gt >= 0.5] = 1

    # eval(pred, gt)
    print(eval(pred, pred_cls, gt))
