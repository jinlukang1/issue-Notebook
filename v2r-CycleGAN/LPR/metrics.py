import numpy as np
from sklearn.metrics import confusion_matrix


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.after_iter = 10

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count


class TimeaverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self._avg = 0

    def update(self, val, n=1):
        self.count += n
        self._avg = val / self.count

    @property
    def avg(self):
        return self._avg


class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self, mean=True):
        hist = self.confusion_matrix
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        if mean:
            return mean_iu
        else:
            return iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class miou_calculator:

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros([self.num_class, self.num_class])
        self.miou = 0.0
        self.labels = [i for i in range(num_class)]

    def update_cm(self, pre, gt):
        pre = pre.reshape(-1)
        gt = gt.reshape(-1)
        self.confusion_matrix += confusion_matrix(y_true=gt, y_pred=pre, labels=self.labels)

    def get_miou(self):
        for i in range(self.num_class):
            iou = 0.0
            for j in range(self.num_class):
                iou += self.confusion_matrix[i, j] + self.confusion_matrix[j, i]
            self.miou += self.confusion_matrix[i, i] / (iou - self.confusion_matrix[i, i])
        self.miou /= self.num_class
        return self.miou

    def clear(self):
        self.confusion_matrix = np.zeros([self.num_class, self.num_class])
        self.miou = 0.0
