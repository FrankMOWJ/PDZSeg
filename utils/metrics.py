import torch
import numpy as np

def fast_hist(a, b, n):
    """
    Return a histogram that's the confusion matrix of a and b
    :param a: np.ndarray with shape (HxW,)
    :param b: np.ndarray with shape (HxW,)
    :param n: num of classes
    :return: np.ndarray with shape (n, n)
    """
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iou(hist):
    """
    Calculate the IoU(Intersection over Union) for each class
    :param hist: np.ndarray with shape (n, n)
    :return: np.ndarray with shape (n,)
    """
    np.seterr(divide="ignore", invalid="ignore")
    res = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    np.seterr(divide="warn", invalid="warn")
    res[np.isnan(res)] = 0.
    return res


class ComputeIoU(object):
    """
    IoU: Intersection over Union
    """

    def __init__(self, num_class):
        self.num_class = num_class
        self.cfsmatrix = np.zeros((num_class, num_class), dtype="uint64")  # confusion matrix
        self.ious = dict()

    def get_cfsmatrix(self):
        return self.cfsmatrix

    def get_ious(self):
        self.ious = dict(zip(range(self.num_class), per_class_iou(self.cfsmatrix)))  # {0: iou, 1: iou, ...}
        return self.ious

    def get_miou(self, ignore=None):
        self.get_ious()
        total_iou = 0
        count = 0
        for key, value in self.ious.items():
            if isinstance(ignore, list) and key in ignore or \
                    isinstance(ignore, int) and key == ignore:
                continue
            total_iou += value
            count += 1
        return total_iou / count

    def __call__(self, pred, label):
        """
        :param pred: [N, H, W]
        :param label:  [N, H, W}
        Channel == 1
        """

        pred = pred.cpu().numpy()
        label = label.cpu().numpy()

        assert pred.shape == label.shape

        self.cfsmatrix += fast_hist(pred.reshape(-1), label.reshape(-1), self.num_class).astype("uint64")


def dice_coefficient(y_true: torch.Tensor, y_pred: torch.Tensor, class_id: int):
    """
    Calculate Dice coefficient for a specific class using PyTorch tensors.
    
    Args:
    y_true (torch.Tensor): Ground truth mask (shape: H x W)
    y_pred (torch.Tensor): Predicted mask (shape: H x W)
    class_id (int): ID of the class to calculate Dice for
    
    Returns:
    float: Dice coefficient for the specified class
    """
    # Create binary masks for the specific class
    y_true_class = (y_true == class_id).float()
    y_pred_class = (y_pred == class_id).float()
    
    # Calculate intersection and union
    intersection = torch.sum(y_true_class * y_pred_class)
    union = torch.sum(y_true_class) + torch.sum(y_pred_class)
    
    # Calculate Dice coefficient
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    
    return dice.item()

if __name__ == "__main__":
    pass

