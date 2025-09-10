import numpy as np
import torch
from texttable import Texttable


class ShowSegmentResult:

    def __init__(self, num_classes=21, ignore_labels=None):
        """
        Args:
            num_classes: float, class number with background
        """
        if ignore_labels is None:
            ignore_labels = [255]
        self.ignore_labels = ignore_labels
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.result = None

    @staticmethod
    def _fast_hist(label_true, label_pred, num_classes):
        """
        Args:
            label_true: size-->[h*w] the true class label map flatted with background
            label_pred: size-->[h*w] the predicted class label map flatted with background
            num_classes: float class number with background

        Returns: confusion matrix of segment in pixel
        """
        mask = (label_true >= 0) & (label_true < num_classes)
        hist = np.bincount(
            num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=num_classes ** 2,
        )
        return hist.reshape(num_classes, num_classes)

    def add_prediction(self, label_true, label_pred):
        """
        Args:
            label_true: numpy size-->[h, w] the true class label map with background
            label_pred: numpy size-->[h, w] the predicted class label map with background

        Returns: a dict like {"pAcc": acc, "mAcc": acc_cls, "mIoU": mean_iu, "IoU": cls_iu}
        """

        # Create a mask to ignore labels 255 and 0
        ignore_mask = ~(np.isin(label_true, self.ignore_labels))

        # Apply the mask and flatten the arrays
        label_true_flat = label_true[ignore_mask].flatten()
        label_pred_flat = label_pred[ignore_mask].flatten()

        self.hist += self._fast_hist(label_true_flat, label_pred_flat, self.num_classes)

    def calculate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        _acc_cls = np.diag(self.hist) / (self.hist.sum(axis=1) + 1e-5)
        acc_cls = np.nanmean(_acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist) + 1e-5)
        valid = self.hist.sum(axis=1) > 0
        mean_iu = np.nanmean(iu[valid])
        cls_iu = dict(zip(range(self.num_classes), iu))

        self.result = {"pAcc": acc, "mAcc": acc_cls, "mIoU": mean_iu, "IoU": cls_iu}

        return self.result

    def clear_prediction(self):
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.result = None


def cam_to_label(cam, cls_label, bkg_thre=0.3, cls_thre=0.4, is_normalize=True):
    """
    Args:
        is_normalize:
        cam: size-->[bs, n_cls, h, w] without the background
        cls_label: size-->[bs, n_cls] identify the class id of this picture without background
        bkg_thre: float, identify the min thresh of a class cam [0-1]
        cls_thre: float, identify the classification thresh [0-1]
    Returns:
        _pseudo_label: size-->[bs, h, w] the class label map with background
    """
    b, c, h, w = cam.shape

    cls_label_rep = (cls_label > cls_thre).unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    reshape_cam = cam.reshape(b, c, -1)
    if is_normalize:
        reshape_cam -= reshape_cam.amin(dim=-1, keepdim=True)
        reshape_cam /= reshape_cam.amax(dim=-1, keepdim=True) + 1e-6
    if bkg_thre > 0:
        reshape_cam[reshape_cam < bkg_thre] = 0
        reshape_cam = reshape_cam.reshape(b, c, h, w)

        valid_cam = cls_label_rep * reshape_cam
        cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
        _pseudo_label += 1
        _pseudo_label[cam_value == 0] = 0
    else:
        reshape_cam = reshape_cam.reshape(b, c, h, w)
        valid_cam = cls_label_rep * reshape_cam
        valid_cam = torch.cat([torch.pow(1 - valid_cam.amax(dim=1, keepdim=True), 2), valid_cam], dim=1)
        _pseudo_label = valid_cam.argmax(dim=1)
    return valid_cam, _pseudo_label
