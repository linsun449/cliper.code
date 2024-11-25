import torch
import yaml


def compute_ap(predictions, labels):
    num_class = predictions.size(1)
    ap = torch.zeros(num_class).to(predictions.device)
    empty_class = 0
    for idx_cls in range(num_class):
        prediction = predictions[:, idx_cls]
        label = labels[:, idx_cls]
        if (label > 0).sum() == 0:
            empty_class += 1
            continue
        binary_label = torch.clamp(label, min=0, max=1)
        sorted_pred, sort_idx = prediction.sort(descending=True)
        sorted_label = binary_label[sort_idx]
        tmp = (sorted_label == 1).float()
        tp = tmp.cumsum(0)
        fp = (sorted_label != 1).float().cumsum(0)
        num_pos = binary_label.sum()
        rec = tp / num_pos
        prec = tp / (tp + fp)
        ap_cls = (tmp * prec).sum() / num_pos
        ap[idx_cls].copy_(ap_cls)
    return ap


def compute_f1(predictions, labels, mode_f1, k_val, use_relative=False):
    if k_val >= 1:
        idx = predictions.topk(dim=1, k=k_val)[1]
        predictions.fill_(0)
        predictions.scatter_(dim=1, index=idx,
                             src=torch.ones(predictions.size(0), k_val, dtype=predictions.dtype).to(predictions.device))
    else:
        if use_relative:
            ma = predictions.max(dim=1)[0]
            mi = predictions.min(dim=1)[0]
            step = ma - mi
            thres = mi + k_val * step

            for i in range(predictions.shape[0]):
                predictions[i][predictions[i] > thres[i]] = 1
                predictions[i][predictions[i] <= thres[i]] = 0
        else:
            predictions[predictions > k_val] = 1
            predictions[predictions <= k_val] = 0

    if mode_f1 == 'overall':
        predictions = predictions.bool()
        labels = labels.bool()
        tp = (predictions & labels).sum()
        fp = (predictions & ~labels).sum()
        fn = (~predictions & labels).sum()
        p = tp / (tp + fp + 1.e-9)
        r = tp / (tp + fn + 1.e-9)
        p = p.mean()
        r = r.mean()
        f1 = 2 * p * r / (p + r)
    elif mode_f1 == 'category':
        # calculate P and R
        predictions = predictions.bool()
        labels = labels.bool()
        tp = (predictions & labels).sum(axis=0)
        fp = (predictions & ~labels).sum(axis=0)
        fn = (~predictions & labels).sum(axis=0)
        eps = 1.e-9
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        p = p.mean()
        r = r.mean()
        f1 = 2 * p * r / (p + r)
    elif mode_f1 == 'sample':
        # calculate P and R
        predictions = predictions.bool()
        labels = labels.bool()
        tp = (predictions & labels).sum(axis=1)
        fp = (predictions & ~labels).sum(axis=1)
        fn = (~predictions & labels).sum(axis=1)
        eps = 1.e-9
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        p = p.mean()
        r = r.mean()
        f1 = 2 * p * r / (p + r)

    return f1, p, r


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, Config(**value))
            else:
                setattr(self, key, value)

    def __str__(self):
        str_output = ""
        for key, value in self.__dict__.items():
            str_output += f"{key}: {value}\n"
        return str_output


def load_yaml(filename):
    with open(filename) as file:
        try:
            data = yaml.safe_load(file)
            return Config(**data)

        except yaml.YAMLError as e:
            print(f"Error while loading YAML file: {e}")
