from typing import Tuple
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader
import numpy as np

from utils.dataset import SWINySEG
from utils.progressbar import ProgressBar


def get_pr_curve(model, weight_path='', dataset_path='./dataset/SWINySEG', split='all') -> Tuple:
    """
    Thresholds: 0-255

    return: 
        p: precision, based on 256 thresholds, shape (256,)
        r: recall, based on 256 thresholds, shape (256,) 
    """
    test_set = SWINySEG(dataset_path, split, 'test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model.set_state_dict(paddle.load(weight_path))
    model.eval()

    bar = ProgressBar(maxStep=len(test_loader))

    precision = paddle.zeros((256, len(test_loader)))
    recall = paddle.zeros((256, len(test_loader)))

    tp = paddle.zeros((256, len(test_loader)))
    fp = paddle.zeros((256, len(test_loader)))
    fn = paddle.zeros((256, len(test_loader)))

    thresholds = paddle.to_tensor(np.array([[[i for i in range(256)]]])).astype('int32')

    for i, (image, label) in enumerate(test_loader()):
        pred = model(image)

        pred_t = F.sigmoid(paddle.squeeze(pred, 1))[0]
        pred_t = pred_t * 255
        label_t = label[0]

        pred_mask = (paddle.unsqueeze(pred_t, -1) > thresholds).astype('int32')
        tfnp = 2 * pred_mask - paddle.unsqueeze(label_t, -1)
        tp[:, i] = paddle.sum((tfnp==1).astype('float32'), axis=(0, 1))
        fp[:, i] = paddle.sum((tfnp==2).astype('float32'), axis=(0, 1))
        fn[:, i] = paddle.sum((tfnp==-1).astype('float32'), axis=(0, 1))

        precision[:, i] = tp[:, i] / (tp[:, i]+fp[:, i])
        recall[:, i] = tp[:, i] / (tp[:, i]+fn[:, i])

        bar.updateBar(i+1, headData={}, endData={})

    p = paddle.mean(precision, axis=1)
    r = paddle.mean(recall, axis=1)
    return p.numpy(), r.numpy()


def get_roc_curve(model, weight_path='', dataset_path='./dataset/SWINySEG', split='all') -> Tuple:
    """
    Thresholds: 0-255

    return: 
        tpr: true positive rate, based on 256 thresholds, shape (256,) 
        fpr: false positive rate, based on 256 thresholds, shape (256,) 
    """
    test_set = SWINySEG(dataset_path, split, 'test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model.set_state_dict(paddle.load(weight_path))
    model.eval()

    bar = ProgressBar(maxStep=len(test_loader))

    fpr = paddle.zeros((256, len(test_loader)))
    tpr = paddle.zeros((256, len(test_loader)))

    tp = paddle.zeros((256, len(test_loader)))
    fp = paddle.zeros((256, len(test_loader)))
    tn = paddle.zeros((256, len(test_loader)))
    fn = paddle.zeros((256, len(test_loader)))

    thresholds = paddle.to_tensor(np.array([[[i for i in range(256)]]])).astype('int32')

    for i, (image, label) in enumerate(test_loader()):
        pred = model(image)

        pred_t = F.sigmoid(paddle.squeeze(pred, 1))[0]
        pred_t = pred_t * 255
        label_t = label[0]

        pred_mask = (paddle.unsqueeze(pred_t, -1) > thresholds).astype('int32')
        tfnp = 2 * pred_mask - paddle.unsqueeze(label_t, -1)
        tp[:, i] = paddle.sum((tfnp==1).astype('float32'), axis=(0, 1))
        fp[:, i] = paddle.sum((tfnp==2).astype('float32'), axis=(0, 1))
        tn[:, i] = paddle.sum((tfnp==0).astype('float32'), axis=(0, 1)) 
        fn[:, i] = paddle.sum((tfnp==-1).astype('float32'), axis=(0, 1))

        fpr[:, i] = fp[:, i] / (fp[:, i]+tn[:, i])
        tpr[:, i] = tp[:, i] / (tp[:, i]+fn[:, i])

        bar.updateBar(i+1, headData={}, endData={})

    tpr = paddle.mean(tpr, axis=1)
    fpr = paddle.mean(fpr, axis=1)
    return tpr.numpy(), fpr.numpy()

def get_metrics(model, weight_path='', dataset_path='./dataset/SWINySEG', split='all') -> Tuple:
    """
    Thresholds: 0.5

    return: precision, recall, F-Measure, error-rate
    """
    test_set = SWINySEG(dataset_path, split, 'test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model.set_state_dict(paddle.load(weight_path))
    model.eval()

    bar = ProgressBar(maxStep=len(test_loader))

    accuracy = 0
    precision = 0
    recall = 0

    for i, (image, label) in enumerate(test_loader()):
        pred = model(image)

        pred_t = (F.sigmoid(paddle.squeeze(pred, 1))[0] > 0.5).astype('int32')
        label_t = label[0]

        tfnp = 2 * pred_t - label_t
        tp = paddle.sum((tfnp==1).astype('float32'))
        fp = paddle.sum((tfnp==2).astype('float32'))
        fn = paddle.sum((tfnp==-1).astype('float32'))

        accuracy += paddle.mean((pred_t==label_t).astype('float32'))
        precision += tp / (tp+fp)
        recall += tp / (tp+fn)

        bar.updateBar(i+1, headData={}, endData={})

    accuracy /= len(test_loader)

    precision /= len(test_loader)
    recall /= len(test_loader)
    f_measure = (2*precision*recall) / (precision+recall)
    error_rate = 1 - accuracy

    return float(precision.numpy()), float(recall.numpy())\
        , float(f_measure.numpy()), float(error_rate.numpy())


