import numpy as np


acc = np.load("accuracy.npy")
f1 = np.load("f1.npy")
auc = np.load("auc.npy")

metric = acc
_metric=[]
metric_mean=[]
for p in range(5,11):
    _metric_p = []
    for i in range(len(metric)):
        _metric_p.append(metric[i][p/10])
    _metric.append(_metric_p)
metric_mean = np.mean(_metric, axis=1)
