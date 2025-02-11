import sklearn.metrics
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np
import matplotlib.pyplot as plt
import sklearn

# 모델을 정의합니다.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    
NNmodel = NeuralNetwork()
NNmodel.load_state_dict(torch.load("model.pth"))

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data_in = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

test_data_out = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

NNmodel.eval()
softmax = nn.Softmax(dim=1)
thres = 0.4

y_true = np.zeros(len(test_data_in) + len(test_data_out))
y_score = np.zeros(len(test_data_in) + len(test_data_out))
ind = 0 
T = 1000
with torch.no_grad():
    print(thres)
    tp_in = 0
    tp_out = 0  
    for i in range(len(test_data_in)):
        x, y = test_data_in[i][0], test_data_in[i][1]
        pred = NNmodel(x)
        probs = softmax(pred/T)
        y_true[ind] = 1
        y_score[ind] = torch.max(probs)
        ind += 1
        if torch.max(probs) >= thres:
            # in-distribution
            tp_in += 1

    print(tp_in)
    for i in range(len(test_data_out)):
        x, y = test_data_out[i][0], test_data_out[i][1]
        pred = NNmodel(x)
        probs = softmax(pred/T)
        y_true[ind] = 0
        y_score[ind] = torch.max(probs)
        ind += 1
        if torch.max(probs) < thres:
            # out-distribution
            tp_out += 1
    print(tp_out)
    print(tp_in + tp_out)
    print("\n")
    
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
roc_auc = sklearn.metrics.auc(fpr, tpr)
display = sklearn.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="ROC curve sample estimator")

display.plot()
plt.show()
print("AUC-ROC")
print(sklearn.metrics.roc_auc_score(y_true, y_score))
print("AUC-PR")
print(sklearn.metrics.average_precision_score(y_true, y_score))





