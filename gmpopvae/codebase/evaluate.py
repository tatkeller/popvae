import argparse
import numpy as np
import os
# import tensorflow as tf
import torch
from codebase import utils as ut
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cmx


def evaluate(model, test_loader, labeled_subset, device, tqdm,
          iter_max=np.inf, iter_save=np.inf,
          model_name='model', y_status='none', reinitialize=False):
    i = 0
    with tqdm(total=iter_max) as pbar:
        for batch_idx, (xu, yu) in enumerate(test_loader):
            i += 1 # i is num of gradient steps taken by end of loop iteration

            if y_status == 'none':
                xu = torch.bernoulli(xu.to(device).reshape(xu.size(0), -1))
                yu = yu.new(np.eye(10)[yu]).to(device).float()
                zu, mu, vu = model.compute_z(xu)

            pbar.update(1)

            hot = plt.get_cmap('gist_rainbow')
            cNorm  = colors.Normalize(vmin=0, vmax=len(yu[0]))
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)

            yu = yu.detach().numpy()
            zu = zu.detach().numpy()

            for j, label in enumerate(yu):
                plt.scatter(zu[j,0], 
                            zu[j,1],
                            marker = '.',
                            color=scalarMap.to_rgba(np.argmax(label)),
                            label=np.argmax(label)
                )

            if i == iter_max:
                break

    # Get the labels and handles
    handles, labels = plt.gca().get_legend_handles_labels()

    # Filter the labels and handles to remove duplicates
    newLeg=dict()
    for h,l in zip(handles,labels):
        if l not in newLeg.keys():
            newLeg[l]=h

    # Create new handles and labels
    handles=[]
    labels=[]
    for l in newLeg.keys():
        handles.append(newLeg[l])
        labels.append(l)

    # Create new Legend
    plt.legend(handles, labels)  

    plt.show()