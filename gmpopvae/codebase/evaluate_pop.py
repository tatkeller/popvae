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
        for batch_idx, xu in enumerate(test_loader):
            i += 1 # i is num of gradient steps taken by end of loop iteration

            if y_status == 'none':
                xu = torch.bernoulli(xu.to(device).reshape(xu.size(0), -1))
                zu, mu, vu = model.compute_z(xu)

            pbar.update(1)

            # hot = plt.get_cmap('gist_rainbow')
            # #cNorm  = colors.Normalize(vmin=0, vmax=len(yu[0]))
            # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)

            zu = zu.detach().numpy()
            mu = mu.detach().numpy()

            for j, point in enumerate(xu):
                plt.scatter(mu[j,0], 
                            mu[j,1],
                            marker = '.'
                )

            if i == iter_max:
                break

    # Get the labels and handles
    # handles, labels = plt.gca().get_legend_handles_labels()

    # # Filter the labels and handles to remove duplicates
    # newLeg=dict()
    # for h,l in zip(handles,labels):
    #     if l not in newLeg.keys():
    #         newLeg[l]=h

    # # Create new handles and labels
    # handles=[]
    # labels=[]
    # for l in newLeg.keys():
    #     handles.append(newLeg[l])
    #     labels.append(l)

    # # Create new Legend
    # plt.legend(handles, labels)  

    plt.show()