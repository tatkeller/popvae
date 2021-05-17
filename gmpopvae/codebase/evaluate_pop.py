import argparse
import numpy as np
import os
import pandas as pd
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
          model_name='model', y_status='none', reinitialize=False,
          samples = None, meta = None, group_by = None, group_on = "sampleID"):
    i = 0

    if meta:
        cat = False
        meta=pd.read_csv(meta,sep="\t")
        if type(meta[group_by][0]) == str:
            cat = True
            group_by2 = group_by+'cat'
            meta[group_by2] = pd.Categorical(meta[group_by]).codes
            group_by_cat_name = group_by
            group_by = group_by2
        min_color = meta[group_by].min()
        max_color = meta[group_by].max()
    else:
        samples = None

    with tqdm(total=iter_max) as pbar:
        for batch_idx, xu in enumerate(test_loader):
            i += 1 # i is num of gradient steps taken by end of loop iteration

            if samples is not None:
                sample_id = samples[batch_idx]
                this_sample = meta.loc[meta[group_on] == sample_id]
                this_samples_data = this_sample[group_by]
                idx = this_samples_data.index.values[0]
                this_samples_data = this_samples_data.values[0]

            if y_status == 'none':
                xu = torch.bernoulli(xu.to(device).reshape(xu.size(0), -1))
                zu, mu, vu = model.compute_z(xu)

            pbar.update(1)

            if samples is not None:
                if cat:
                    hot = plt.get_cmap('gist_rainbow')
                    cNorm  = colors.Normalize(vmin=min_color, vmax=max_color)
                    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)
                    color_plot = scalarMap.to_rgba(this_samples_data)
                    this_samples_data = meta[group_by_cat_name].iloc[idx]
                else:
                    viridis = plt.get_cmap('viridis')
                    cNorm  = colors.Normalize(vmin=min_color, vmax=max_color)
                    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=viridis)
                    color_plot = scalarMap.to_rgba(this_samples_data)
            else:
                color_plot = 'b'

            zu = zu.detach().numpy()
            mu = mu.detach().numpy()

            plt.scatter(mu[0,0], 
                        mu[0,1],
                        marker = '.',
                        color = color_plot,
                        label = this_samples_data
            )

            if i == iter_max:
                break

    # m, v = model.get_priors()
    # m = m.detach().numpy()[0]
    # v = v.detach().numpy()[0]

    # pi = model.get_weights()
    # pi = pi.detach().numpy()

    # for k, m_k in enumerate(m):
    #     ut.draw_ellipse(m_k, v[k] * np.eye(v[k].shape[-1]), alpha = 0.1)

    # print(m)
    # print("-")
    # print(v)

    if samples is not None:
        if cat:
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
        else:
            plt.colorbar(scalarMap)    
        

    plt.show()