import argparse
import numpy as np
import os
# import tensorflow as tf
import torch
from codebase import utils as ut
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

def train(model, train_loader, labeled_subset, device, tqdm, writer,
          iter_max=np.inf, iter_save=np.inf,
          model_name='model', y_status='none', reinitialize=False):
    # Optimization
    if reinitialize:
        model.apply(ut.reset_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    i = 0
    with tqdm(total=iter_max) as pbar:
        while True:
            for batch_idx, xu in enumerate(train_loader):
                i += 1 # i is num of gradient steps taken by end of loop iteration
                optimizer.zero_grad()

                if y_status == 'none':
                    xu = torch.bernoulli(xu.to(device).reshape(xu.size(0), -1))
                    loss, summaries = model.loss(xu)

                loss.backward()
                optimizer.step()

                # Feel free to modify the progress bar
                if y_status == 'none':
                    pbar.set_postfix(
                        loss='{:.2e}'.format(loss))
                pbar.update(1)

                # Log summaries
                if i % 50 == 0: ut.log_summaries(writer, summaries, i)

                # Save model
                if i % iter_save == 0:
                    ut.save_model_by_name(model, i)

                if i == iter_max:
                    return
