import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.gmvae import GMVAE
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=10,    help="Number of latent dimensions")
parser.add_argument('--k',         type=int, default=500,   help="Number mixture components in MoG prior")
parser.add_argument('--iter_max',  type=int, default=20000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
args = parser.parse_args()
layout = [
    ('model={:s}',  'gmvae'),
    ('z={:02d}',  args.z),
    ('k={:03d}',  args.k),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, labeled_subset, _ = ut.get_mnist_data(device, use_test_subset=True)
gmvae = GMVAE(z_dim=args.z, k=args.k, name=model_name).to(device)

ut.load_model_by_name(gmvae, global_step=args.iter_max)
ut.evaluate_lower_bound(gmvae, labeled_subset, run_iwae=False)
samples = torch.reshape(gmvae.sample_x(200), (10, 20, 28, 28))

f, axarr = plt.subplots(10,20)

for i in range(samples.shape[0]):
    for j in range(samples.shape[1]):
        axarr[i,j].imshow(samples[i,j].detach().numpy())
        axarr[i,j].axis('off')

plt.show() 