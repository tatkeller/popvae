import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.vae import VAE
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 

from six.moves import urllib    
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=10,     help="Number of latent dimensions")
parser.add_argument('--iter_max',  type=int, default=20000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
args = parser.parse_args()
layout = [
    ('model={:s}',  'vae'),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, labeled_subset, _ = ut.get_mnist_data(device, use_test_subset=True)
vae = VAE(z_dim=args.z, name=model_name).to(device)

ut.load_model_by_name(vae, global_step=args.iter_max)
ut.evaluate_lower_bound(vae, labeled_subset, run_iwae=False)
samples = torch.reshape(vae.sample_x(200), (10, 20, 28, 28))
#print(torch.reshape(vae.sample_x(200), (200, 28, 28)))

f, axarr = plt.subplots(10,20)

for i in range(samples.shape[0]):
    for j in range(samples.shape[1]):
        axarr[i,j].imshow(samples[i,j].detach().numpy())
        axarr[i,j].axis('off')

plt.show() 