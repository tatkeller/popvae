import numpy as np
from scipy import linalg
import os
import shutil
# import tensorflow as tf
import torch
from codebase.models.gmvae import GMVAE
from codebase.models.ssvae import SSVAE
from codebase.models.vae import VAE
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 
import matplotlib as mpl


bce = torch.nn.BCEWithLogitsLoss(reduction='none')
mse = torch.nn.MSELoss(reduction='none')
################################################################################
# Please familiarize yourself with the code below.
#
# Note that the notation is
# argument: argument_type: argument_shape
#
# Furthermore, the expected argument_shape is only a guideline. You're free to
# pass in inputs that violate the expected argument_shape provided you know
# what you're doing
################################################################################

def sample_gaussian(m, v):
    """
    Element-wise application reparameterization trick to sample from Gaussian

    Args:
        m: tensor: (batch, ...): Mean
        v: tensor: (batch, ...): Variance

    Return:
        z: tensor: (batch, dim): Samples
    """
    # Create a standard/unit multivariate normal distibution in PyTorch
    # The standard mean and variance will have the same shape as the last dim of the input
    unit_normal = torch.distributions.MultivariateNormal(torch.zeros(m.shape[1]), torch.eye(v.shape[1]))

    # Sample the number of batches that are present in the mean and variance
    e = unit_normal.sample(sample_shape=torch.Size([m.shape[0]]))

    # Perform the reparameterization trick by multiplying the samples with
    # the respective variances and adding that result to the respective means
    z = m + (torch.sqrt(v) * e)
    return z



def log_normal(x, m, v):
    """
    Computes the elem-wise log probability of a Gaussian and then sum over the
    last dim. Basically we're assuming all dims are batch dims except for the
    last dim.

    Args:
        x: tensor: (batch_1, batch_2, ..., batch_k, dim): Observation
        m: tensor: (batch_1, batch_2, ..., batch_k, dim): Mean
        v: tensor: (batch_1, batch_2, ..., batch_k, dim): Variance

    Return:
        log_prob: tensor: (batch_1, batch_2, ..., batch_k): log probability of
            each sample. Note that the summation dimension is not kept
    """
    # TODO: Implement this

    # The input shape of mean and variance to this function is
    # (batches, k, dim)

    # Grab n, which will be needed for the log of the multivariate gaussian function
    n = m.shape[2]

    # Reshape the X or Z input such that there is a dimension for mixtures (if there are any)
    z_r = x.view(x.shape[0], 1, x.shape[1])

    # The log of denominator of the multivariate gaussian function
    # The determinant of sigma is simply the product of the variance along the dims dimension (sigma is diag)
    div_term = 0.5 * torch.log(torch.prod(v, axis = 2)) + (n/2 * np.log(2*np.pi))

    # The numerator of the multivariate gaussian function
    # The resulting value is the sum along the dims dimension 
    # In the case of k = 1, this is already the log of the exponential, since the terms cancel out
    exp_term = -0.5 * torch.sum((z_r - m) * (1/v) * (z_r - m), dim = 2)

    # Finally, perform the subtractions of the logged values accordingly
    log_prob = (exp_term - div_term)

    return log_prob

def log_normal_mixture(z, m, v):
    """
    Computes log probability of Gaussian mixture.

    Args:
        z: tensor: (batch, dim): Observations
        m: tensor: (batch, mix, dim): Mixture means
        v: tensor: (batch, mix, dim): Mixture variances

    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    # TODO: Implement this

    # Simply call the log_normal function, but now we can apply log_mean_exp
    # log_mean_exp will take the mean of the log_prob in a numerically stable manner
    # the values that were not logged will be logged in this function, and 
    # those that were already logged will remain
    log_prob = log_mean_exp(log_normal(z, m, v), dim = 1)

    return log_prob


def gaussian_parameters(h, dim=-1):
    """
    Converts generic real-valued representations into mean and variance
    parameters of a Gaussian distribution

    Args:
        h: tensor: (batch, ..., dim, ...): Arbitrary tensor
        dim: int: (): Dimension along which to split the tensor for mean and
            variance

    Returns:
        m: tensor: (batch, ..., dim / 2, ...): Mean
        v: tensor: (batch, ..., dim / 2, ...): Variance
    """
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v


def log_bernoulli_with_logits(x, logits):
    """
    Computes the log probability of a Bernoulli given its logits

    Args:
        x: tensor: (batch, dim): Observation
        logits: tensor: (batch, dim): Bernoulli logits

    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    log_prob = -bce(input=logits, target=x).sum(-1)
    return log_prob

def loss_sigmoid(x, logits):
    """
    Computes the log probability of a Bernoulli given its logits

    Args:
        x: tensor: (batch, dim): Observation
        logits: tensor: (batch, dim): Bernoulli logits

    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    prob = -mse(input=logits.float(), target=x.float()).sum(-1)
    return prob


def kl_cat(q, log_q, log_p):
    """
    Computes the KL divergence between two categorical distributions

    Args:
        q: tensor: (batch, dim): Categorical distribution parameters
        log_q: tensor: (batch, dim): Log of q
        log_p: tensor: (batch, dim): Log of p

    Return:
        kl: tensor: (batch,) kl between each sample
    """
    element_wise = (q * (log_q - log_p))
    kl = element_wise.sum(-1)
    return kl


def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension

    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance

    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl


def duplicate(x, rep):
    """
    Duplicates x along dim=0

    Args:
        x: tensor: (batch, ...): Arbitrary tensor
        rep: int: (): Number of replicates. Setting rep=1 returns orignal x

    Returns:
        _: tensor: (batch * rep, ...): Arbitrary replicated tensor
    """
    return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])


def log_mean_exp(x, dim):
    """
    Compute the log(mean(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which mean is computed

    Return:
        _: tensor: (...): log(mean(exp(x), dim))
    """
    return log_sum_exp(x, dim) - np.log(x.size(dim))


def log_sum_exp(x, dim=0):
    """
    Compute the log(sum(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which sum is computed

    Return:
        _: tensor: (...): log(sum(exp(x), dim))
    """
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()


def load_model_by_name(model, global_step):
    """
    Load a model based on its name model.name and the checkpoint iteration step

    Args:
        model: Model: (): A model
        global_step: int: (): Checkpoint iteration
    """
    file_path = os.path.join('checkpoints',
                             model.name,
                             'model-{:05d}.pt'.format(global_step))
    print(file_path)
    state = torch.load(file_path)
    model.load_state_dict(state)
    print("Loaded from {}".format(file_path))


################################################################################
# No need to read/understand code beyond this point. Unless you want to.
# But do you tho ¯\_(ツ)_/¯
################################################################################


def evaluate_lower_bound(model, labeled_test_subset, run_iwae=True, ox = None):
    #check_model = isinstance(model, VAE) or isinstance(model, GMVAE) or isinstance(model, GMVAE)
    #assert check_model, "This function is only intended for VAE and GMVAE"

    print('*' * 80)
    print("LOG-LIKELIHOOD LOWER BOUNDS ON TEST SUBSET")
    print('*' * 80)

    #print(labeled_test_subset)
    if labeled_test_subset is None:
        xl = ox
    else:
        xl, _ = labeled_test_subset
    #print(xl)
    torch.manual_seed(0)
    xl = torch.bernoulli(xl)

    def detach_torch_tuple(args):
        return (v.detach() for v in args)

    def compute_metrics(fn, repeat):
        metrics = [0, 0, 0]
        for _ in range(repeat):
            if labeled_test_subset is None:
                niwae, kl, rec = detach_torch_tuple(fn(xl))
            else:
                niwae, kl, rec = fn(xl)
            metrics[0] += niwae / repeat
            metrics[1] += kl / repeat
            metrics[2] += rec / repeat
        return metrics

    # Run multiple times to get low-var estimate
    nelbo, kl, rec = compute_metrics(model.negative_elbo_bound, 100)
    print("NELBO: {}. KL: {}. Rec: {}".format(nelbo, kl, rec))

    if run_iwae:
        for iw in [1, 10, 100, 1000]:
            repeat = max(100 // iw, 1) # Do at least 100 iterations
            fn = lambda x: model.negative_iwae_bound(x, iw)
            niwae, kl, rec = compute_metrics(fn, repeat)
            print("Negative IWAE-{}: {}".format(iw, niwae))


def evaluate_classifier(model, test_set):
    check_model = isinstance(model, SSVAE)
    assert check_model, "This function is only intended for SSVAE"

    print('*' * 80)
    print("CLASSIFICATION EVALUATION ON ENTIRE TEST SET")
    print('*' * 80)

    X, y = test_set
    pred = model.cls.classify(X)
    accuracy = (pred.argmax(1) == y).float().mean()
    print("Test set classification accuracy: {}".format(accuracy))


def save_model_by_name(model, global_step):
    save_dir = os.path.join('checkpoints', model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))


def prepare_writer(model_name, overwrite_existing=False):
    log_dir = os.path.join('logs', model_name)
    save_dir = os.path.join('checkpoints', model_name)
    if overwrite_existing:
        delete_existing(log_dir)
        delete_existing(save_dir)
    # Sadly, I've been told *not* to use tensorflow :<
    # writer = tf.summary.FileWriter(log_dir)
    writer = None
    return writer


def log_summaries(writer, summaries, global_step):
    pass # Sad :<
    # for tag in summaries:
    #     val = summaries[tag]
    #     tf_summary = tf.Summary.Value(tag=tag, simple_value=val)
    #     writer.add_summary(tf.Summary(value=[tf_summary]), global_step)
    # writer.flush()


def delete_existing(path):
    if os.path.exists(path):
        print("Deleting existing path: {}".format(path))
        shutil.rmtree(path)


def reset_weights(m):
    try:
        m.reset_parameters()
    except AttributeError:
        pass

def get_mnist_data_and_test(device, use_test_subset=True):
    preprocess = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True)

    # Create pre-processed training and test sets
    X_train = train_loader.dataset.train_data.to(device).reshape(-1, 784).float() / 255
    y_train = train_loader.dataset.train_labels.to(device)
    X_test = test_loader.dataset.test_data.to(device).reshape(-1, 784).float() / 255
    y_test = test_loader.dataset.test_labels.to(device)

    # Create supervised subset (deterministically chosen)
    # This subset will serve dual purpose of log-likelihood evaluation and
    # semi-supervised learning. Pretty hacky. Don't judge :<
    X = X_test if use_test_subset else X_train
    y = y_test if use_test_subset else y_train

    xl, yl = [], []
    for i in range(10):
        idx = y == i
        idx_choice = get_mnist_index(i, test=use_test_subset)
        xl += [X[idx][idx_choice]]
        yl += [y[idx][idx_choice]]
    xl = torch.cat(xl).to(device)
    yl = torch.cat(yl).to(device)
    yl = yl.new(np.eye(10)[yl])
    labeled_subset = (xl, yl)

    return train_loader, labeled_subset, test_loader

def get_mnist_data(device, use_test_subset=True):
    preprocess = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True)

    # Create pre-processed training and test sets
    X_train = train_loader.dataset.train_data.to(device).reshape(-1, 784).float() / 255
    y_train = train_loader.dataset.train_labels.to(device)
    X_test = test_loader.dataset.test_data.to(device).reshape(-1, 784).float() / 255
    y_test = test_loader.dataset.test_labels.to(device)

    # Create supervised subset (deterministically chosen)
    # This subset will serve dual purpose of log-likelihood evaluation and
    # semi-supervised learning. Pretty hacky. Don't judge :<
    X = X_test if use_test_subset else X_train
    y = y_test if use_test_subset else y_train

    xl, yl = [], []
    for i in range(10):
        idx = y == i
        idx_choice = get_mnist_index(i, test=use_test_subset)
        xl += [X[idx][idx_choice]]
        yl += [y[idx][idx_choice]]
    xl = torch.cat(xl).to(device)
    yl = torch.cat(yl).to(device)
    yl = yl.new(np.eye(10)[yl])
    labeled_subset = (xl, yl)

    return train_loader, labeled_subset, (X_test, y_test)


def get_mnist_index(i, test=True):
    # Obviously *hand*-coded
    train_idx = np.array([[2732,2607,1653,3264,4931,4859,5827,1033,4373,5874],
                          [5924,3468,6458,705,2599,2135,2222,2897,1701,537],
                          [2893,2163,5072,4851,2046,1871,2496,99,2008,755],
                          [797,659,3219,423,3337,2745,4735,544,714,2292],
                          [151,2723,3531,2930,1207,802,2176,2176,1956,3622],
                          [3560,756,4369,4484,1641,3114,4984,4353,4071,4009],
                          [2105,3942,3191,430,4187,2446,2659,1589,2956,2681],
                          [4180,2251,4420,4870,1071,4735,6132,5251,5068,1204],
                          [3918,1167,1684,3299,2767,2957,4469,560,5425,1605],
                          [5795,1472,3678,256,3762,5412,1954,816,2435,1634]])

    test_idx = np.array([[684,559,629,192,835,763,707,359,9,723],
                         [277,599,1094,600,314,705,551,87,174,849],
                         [537,845,72,777,115,976,755,448,850,99],
                         [984,177,755,797,659,147,910,423,288,961],
                         [265,697,639,544,543,714,244,151,675,510],
                         [459,882,183,28,802,128,128,53,550,488],
                         [756,273,335,388,617,42,442,543,888,257],
                         [57,291,779,430,91,398,611,908,633,84],
                         [203,324,774,964,47,639,131,972,868,180],
                         [1000,846,143,660,227,954,791,719,909,373]])

    if test:
        return test_idx[i]

    else:
        return train_idx[i]


def get_svhn_data(device):
    preprocess = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN('data', split='extra', download=True, transform=preprocess),
        batch_size=100,
        shuffle=True)

    return train_loader, (None, None), (None, None)


def gumbel_softmax(logits, tau, eps=1e-8):
    U = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(U + eps) + eps)
    y = logits + gumbel
    y = F.softmax(y / tau, dim=1)
    return y

from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # # Convert covariance to principal axes
    # U, s, Vt = np.linalg.svd(covariance)
    # angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    # width, height = 2 * np.sqrt(s)
    
    # # Draw the Ellipse
    # for nsig in range(1, 4):
    #     e = Ellipse(position, nsig * width, nsig * height, angle, **kwargs)
    #     print(e)
    #     ax.add_patch(e)
        
    v, w = linalg.eigh(covariance)
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    u = w[0] / linalg.norm(w[0])
    # as the DP will not use every component it has access to
    # unless it needs it, we shouldn't plot the redundant
    # components.

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan(u[1] / u[0])
    angle = 180. * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(position, v[0], v[1], 180. + angle, **kwargs) #color
    ax.add_patch(ell)
    # ell.set_clip_box(splot.bbox)
    # ell.set_alpha(0.5)
    # splot.add_artist(ell)




class FixedSeed:
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        np.random.set_state(self.state)
