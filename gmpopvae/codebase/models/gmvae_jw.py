import numpy as np
import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        self.km = torch.nn.Parameter(torch.randn(1,self.k,z_dim) / np.sqrt(self.k * self.z_dim))
        self.kv = 1 + torch.nn.Parameter(torch.randn(1,self.k,z_dim) / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        print(x)
        m, v = self.enc.encode(x)
        z = ut.sample_gaussian(m, v)
        logits = self.dec.decode(z)

        batch_size, dim = m.shape
        # Compute KL term
        # km = torch.zeros(batch_size,self.k,dim)
        # kv = torch.ones(batch_size,self.k,dim)
        km = self.km.repeat(batch_size,1,1)
        kv = self.kv.repeat(batch_size,1,1)
        kl_vec = ut.log_normal(z,m,v) - ut.log_normal_mixture(z,km,kv)
        kl = torch.mean(kl_vec)

        # Compute reconstruction loss
        rec_vec = torch.neg(ut.log_bernoulli_with_logits(x,logits))
        rec = torch.mean(rec_vec)

        # Compute nelbo
        nelbo = rec + kl

        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """


        m, v = self.enc.encode(x)
        batch_size, dim = m.shape


        # Duplicate
        m = ut.duplicate(m, iw)
        v = ut.duplicate(v, iw)
        x = ut.duplicate(x, iw)
        z = ut.sample_gaussian(m, v)
        logits = self.dec.decode(z)

        km = self.km.repeat(batch_size,1,1)
        kv = self.kv.repeat(batch_size,1,1)
        km = ut.duplicate(km, iw)
        kv = ut.duplicate(kv, iw)
        kl_vec = ut.log_normal(z,m,v) - ut.log_normal_mixture(z,km,kv)
        kl = torch.mean(kl_vec)

        # TODO: compute the values below
        rec_vec = ut.log_bernoulli_with_logits(x,logits)
        rec = torch.neg(torch.mean(rec_vec))

        if iw > 1:
            iwtensor = torch.zeros(iw)
            j = 0
            while j < iw:
                i = 0
                sum = 0
                while i < batch_size:
                    sum += rec_vec[j*batch_size + i]
                    i+=1
                iwtensor[j] = sum/batch_size -kl
                j+=1
            niwae = torch.neg(ut.log_mean_exp(iwtensor,0))

        else:
            niwae = rec+kl

        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def compute_z(self, x):
        m, v = self.enc.encode(x)
        z = ut.sample_gaussian(m, v)
        return z, m, v

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
