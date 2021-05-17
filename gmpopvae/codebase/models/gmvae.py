import numpy as np
import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class GMVAE(nn.Module):
    def __init__(self, nn='v1', encode_dim = None, z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        if nn == 'popv':
            nn = getattr(nns, nn)
            self.enc = nn.Encoder(encode_dim, self.z_dim)
            self.dec = nn.Decoder(encode_dim, self.z_dim)
        else:
            nn = getattr(nns, nn)
            self.enc = nn.Encoder(self.z_dim)
            self.dec = nn.Decoder(self.z_dim)

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
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
        m, v = self.enc.encode(x)
        z = ut.sample_gaussian(m, v)
        logits = self.dec.decode(z)

        # TODO: compute the values below

        # The function log_normal takes m and v as input with dimension: (batch, k, dims)
        # Extend the m and v vectors to reflect this shape
        m_x = m.view(m.shape[0], 1, m.shape[1])
        v_x = v.view(v.shape[0], 1, v.shape[1])

        # Get the log of Q of z given x
        term1 = ut.log_normal(z, m_x, v_x).squeeze()

        # Now get the model priors - shape: (1, k, dims)
        m_i, v_i = ut.gaussian_parameters(self.z_pre, dim=1)
        
        # Duplicate the shape to have the correct number of batches - shape: (batch, k, dims)
        m_K = ut.duplicate(m_i, m.shape[0])
        v_K = ut.duplicate(v_i, v.shape[0])

        # Now calculate the log of P of z, which is a log mean of the mixtures
        term2 = ut.log_normal_mixture(z, m_K, v_K).squeeze()

        # Appoximation of KL divergence
        kl = torch.mean(term1 - term2)

        # Reconstruction Loss
        rec = torch.mean(ut.log_bernoulli_with_logits(x,logits))

        # Negative ELBO definition
        nelbo = kl - rec

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

        # Duplicate
        m = ut.duplicate(m, iw)
        v = ut.duplicate(v, iw)
        x = ut.duplicate(x, iw)
        z = ut.sample_gaussian(m, v)
        logits = self.dec.decode(z)

        # TODO: compute the values below

        # Since p(z|x) is intractable, I solve for p(x|z) and p(z)
        # And replace p(z|x)*p(x) with p(x|z) and p(z)
        p_x_g_z = ut.log_bernoulli_with_logits(x,logits)

        # define Pi using torch
        pi = torch.acos(torch.zeros(1)) * 2

        #calculate p(z)
        m_i, v_i = ut.gaussian_parameters(self.z_pre, dim=1)
        m_K = ut.duplicate(m_i, m.shape[0])
        v_K = ut.duplicate(v_i, v.shape[0])
        z_i = z.view(z.shape[0], 1, z.shape[1])
        p_z_K = -0.5 * (torch.sum((z_i - m_K) * (1/v_K) * (z_i - m_K), dim = 2) + (torch.log(2 * pi) * m_K.shape[2]) + torch.log(torch.prod(v_K,2)))
        p_z = torch.mean(p_z_K, dim = 1)

        #calculate q(z|x)
        q_z_g_x = -0.5 * (torch.sum((z - m) * (1/v) * (z - m), dim = 1) + (torch.log(2 * pi)* m.shape[1]) + torch.log(torch.prod(v,1)))

        #calculate the unnormalized density ratio and take the log mean exp of it 
        niwae = ut.log_mean_exp((p_z * p_x_g_z) / q_z_g_x, dim = 0)

        #take the mean of the log of the values otherwise for comparison to the lower ELBO bound
        rec = torch.mean(torch.log(p_x_g_z))
        kl = torch.mean(torch.log(p_z) - torch.log(q_z_g_x))

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

    def get_priors(self):
        return ut.gaussian_parameters(self.z_pre, dim=1)
    
    def get_weights(self):
        return self.pi