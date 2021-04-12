import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

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

        # The model priors for a VAE are standard Mean and Variance
        pm = torch.zeros((m.shape))
        pv = torch.ones((v.shape))

        # Compute the KL divergence from the calculated m and v given x to the priors
        kl = torch.mean(ut.kl_normal(m, v, pm, pv))

        # Calculate the reconstruction loss of p(x|z) = log(Bern(x|decoder(z)))
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

        # Get KL and Rec of elbo again
        pm = torch.zeros((m.shape))
        pv = torch.ones((v.shape))
        kl = ut.kl_normal(m, v, pm, pv)
        rec = ut.log_bernoulli_with_logits(x,logits)

        # Now get the log mean of the exp of the KL divergence and subtact the 
        # reconstuction from all of the weighted samples
        niwae = ut.log_mean_exp(ut.kl_normal(m, v, pm, pv), dim = 0) - torch.mean(ut.log_bernoulli_with_logits(x,logits))

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

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
