# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
log = logging.getLogger(__file__)

import torch
import torch.nn as nn

from argparse import Namespace
from .accumulator import Accumulator


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, ema=0, num_classes=10, predictor_reg=None, sup_branch=False):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        encoder = base_encoder(num_classes=num_classes, zero_init_residual=True)
        self.cls = encoder.fc if sup_branch else None

        # Online encoder and projector
        self.encoder = encoder
        prev_dim = encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()

        layers = [nn.Linear(prev_dim, prev_dim, bias=True),
                  nn.BatchNorm1d(prev_dim),
                  nn.ReLU(inplace=True), # first layer
                  nn.Linear(prev_dim, dim, bias=True)]
        if not predictor_reg: # To have the same projector as BYOL
            layers[-1].bias.requires_grad = False # hack: not use bias as it is followed by BN
            layers.append(nn.BatchNorm1d(dim, affine=False))
        self.projector = nn.Sequential(*layers)

        # Target encoder
        self.ema = ema
        self.target_encoder = copy.deepcopy(self.encoder) if ema > 0 else self.encoder
        self.target_projector = copy.deepcopy(self.projector) if ema > 0 else self.projector

        # Projector
        if predictor_reg is not None:
            # One-layer predictor for analytical solution
            self.predictor = nn.Linear(dim, dim, bias=False)
        else:
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                            nn.BatchNorm1d(pred_dim),
                                            nn.ReLU(inplace=True), # hidden layer
                                            nn.Linear(pred_dim, dim)) # output layer

        self.predictor_reg = predictor_reg
        if predictor_reg is not None:
            self._set_predictor_params()
            self.cum_corr = Accumulator(dyn_lambda=self.predictor_opt.dyn_lambda)

            for m in self.predictor.parameters():
                m.requires_grad = False

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        f1 = self.encoder(x1) # NxC
        f2 = self.encoder(x2) # NxC

        z1 = self.projector(f1)
        z2 = self.projector(f2)

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        if self.ema > 0:
            f1 = self.target_encoder(x1)
            f2 = self.target_encoder(x2)
            z1 = self.target_projector(f1)
            z2 = self.target_projector(f2)

        if self.cls is not None:
            logits = self.cls(f1)
        else:
            logits = None

        return p1, p2, z1.detach(), z2.detach(), logits

    @torch.no_grad()
    def update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        if self.ema > 0:
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = param_k.data * self.ema + param_q.data * (1. - self.ema)

    def _set_predictor_params(self):
        predictor_opt = {
            'predictor_freq': 5,
            'dyn_lambda': 0.3,
            'predictor_signaling_2': False,
            'corr_eigen_decomp': True,
            'balance_type': 'boost_scale',
            'dyn_reg': None,
            'dyn_eps_inside': False,
            'dyn_eps': 0.1,
            'dyn_convert': 2,
            'dyn_noise': None,
            'predictor_reg': self.predictor_reg
        }
        self.predictor_opt = Namespace(**predictor_opt)

        # if self.predictor_opt.dyn_noise is not None:
        #     self.predictor_opt.skew = torch.randn(128, 128).to(device=self.device)
        #     self.predictor_opt.skew = (self.predictor_opt.skew - self.predictor_opt.skew.t()) * self.predictor_opt.dyn_noise


    def update(self, z1, z2):
        if self.predictor_reg is None:
            return

        z1, z2 = z1.detach(), z2.detach()
        corrs = []
        for b in (z1, z2):
            corr = torch.bmm(b.unsqueeze(2), b.unsqueeze(1))
            corrs.append(corr)
            if torch.any(torch.isnan(corr)).item():
                import pdb
                pdb.set_trace()
        self.cum_corr.add_list(corrs)


    def regulate_predictor(self, niter, epoch_start=False):
        if self.predictor_reg is None:
            return

        with torch.no_grad():
            w = self.predictor.weight.clone()
            if torch.any(torch.isnan(w)).item() or torch.any(torch.isinf(w)).item():
                import pdb
                pdb.set_trace()
            # prev_w = w.clone()

            if self.predictor_reg == "corr":
                if self.predictor_opt.predictor_freq > 0 and niter % self.predictor_opt.predictor_freq == 0:
                    M = self.cum_corr.get()
                    if M is not None:
                        if not self.predictor_opt.predictor_signaling_2:
                            log.info(f"Set predictor to align with input correlation. , freq={self.predictor_opt.predictor_freq}, type={self.predictor_opt.balance_type}, pow=1/{self.predictor_opt.dyn_convert}, eps={self.predictor_opt.dyn_eps}, reg={self.predictor_opt.dyn_reg}, noise={self.predictor_opt.dyn_noise}, eps_inside={self.predictor_opt.dyn_eps_inside}")

                        # if self.predictor_opt.dyn_zero_mean:
                        #     mean_f = self.cum_mean1.get()
                        #     M -= torch.ger(mean_f, mean_f)

                        w = self._compute_w_corr(M)

                        # if self.predictor_opt.dyn_noise is not None:
                        #     w += self.predictor_opt.skew / (niter + 1)

            if torch.any(torch.isnan(w)).item() or torch.any(torch.isinf(w)).item():
                import pdb
                pdb.set_trace()

            self.predictor.weight.copy_(w)
            self.predictor_opt.predictor_signaling_2 = True


    def _compute_w_corr(self, M):
        if self.predictor_opt.corr_eigen_decomp:
            if not self.predictor_opt.predictor_signaling_2:
                log.info("compute_w_corr: Use eigen_decomp!")
            D, Q = torch.eig(M, eigenvectors=True)
            # Only use the real part.
            D = D[:,0]
        else:
            # Just use diagonal element.
            if not self.predictor_opt.predictor_signaling_2:
                log.info("compute_w_corr: No eigen_decomp, just use diagonal elements!")
            D = M.diag()
            Q = torch.eye(M.size(0)).to(D.device)

        # if eigen_values >= 1, scale everything down.
        balance_type = self.predictor_opt.balance_type
        reg = self.predictor_opt.dyn_reg

        if balance_type == "shrink2mean":
            mean_eig = D.mean()
            eigen_values = (D - mean_eig) / 2 + mean_eig
        elif balance_type == "clamp":
            eigen_values = D.clamp(min=0, max=1-reg)
        elif balance_type == "boost_scale":
            max_eig = D.max()
            eigen_values = D.clamp(0) / max_eig
            # Going through a concave function (dyn_convert > 1, e.g., 2 or sqrt function) to boost small eigenvalues (while still keep very small one to be 0)
            if self.predictor_opt.dyn_eps_inside:
                # Note that here dyn_eps is allowed to be negative.
                eigen_values = (eigen_values + self.predictor_opt.dyn_eps).clamp(1e-4).pow(1/self.predictor_opt.dyn_convert)
            else:
                # Note that here dyn_eps is allowed to be negative.
                eigen_values = eigen_values.pow(1/self.predictor_opt.dyn_convert) + self.predictor_opt.dyn_eps
                eigen_values = eigen_values.clamp(1e-4)
            if not self.predictor_opt.predictor_signaling_2:
                sorted_values, _ = eigen_values.sort(descending=True)
                log.info(f"Compute eigenvalues with boost_scale: Top-5: {sorted_values[:5]}, Bottom-5: {sorted_values[-5:]}")

        elif balance_type == "scale":
            max_eig = D.max()
            if max_eig > 1 - reg:
                eigen_values = D / (max_eig + reg)
        else:
            raise RuntimeError(f"Unkonwn balance_type: {balance_type}")

        return Q @ eigen_values.diag() @ Q.t()


class SimSiamEncoder(nn.Module):
    """
    SimSiam feature extractor.
    """
    def __init__(self, encoder):
        super(SimSiamEncoder, self).__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)
