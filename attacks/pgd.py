#!/usr/bin/env python3

from .base import Attack

import torch
from torch.nn import functional as F

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
            norm, early_stop=False,
            mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

class PGD(Attack):
    # def __init__(self, max_norm=8, alpha=2, iters=10, type='l_inf'):
    def __init__(self, args, model):
        self.epsilon = args.epsilon / 255
        self.norm = args.norm
        self.alpha = args.pgd_alpha / 255
        self.iters = args.attack_iters

        self.model = model

        # TODO:
        if self.norm == "l_inf":
            from advertorch.attacks import LinfPGDAttack

            self.adversary = LinfPGDAttack(
                model,
                # loss_fn=self.loss_fn,
                eps=self.epsilon,
                nb_iter=self.iters,
                eps_iter=self.alpha,
                rand_init=True,
                clip_min=0,
                clip_max=1,
                targeted=False,
            )

    def pgd_linf(self, ):
        pass



    def perturb(self, inputs, labels=None):
        # delta = attack_pgd(self.model, )
        return self.adversary.perturb(inputs, labels)
