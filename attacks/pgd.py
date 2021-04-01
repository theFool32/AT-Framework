#!/usr/bin/env python3

from .base import Attack


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

    def perturb(self, inputs):
        return self.adversary.perturb(inputs)
