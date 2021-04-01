#!/usr/bin/env python3

from .resnet import *
from .preactresnet import *
from .wideresnet import *

def get_network(args):
    # TODO: ugly
    # device
    # parallel
    return globals()[args.model]()
