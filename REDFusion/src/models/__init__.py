#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from src.models.wasserstein import MultimodalEarlyFusionClf
MODELS = {

    'early_wd': MultimodalEarlyFusionClf  
}


def get_model(args):
    print('the model chosed is : {}'.format(args.model))
    return MODELS[args.model](args)
