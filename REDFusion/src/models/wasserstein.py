#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 这是最开始的方法，也就是在seed=1的情况下得到0.80732
#

import torch
import torch.nn as nn

from src.models.bert import BertEncoder, early_BertClf
from src.models.image import ImageEncoder, early_ImageClf


class MultimodalEarlyFusionClf(nn.Module):
    def __init__(self, args):
        super(MultimodalEarlyFusionClf, self).__init__()
        self.args = args

        self.txtclf = early_BertClf(args)

        self.imgclf = early_ImageClf(args)

        # self.text_transform = nn.Linear(args.hidden_sz, args.before_clf_size)
        # self.image_transform = nn.Linear(args.img_hidden_sz * args.num_image_embeds, args.before_clf_size)
        self.classification_layer = nn.Linear(args.before_clf_size, args.n_classes)

    def forward(self, txt, mask, segment, img):

        txt_out = self.txtclf(txt, mask, segment)  # 16 * 1024
        img_out = self.imgclf(img)  # 16 * 1024

        txt_logits = self.classification_layer(txt_out)
        img_logits = self.classification_layer(img_out)

        txt_energy = torch.log(torch.sum(torch.exp(txt_logits), dim=1))
        img_energy = torch.log(torch.sum(torch.exp(img_logits), dim=1))

        txt_conf = txt_energy / 10  # [bs]
        img_conf = img_energy / 10  # [bs]

        txt_conf = torch.reshape(txt_conf, (-1, 1))  # bs * 1
        img_conf = torch.reshape(img_conf, (-1, 1))  # bs * 1

        if self.args.df:
            txt_img_out = (txt_out * txt_conf.detach() + img_out * img_conf.detach())  # 16 * 1024

            txt_img_out = self.classification_layer(txt_img_out)  # 16 * 3

        else:
            # txt_conf.detach()
            # img_conf.detach()
            txt_img_out = 0.5 * txt_out + 0.5 * img_out

        return txt_img_out, txt_logits, img_logits, txt_conf, img_conf, txt_out, img_out
        # return txt_img_out, txt_logits, img_logits, txt_out, img_out