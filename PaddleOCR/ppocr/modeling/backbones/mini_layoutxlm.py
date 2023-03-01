# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from paddle import nn

from paddlenlp.transformers import LayoutXLMModel, LayoutXLMForTokenClassification, LayoutXLMForRelationExtraction
from paddlenlp.transformers import LayoutLMModel, LayoutLMForTokenClassification
from paddlenlp.transformers import LayoutLMv2Model, LayoutLMv2ForTokenClassification, LayoutLMv2ForRelationExtraction
from paddlenlp.transformers import AutoModel

__all__ = ["MiniLayoutXLMForSer"]

config = {
            "attention_probs_dropout_prob": 0.1,
            "bos_token_id": 0,
            "coordinate_size": 128,
            "eos_token_id": 2,
            "fast_qkv": False,
            "gradient_checkpointing": False,
            "has_relative_attention_bias": False,
            "has_spatial_attention_bias": False,
            "has_visual_segment_embedding": True,
            "use_visual_backbone": False,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "image_feature_pool_shape": [7, 7, 256],
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-05,
            "max_2d_position_embeddings": 1024,
            "max_position_embeddings": 514,
            "max_rel_2d_pos": 256,
            "max_rel_pos": 128,
            "model_type": "layoutlmv2",
            "num_attention_heads": 12,
            "output_past": True,
            "pad_token_id": 1,
            "shape_size": 128,
            "rel_2d_pos_bins": 64,
            "rel_pos_bins": 32,
            "type_vocab_size": 1,
            "vocab_size": 250002,
        }

class NLPBaseModel(nn.Layer):
    def __init__(self,
                 base_model_class,
                 model_class,
                 checkpoints=None,
                 num_hidden_layers=2,
                 **kwargs):
        super(NLPBaseModel, self).__init__()
        if checkpoints is not None:  # load the trained model
            self.model = model_class.from_pretrained(checkpoints)
        else:  # load the pretrained-model
            base_model = base_model_class(num_hidden_layers=num_hidden_layers, **config)
            self.model = model_class(
                base_model, num_classes=kwargs["num_classes"], dropout=None)
        self.out_channels = 1
        self.use_visual_backbone = True

class MiniLayoutXLMForSer(NLPBaseModel):
    def __init__(self,
                 num_classes,
                 checkpoints=None,
                 num_hidden_layers=2,
                 **kwargs):
        super(MiniLayoutXLMForSer, self).__init__(
            LayoutXLMModel,
            LayoutXLMForTokenClassification,
            checkpoints,
            num_classes=num_classes,
            num_hidden_layers=num_hidden_layers)
        if hasattr(self.model.layoutxlm, "use_visual_backbone"
                   ) and self.model.layoutxlm.use_visual_backbone is False:
            self.use_visual_backbone = False

    def forward(self, x):
        if self.use_visual_backbone is True:
            image = x[4]
        else:
            image = None
        x = self.model(
            input_ids=x[0],
            bbox=x[1],
            attention_mask=x[2],
            token_type_ids=x[3],
            image=image,
            position_ids=None,
            head_mask=None,
            labels=None)
        if self.training:
            res = {"backbone_out": x[0]}
            res.update(x[1])
            return res
        else:
            return x