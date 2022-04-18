"""
Inspired by https://github.com/SHI-Labs/Compact-Transformers.git
"""

import torch.nn as nn
from transformer import TransformerClassifier
from tokenizer import Tokenizer, OneTokenizer


class CCT(nn.Module):
    def __init__(self,
                 frame=224,
                 feature=60,
                 embedding_dim=768,
                 n_input_channels=1,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 *args, **kwargs):
        super(CCT, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   #    padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   #    pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=frame,
                                                           width=feature),
            embedding_dim=embedding_dim,
            seq_pool=True,
            # dropout_rate=0.1,
            # attention_dropout=0.1,
            # stochastic_depth=0.1,
            *args, **kwargs)

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


class OCT(nn.Module):
    def __init__(self,
                 frame=224,
                 embedding_dim=768,
                 n_input_channels=1,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 *args, **kwargs):
        super().__init__()

        self.tokenizer = OneTokenizer(n_input_channels=n_input_channels,
                                      n_output_channels=embedding_dim,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      pooling_kernel_size=pooling_kernel_size,
                                      pooling_stride=pooling_stride,
                                      pooling_padding=pooling_padding,
                                      max_pool=True,
                                      activation=nn.ReLU,
                                      n_conv_layers=n_conv_layers,
                                      conv_bias=False)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           frames=frame),
            embedding_dim=embedding_dim,
            seq_pool=True,
            # dropout_rate=0.1,
            # attention_dropout=0.1,
            # stochastic_depth=0.1,
            *args, **kwargs)

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)