import torch
import torch.nn.functional as F
from torch import nn
from mmcv.cnn import ConvModule
from .memory import MemoryModule
from .model_utils import resize

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes, in_channels, embedding_dim, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()

        self.linear_c4 = MLP(input_dim=in_channels[3], embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=in_channels[2], embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=in_channels[1], embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=in_channels[0], embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)



    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = resize(x, size=(256,256),mode='bilinear',align_corners=False)

        return x



class SequenceConv(nn.ModuleList):

    def __init__(self, in_channels, out_channels, kernel_size, sequence_num, conv_cfg, norm_cfg, act_cfg):
        super(SequenceConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sequence_num = sequence_num
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for _ in range(sequence_num):
            self.append(
                ConvModule(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    padding=self.kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
            )

    def forward(self, sequence_imgs):
        """

        Args:
            sequence_imgs (Tensor): TxBxCxHxW

        Returns:
            sequence conv output: TxBxCxHxW
        """
        sequence_outs = []
        assert sequence_imgs.shape[0] == self.sequence_num
        for i, sequence_conv in enumerate(self):
            sequence_out = sequence_conv(sequence_imgs[i, ...])
            sequence_out = sequence_out.unsqueeze(0)
            sequence_outs.append(sequence_out)

        sequence_outs = torch.cat(sequence_outs, dim=0)  # TxBxCxHxW
        return sequence_outs



class TMAHead(nn.Module):
    """TMAHead decoder for video semantic segmentation."""

    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 sequence_num, 
                 key_channels, 
                 value_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 low_in_index=-1,
                 align_corners = False
                 ):
        super(TMAHead, self).__init__()
        self.sequence_num = sequence_num
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.low_in_index = low_in_index
        self.align_corners = align_corners
        self.input_transform = None

        self.memory_key_conv = nn.Sequential(
            SequenceConv(self.in_channels, key_channels, 1, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(key_channels, key_channels, 3, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)
                         )
        self.memory_value_conv = nn.Sequential(
            SequenceConv(self.in_channels, value_channels, 1, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(value_channels, value_channels, 3, sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)
        )
        self.query_key_conv = nn.Sequential(
            ConvModule(
                self.in_channels,
                key_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                key_channels,
                key_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )

        self.query_value_conv = nn.Sequential(
            ConvModule(
                self.in_channels,
                value_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                value_channels,
                value_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )
        self.memory_module = MemoryModule(matmul_norm=False)
        self.bottleneck = ConvModule(
            value_channels * 2,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        self.conv_seg = nn.Conv2d(self.channels, num_classes, kernel_size=1)

    def forward(self, inputs, sequence_imgs):
        """
        Forward fuction.
        Args:
            inputs (list[Tensor]): backbone multi-level outputs.
            sequence_imgs (list[Tensor]): len(sequence_imgs) is equal to batch_size,
                each element is a Tensor with shape of TxCxHxW.

        Returns:
            decoder logits.
        """

        x = self._transform_inputs(inputs)
        sequence_imgs = [self._transform_inputs(inputs).unsqueeze(0) for inputs in sequence_imgs]  # TxBxCxHxW
        sequence_imgs = torch.cat(sequence_imgs,0)
        memory_keys = self.memory_key_conv(sequence_imgs)
        memory_values = self.memory_value_conv(sequence_imgs)
        query_key = self.query_key_conv(x)  # BxCxHxW
        query_value = self.query_value_conv(x)  # BxCxHxW

        # memory read
        output = self.memory_module(memory_keys, memory_values, query_key, query_value)
        output = self.bottleneck(output)
        output = self.conv_seg(output)

        return output
    
    def _transform_inputs(self, inputs, select_low=False):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if select_low:
            if self.input_transform == 'resize_concat':
                inputs = [inputs[i] for i in self.in_index]
                upsampled_inputs = [
                    resize(
                        input=x,
                        size=inputs[0].shape[2:],
                        mode='bilinear',
                        align_corners=self.align_corners) for x in inputs
                ]
                inputs = torch.cat(upsampled_inputs, dim=1)
            elif self.input_transform == 'multiple_select':
                inputs = [inputs[i] for i in self.in_index]
            else:
                inputs = inputs[self.low_in_index]
        else:
            if self.input_transform == 'resize_concat':
                inputs = [inputs[i] for i in self.in_index]
                upsampled_inputs = [
                    resize(
                        input=x,
                        size=inputs[0].shape[2:],
                        mode='bilinear',
                        align_corners=self.align_corners) for x in inputs
                ]
                inputs = torch.cat(upsampled_inputs, dim=1)
            elif self.input_transform == 'multiple_select':
                inputs = [inputs[i] for i in self.in_index]
            else:
                inputs = inputs[self.in_index]

        return inputs
    




if __name__ == '__main__':

    norm_cfg = dict(type='SyncBN', requires_grad=True)

    model = TMAHead(in_channels=64,
        in_index=-1,
        channels=32,
        sequence_num=8,
        key_channels=16,
        value_channels=16,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False)
    x = [torch.rand([4, 64, 256, 256])]
    seq = [[torch.rand([4, 64, 256, 256])]]*8
    y = model(x,seq)
    print(y.shape)