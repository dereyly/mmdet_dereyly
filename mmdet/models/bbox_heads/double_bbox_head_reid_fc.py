import torch.nn as nn
from mmcv.cnn.weight_init import normal_init, xavier_init

from mmdet.ops import ConvModule
from ..backbones.resnet import Bottleneck
from ..registry import HEADS
from .bbox_head import BBoxHead


class BasicResBlock(nn.Module):
    """Basic residual block.

    This block is a little different from the block in the ResNet backbone.
    The kernel size of conv1 is 1 in this block while 3 in ResNet BasicBlock.

    Args:
        in_channels (int): Channels of the input feature map.
        out_channels (int): Channels of the output feature map.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(BasicResBlock, self).__init__()

        # main path
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.conv2 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        # identity path
        self.conv_identity = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        identity = self.conv_identity(identity)
        out = x + identity

        out = self.relu(out)
        return out


@HEADS.register_module
class DoubleConvFCBBoxHeadReIdFC(BBoxHead):
    r"""Bbox head used in Double-Head R-CNN

                                      /-> cls
                  /-> shared convs ->
                                      \-> reg
    roi features
                                      /-> cls
                  \-> shared fc    ->
                                      \-> reg
    """  # noqa: W605

    def __init__(self,
                 num_convs=0,
                 num_fcs=0,
                 conv_out_channels=1024,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 reid_dim=32,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        kwargs.setdefault('with_avg_pool', True)
        super(DoubleConvFCBBoxHeadReIdFC, self).__init__(**kwargs)
        assert self.with_avg_pool
        assert num_convs > 0
        assert num_fcs > 0
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.fc_out_channels_cls = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reid_dim=reid_dim
        # increase the channel of input features
        # self.res_block = BasicResBlock(self.in_channels,
        #                                self.conv_out_channels)

        # add conv heads
        self.shared_fcs = self._add_fc_branch(self.fc_out_channels_cls)
        # add fc heads
        self.fc_branch = self._add_fc_branch(self.fc_out_channels)

        out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
        self.fc_reg = nn.Linear(self.fc_out_channels_cls, out_dim_reg)
        self.fc_cls = nn.Linear(self.fc_out_channels_cls, self.num_classes)
        self.density = nn.Linear(self.fc_out_channels_cls, 1)
        self.density_bin = nn.Linear(self.fc_out_channels_cls, 2)
        self.reid = nn.Linear(self.fc_out_channels, self.reid_dim)
        self.pre_fc_cls2 = nn.Linear(self.reid_dim, 256)
        self.fc_cls2=nn.Linear(256, self.num_classes)
        self.relu = nn.ReLU(inplace=True)



    def _add_fc_branch(self,fc_out_channels):
        """Add the fc branch which consists of a sequential of fc layers"""
        branch_fcs = nn.ModuleList()
        for i in range(self.num_fcs):
            fc_in_channels = (
                self.in_channels *
                self.roi_feat_area if i == 0 else fc_out_channels)
            branch_fcs.append(nn.Linear(fc_in_channels, fc_out_channels))
        return branch_fcs

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        normal_init(self.fc_cls, std=0.01)
        normal_init(self.fc_reg, std=0.001)

        for m in self.fc_branch.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')

    def forward(self, x_cls, x_reg):
        # conv head
        # x_conv = self.res_block(x_reg)
        # fc head
        x_fc1 = x_reg.view(x_reg.size(0), -1)
        for fc in self.shared_fcs:
            x_fc1 = self.relu(fc(x_fc1))

        bbox_pred = self.fc_reg(x_fc1)
        cls_score = self.fc_cls(x_fc1)
        dens = self.density(x_fc1)
        dens_bin=self.density_bin(x_fc1)
        # fc head
        x_fc = x_cls.view(x_cls.size(0), -1)
        for fc in self.fc_branch:
            x_fc = self.relu(fc(x_fc))

        embed = self.reid(x_fc)
        x_pre=self.pre_fc_cls2(embed)
        x_pre=self.relu(x_pre)
        cls_v2=self.fc_cls2(x_pre)

        return cls_score, bbox_pred, embed, dens, dens_bin, cls_v2
