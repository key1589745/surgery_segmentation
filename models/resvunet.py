import torch
from torch import nn
from networks.conv_modules import Decoder, Backbone
import torch.nn.functional as F

class Res_VUNet(nn.Module):

    def __init__(self, num_cls, hidden_dim = 256, skip=True, backbone='resnet34'):
        super().__init__()
        
        self.num_cls = num_cls
        
        self.decoder = Decoder(hidden_dim, skip)    
        
        self.backbone = Backbone(backbone, skip)
        
        self.seg_head = nn.Conv2d(hidden_dim // 8, num_cls, kernel_size=3,padding=1)


    def forward(self, x):
        
        B, T, C, H, W = x.shape
        x = self.inc(x.contiguous().view(-1,C,H,W))
        
        x = list(self.backbone(x).values())[::-1]     

        x = self.decoder(x[0], x[1:])
        
        x = self.seg_head(x) 
        
        return x
    
    def feat_extract(self,x):
        
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        
        x = list(self.backbone(x).values())[::-1]     

        x = self.decoder(x[0], x[1:])
        
        return x 




class BackboneBase(nn.Module):

    def __init__(self, backbone, return_interm_layers):
        super(BackboneBase,self).__init__()
        # for name, parameter in backbone.named_parameters():
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"relu": "0","layer1": "1","layer2": "2","layer3": "3"}
        else:
            return_layers = {"layer3": "3"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, x0):
        xs = self.body(x0)
        out= {}
        for name, x in xs.items():
            out[name] = x
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name,return_interm_layers,dilation=False):
        backbone = getattr(torchvision.models, name)(replace_stride_with_dilation=[False, False, dilation],pretrained=True)
        self.num_channels = 512//2 if name in ('resnet18', 'resnet34') else 2048
        super(Backbone,self).__init__(backbone, return_interm_layers)



#####################################################                ####################################


class Conv2d(nn.Conv2d):
    def __init__(self, weight_prior, bias_prior=None, stride=1,
            padding=0, dilation=1, groups=1, padding_mode='zeros'):
        nn.Module.__init__(self)
        
        self.stride = nn.modules.utils._pair(stride)
        self.padding = nn.modules.utils._pair(padding)
        self.dilation = nn.modules.utils._pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.transposed = False
        self.output_padding = nn.modules.utils._pair(0)
        
        (self.out_channels, in_channels, ksize_0, ksize_1) = weight_prior.p.shape
        self.in_channels = in_channels * self.groups
        self.kernel_size = (ksize_0, ksize_1)
        self.weight_prior = weight_prior
        self.bias_prior = bias_prior

    @property
    def weight(self):
        return self.weight_prior()

    @property
    def bias(self):
        return (None if self.bias_prior is None else self.bias_prior())


def Conv2dPrior(in_channels, out_channels, kernel_size=3, stride=1,
            padding=1, dilation=1, groups=1, padding_mode='zeros',
            prior_w=ConvCorrelatedNormal, loc_w=0., std_w=1., prior_b=Normal,
            loc_b=0., std_b=1., scaling_fn=None, weight_prior_params={}, bias_prior_params={}):
    if scaling_fn is None:
        def scaling_fn(std, dim):
            return std/dim**0.5

    in_dim = in_channels * kernel_size**2
    loc_w *= torch.ones(kernel_size**2)
    kernel_size = nn.modules.utils._pair(kernel_size)
    bias_prior = prior_b((out_channels,), 0., std_b, **bias_prior_params) if prior_b is not None else None
    return Conv2d(weight_prior=prior_w((out_channels, in_channels//groups, kernel_size[0], kernel_size[1]),
                                       loc_w, scaling_fn(std_w, in_channels),  # TODO: use `in_dim` here to prevent the variance from blowing up
                                       **weight_prior_params),
                  bias_prior=bias_prior,
                 stride=stride, padding=padding, dilation=dilation,
                  groups=groups, padding_mode=padding_mode)

                                                                ####################################
##################################################### **********  segmentor  ********  ##########################################
                                                                ####################################




class UpsampleBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
    ):
        super().__init__()

        self.up_block = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels+skip_channels, out_channels, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm2d(out_channels)),
            ('relu2', nn.ReLU(inplace=True)),
            ('up_conv', nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)),
            ('bn3', nn.BatchNorm2d(out_channels)),
            ('relu3', nn.ReLU(inplace=True))]))

    def forward(self, x, skip=None):
        
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.up_block(x)
        return x    
    
    

class Decoder(nn.Module):

    def __init__(self, hidden_dim, skip):
        super(Decoder,self).__init__()

        in_channels = [hidden_dim, hidden_dim//2, hidden_dim//4, hidden_dim//4]
        out_channels = [hidden_dim//2, hidden_dim//4, hidden_dim//4, hidden_dim//8]
        if skip:
            skip_channels = [0, hidden_dim//2, hidden_dim//4, hidden_dim//4]
        else:
            skip_channels = [0,0,0,0]
        
        self.up_blocks = nn.ModuleList([])
        for in_channel, out_channel, skip_channel in zip(in_channels,out_channels,skip_channels):
            
            self.up_blocks.append(UpsampleBlock(in_channel, out_channel, skip_channel))


    def forward(self, x, fs):
        x = self.up_blocks[0](x) 
        
        for block, f in itertools.zip_longest(self.up_blocks[1:], fs):
            #print(x.shape,f.shape)
            x = block(x,f)
        
        return x


