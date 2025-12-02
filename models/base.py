import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class VUNet(nn.Module):
    def __init__(self, n_channels, n_classes,dims=64):
        super(VUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dims = dims

        self.inc = (DoubleConv(n_channels, dims))
        self.down1 = (Down(dims, dims*2))
        self.down2 = (Down(dims*2, dims*4))
        self.down3 = (Down(dims*4, dims*8))
        self.down4 = (Down(dims*8, dims*16))
        self.up1 = (Up(dims*16, dims*8))
        self.up2 = (Up(dims*8, dims*4))
        self.up3 = (Up(dims*4, dims*2))
        self.up4 = (Up(dims*2, dims))
        self.outc = (OutConv(dims, n_classes))

    def forward(self, x):
        B, T, C, H, W = x.shape
        x1 = self.inc(x.contiguous().view(-1,C,H,W))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5.view(B,T,self.dims*16,H//16,W//16).mean(1), 
                     x4.view(B,T,self.dims*8,H//8,W//8).mean(1))
        x = self.up2(x, x3.view(B,T,self.dims*4,H//4,W//4).mean(1))
        x = self.up3(x, x2.view(B,T,self.dims*2,H//2,W//2).mean(1))
        x = self.up4(x, x1.view(B,T,self.dims,H,W).mean(1))
        logits = self.outc(x)
        return logits



if __name__ == '__main__':

    model = VUNet(n_channels=3, n_classes=3)
    x = torch.rand(4,8,3,256,256)
    y = model(x)
    print(y.shape)














# class Segmentation_base(nn.Module):
#     """Encoder Decoder segmentors.

#     EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
#     Note that auxiliary_head is only used for deep supervision during training,
#     which could be dumped during inference.
#     """

#     def __init__(self, backbone, decoder, loss='CE', pretrained=None):
#         self.backbone = build_backbone(backbone)
#         self.decoder = build_decoder(decoder)
#         # self.loss = get_loss(loss)


#     def forward(self, sequence_imgs):
#         """Encode images with backbone and decode into a semantic segmentation
#         map of the same size as input."""
#         B, T, C, H, W = sequence_imgs.shape
#         query = self.backbone(copy.deepcopy(sequence_imgs[:,0]))
#         memory = [self.backbone(img) for img in sequence_imgs.permute(1,0,2,3,4)]

#         out = self.decoder(query, memory)

#         return out


    

#     # TODO refactor
#     def slide_inference(self, img, sequence_imgs, img_meta, rescale):
#         """Inference by sliding-window with overlap.

#         If h_crop > h_img or w_crop > w_img, the small patch will be used to
#         decode without padding.
#         """

#         h_stride, w_stride = self.test_cfg.stride
#         h_crop, w_crop = self.test_cfg.crop_size
#         batch_size, _, h_img, w_img = img.size()
#         assert sequence_imgs.shape[3:] == (h_img, w_img)
#         num_classes = self.num_classes
#         h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
#         w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
#         preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
#         count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
#         for h_idx in range(h_grids):
#             for w_idx in range(w_grids):
#                 y1 = h_idx * h_stride
#                 x1 = w_idx * w_stride
#                 y2 = min(y1 + h_crop, h_img)
#                 x2 = min(x1 + w_crop, w_img)
#                 y1 = max(y2 - h_crop, 0)
#                 x1 = max(x2 - w_crop, 0)
#                 crop_img = img[:, :, y1:y2, x1:x2]
#                 crop_sequence_imgs = sequence_imgs[:, :, :, y1:y2, x1:x2]  # TxBxCxHxW
#                 crop_seg_logit = self.encode_decode(crop_img, crop_sequence_imgs, img_meta)
#                 preds += F.pad(crop_seg_logit,
#                                (int(x1), int(preds.shape[3] - x2), int(y1),
#                                 int(preds.shape[2] - y2)))

#                 count_mat[:, :, y1:y2, x1:x2] += 1
#         assert (count_mat == 0).sum() == 0
#         if torch.onnx.is_in_onnx_export():
#             # cast count_mat to constant while exporting to ONNX
#             count_mat = torch.from_numpy(
#                 count_mat.cpu().detach().numpy()).to(device=img.device)
#         preds = preds / count_mat
#         if rescale:
#             preds = resize(
#                 preds,
#                 size=img_meta[0]['ori_shape'][:2],
#                 mode='bilinear',
#                 align_corners=self.align_corners,
#                 warning=False)
#         return preds





