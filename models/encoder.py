from sam2.build_sam import build_sam2
import torch.nn as nn
import torch

class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net

class SAM2_encoder(nn.Module):
    def __init__(self, checkpoint_path, backbone):
        super(SAM2_encoder, self).__init__()    
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"]
        checkpoint_encoder = {k[20:]: v for k, v in checkpoint.items() if k.startswith("image_encoder.trunk")}
        backbone.load_state_dict(checkpoint_encoder, strict=False)
        self.encoder = backbone
        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )


    def forward(self, x):
        return self.encoder(x)