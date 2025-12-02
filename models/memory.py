import torch
from torch import nn
import torch.nn.functional as F
from sam2.modeling.sam2_utils import get_clones, LayerNorm2d
from .model_utils import resize
import math
import torch.distributions as distributions

class MemoryModule(nn.Module):

    def __init__(self, memory_attention, memory_encoder, memory_length):
        super(MemoryModule, self).__init__()
        self.attention = memory_attention
        self.memory_encoder = memory_encoder
        self.memory_bank = None
        self.memory_length = memory_length

    def feature_enhance(self, f_curr, f_memo):
        # f_curr: B x C x H x W; f_memo: B x (T-1) x C x H x W
        return self.attention(f_curr, f_memo)


class PriorEncoder(nn.Module):
    def __init__(self, in_dim, d_latent, num_clusters):
        super(PriorEncoder, self).__init__()
        self.d_latent = d_latent
        self.num_clusters = num_clusters
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_dim, num_clusters * (2 * d_latent + 1))
        
    def forward(self, f_curr):
        B = f_curr.shape[0]
        x = self.gap(f_curr).flatten(1)
        x = self.fc(x)
        x = x.view(B, 2 * self.d_latent + 1, self.num_clusters)
        
        mu = x[:, :self.d_latent, :]
        log_sigma = x[:, self.d_latent:2*self.d_latent, :]
        pi_logits = x[:, 2*self.d_latent, :]
        
        pi = F.softmax(pi_logits, dim=1)
        
        return mu, log_sigma, pi

class PostEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        d_latent,
        mask_downsampler,
        fuser,
        embed_dim=256,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.mask_downsampler = mask_downsampler
        self.pix_feat_proj = nn.Conv2d(in_dim, embed_dim, kernel_size=1)
        self.fuser = fuser
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(embed_dim, 2 * d_latent)

    def forward(
        self,
        f_curr: torch.Tensor,
        y_curr: torch.Tensor,
    ):
        ## Process masks
        masks = self.mask_downsampler(y_curr)

        ## Fuse pix_feats and downsampled masks
        f_curr = f_curr.to(masks.device)

        x = self.pix_feat_proj(f_curr)
        x = x + masks
        x = self.fuser(x)
        
        x = self.gap(x).flatten(1)
        x = self.fc(x)
        
        mu = x[:, :self.d_latent]
        log_sigma = x[:, self.d_latent:]
        
        return mu, log_sigma


class GlobalMemoryModule(nn.Module):
    def __init__(self, prior_encoder, post_encoder):
        super(GlobalMemoryModule, self).__init__()
        self.prior_encoder = prior_encoder
        self.post_encoder = post_encoder
        self.eps = 1e-8
        
        self.in_dim = prior_encoder.fc.in_features
        self.z_dim = prior_encoder.d_latent
        self.proj = nn.Conv2d(self.in_dim + self.z_dim, self.in_dim, kernel_size=1)

    def forward(self, f_curr, y_curr):
        if self.training:
            mu_prior, log_sigma_prior, pi_prior = self.prior_encoder(f_curr) # B x D x C
            mu_post, log_sigma_post = self.post_encoder(f_curr, y_curr) 
            
            pi_post = self._cluster_sampler(mu_post.detach(), log_sigma_post.detach(), \
                mu_prior.detach(), log_sigma_prior.detach()) # B x C

            q_post = distributions.Normal(mu_post, torch.exp(log_sigma_post) + self.eps)
            z_c = q_post.sample()

            f_curr = self._f_combine(f_curr, z_c)
            kld = self._kld(mu_prior, log_sigma_prior, mu_post, log_sigma_post, pi_prior, pi_post)

            return f_curr, kld
        else:
            mu_prior, log_sigma_prior, pi_prior = self.prior_encoder(f_curr) 
            z_c = self._global_context_sampler(mu_prior, log_sigma_prior, pi_prior)
            f_curr = self._f_combine(f_curr, z_c)
            return f_curr

    def _cluster_sampler(self, mu, log_sigma, mu_prior, log_sigma_prior):
        q_post = distributions.Normal(mu, torch.exp(log_sigma))
        z_post = q_post.sample()
        
        # Calculate probability of z_post under each prior component
        # z_post: B x D
        # mu_prior, log_sigma_prior: B x D x C
        
        z_post_expanded = z_post.unsqueeze(2) # B x D x 1
        
        # Calculate log probability for each component
        # log_prob: B x D x C
        dist = distributions.Normal(mu_prior, torch.exp(log_sigma_prior))
        log_probs = dist.log_prob(z_post_expanded)
        
        # Sum over dimensions -> B x C
        log_prob_c = torch.sum(log_probs, dim=1)
        
        # Convert to probabilities 
        pi = F.softmax(log_prob_c, dim=1)

        return pi

    def _global_context_sampler(self, mu, log_sigma, pi):
        # Sample from categorical distribution
        c = torch.argmax(pi, dim=1) # B

        # Gather the mu and log_sigma for the selected components
        # mu, log_sigma: B x D x C
        B, D, C = mu.shape
        
        # Expand c to gather along the last dimension
        c_expanded = c.view(B, 1, 1).expand(B, D, 1)

        mu_c = torch.gather(mu, 2, c_expanded).squeeze(2) # B x D
        log_sigma_c = torch.gather(log_sigma, 2, c_expanded).squeeze(2) # B x D

        # Sample from the selected Normal distribution
        p_c = distributions.Normal(mu_c, torch.exp(log_sigma_c))
        z_c = p_c.sample()
        return z_c

    def _f_combine(self, f_curr, z_c_post):
        # f_curr: B x C x H x W
        # z_c_post: B x D
        B, C, H, W = f_curr.shape
        
        # Expand z_c_post to match H, W
        z_c_expanded = z_c_post.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
        
        # Concatenate along channel dimension
        combined = torch.cat([f_curr, z_c_expanded], dim=1)
        
        # Project back to C
        return self.proj(combined)

    def _kld(self, mu_prior, log_sigma_prior, mu_post, log_sigma_post, pi_prior, pi_post):
        # 1. KL term for Gaussians: sum_c pi_post_c * KL(q(z|x) || p(z|c))
        # q(z|x): mu_post, log_sigma_post (B x D)
        # p(z|c): mu_prior, log_sigma_prior (B x D x C)
        
        B, D, C = mu_prior.shape
        
        mu_post = mu_post.unsqueeze(2) # B x D x 1
        log_sigma_post = log_sigma_post.unsqueeze(2) # B x D x 1
        
        sigma_prior = torch.exp(log_sigma_prior) 
        sigma_post = torch.exp(log_sigma_post) 
        
        # Gaussian KL: log(sig2/sig1) + (sig1^2 + (mu1-mu2)^2)/(2*sig2^2) - 0.5
        # sig1 = post, sig2 = prior
        
        kl_g = (log_sigma_prior - log_sigma_post) + \
               (sigma_post**2 + (mu_post - mu_prior)**2) / \
               (2 * sigma_prior**2 + self.eps) - 0.5
               
        # Sum over D
        kl_g = torch.sum(kl_g, dim=1) # B x C
        
        # Weighted by posterior probabilities
        kl_g_weighted = torch.sum(pi_post * kl_g, dim=1) # B
        
        # 2. KL term for Categorical: KL(pi_post || pi_prior)
        # sum pi_post * log(pi_post / pi_prior)
        
        
        kl_cat = torch.sum(pi_post * (torch.log(pi_post+self.eps) - torch.log(pi_prior+self.eps)), dim=1) # B
        
        # 3. KL term for Standard Normal: KL(q(z|x) || N(0, I))
        kl_std = -log_sigma_post + 0.5 * (torch.exp(2 * log_sigma_post) + mu_post**2) - 0.5
        kl_std = torch.sum(kl_std, dim=1) # B
        
        return torch.mean(kl_g_weighted + kl_cat + kl_std)
        


class LocalMemoryModule(nn.Module):
    def __init__(self, local_filter, memory_attention, in_dim=1152, mem_dim=64):
        super(LocalMemoryModule, self).__init__()
        self.attention = memory_attention
        self.local_filter = local_filter
        self.mem_proj = nn.Conv2d(in_dim, mem_dim, kernel_size=1)

    def forward(self, f_curr, f_memo):
        # f_curr: B x C x H x W; f_memo: B x (T-1) x C x H x W
        if isinstance(f_memo, list):
            f_memo = torch.stack(f_memo, dim=1)
        f_memo = self.local_filter(f_curr, f_memo)
        
        B, T, C, H, W = f_memo.shape
        f_memo = f_memo.reshape(B * T, C, H, W)
        f_memo = self.mem_proj(f_memo)
        f_memo = f_memo.reshape(B, T, -1, H, W)
        
        return self.attention(f_curr, f_memo)
        

class LocalFilter(nn.Module):
    def __init__(self, similarity_metric, num_frames):
        super(LocalFilter, self).__init__()
        self.s_metric = similarity_metric
        self.n_frames = num_frames

    def forward(self, f_curr, f_memo):
        """ 
        Filter the most similar frame in the memory bank based on the similarity metric.
        Keep num_frames most similar frames.
        Args:
            f_curr: B x C x H x W
            f_memo: B x (T-1) x C x H x W
        Returns:
            f_memo: B x num_frames x C x H x W
        """
        B, T, C, H, W = f_memo.shape
        if T <= self.n_frames:
            return f_memo

        # Global Average Pooling
        curr_feat = torch.mean(f_curr, dim=(2, 3)) # B x C
        memo_feat = torch.mean(f_memo, dim=(3, 4)) # B x T x C
        
        if self.s_metric == 'cosine':
            curr_feat = F.normalize(curr_feat, p=2, dim=1)
            memo_feat = F.normalize(memo_feat, p=2, dim=2)
            # B x 1 x C * B x T x C -> B x T
            sim = torch.sum(curr_feat.unsqueeze(1) * memo_feat, dim=2)
        else:
            # Default to negative L2 distance (so max is closest)
            sim = -torch.norm(curr_feat.unsqueeze(1) - memo_feat, p=2, dim=2)

        # Select topk most similar frames
        _, idx = torch.topk(sim, k=self.n_frames, dim=1, largest=False) # B x k
        
        # Gather selected frames
        idx = idx.view(B, self.n_frames, 1, 1, 1).expand(B, self.n_frames, C, H, W)
        f_memo_selected = torch.gather(f_memo, 1, idx)
        
        return f_memo_selected

class MemoryEncoder(nn.Module):
    def __init__(
        self,
        out_dim,
        in_dim,
        mask_downsampler,
        fuser,
        embed_dim=256,
    ):
        super().__init__()

        self.mask_downsampler = mask_downsampler
        self.pix_feat_proj = nn.Conv2d(in_dim, embed_dim, kernel_size=1)
        self.fuser = fuser
        self.out_proj = nn.Identity()
        if out_dim != embed_dim:
            self.out_proj = nn.Conv2d(embed_dim, out_dim, kernel_size=1)

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
    ):
        ## Process masks
        # sigmoid, so that less domain shift from gt masks which are bool
        masks = self.mask_downsampler(masks)

        ## Fuse pix_feats and downsampled masks
        # in case the visual features are on CPU, cast them to CUDA
        pix_feat = pix_feat.to(masks.device)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        return x


class MaskDownSampler(nn.Module):
    """
    Progressively downsample a mask by total_stride, each time by stride.
    Note that LayerNorm is applied per *token*, like in ViT.

    With each downsample (by a factor stride**2), channel capacity increases by the same factor.
    In the end, we linearly project to embed_dim channels.
    """

    def __init__(
        self,
        mask_in_chans=4,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding=0,
        total_stride=8,
        activation=nn.GELU,
    ):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride**num_layers == total_stride
        self.encoder = nn.Sequential()
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride**2)
            self.encoder.append(
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans

        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float()
        x = resize(x, size=(64,64),mode='bilinear',align_corners=False)
        return self.encoder(x)


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        layer: nn.Module,
        num_layers: int,
        d_feat: int
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.linear_in = nn.Linear(d_feat, d_model)
        self.linear_out = nn.Linear(d_model, d_feat)

    def forward(self, curr, memory):
        
        B, C, H, W = curr.shape
        # Reshape features to sequence format
        curr = curr.flatten(2).transpose(1, 2)  # [B, H*W, C]
        # Memory: [B, T, C, H, W] -> [B, T*H*W, C]
        memory = memory.transpose(1,2).flatten(2).transpose(1, 2)
        curr = self.linear_in(curr)

        for layer in self.layers:
            curr = layer(tgt=curr,memory=memory)
        curr = self.norm(curr)
        curr = self.linear_out(curr)

        return curr.transpose(1, 2).view(B, C, H, W)




# class MemoryModule(nn.Module):

#     def __init__(self,matmul_norm=False):
#         super(MemoryModule, self).__init__()
#         self.matmul_norm = matmul_norm

#     def forward(self, memory_keys, memory_values, query_key, query_value):
#         """
#         Memory Module forward.
#         Args:
#             memory_keys (Tensor): memory keys tensor, shape: TxBxCxHxW
#             memory_values (Tensor): memory values tensor, shape: TxBxCxHxW
#             query_key (Tensor): query keys tensor, shape: BxCxHxW
#             query_value (Tensor): query values tensor, shape: BxCxHxW

#         Returns:
#             Concat query and memory tensor.
#         """
#         sequence_num, batch_size, key_channels, height, width = memory_keys.shape
#         _, _, value_channels, _, _ = memory_values.shape
#         assert query_key.shape[1] == key_channels and query_value.shape[1] == value_channels
#         memory_keys = memory_keys.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW
#         memory_keys = memory_keys.view(batch_size, key_channels, sequence_num * height * width)  # BxCxT*H*W

#         query_key = query_key.view(batch_size, key_channels, height * width).permute(0, 2, 1).contiguous()  # BxH*WxCk
#         key_attention = torch.bmm(query_key, memory_keys)  # BxH*WxT*H*W
#         if self.matmul_norm:
#             key_attention = (key_channels ** -.5) * key_attention
#         key_attention = F.softmax(key_attention, dim=-1)  # BxH*WxT*H*W

#         memory_values = memory_values.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW
#         memory_values = memory_values.view(batch_size, value_channels, sequence_num * height * width)
#         memory_values = memory_values.permute(0, 2, 1).contiguous()  # BxT*H*WxC
#         memory = torch.bmm(key_attention, memory_values)  # BxH*WxC
#         memory = memory.permute(0, 2, 1).contiguous()  # BxCxH*W
#         memory = memory.view(batch_size, value_channels, height, width)  # BxCxHxW

#         query_memory = torch.cat([query_value, memory], dim=1)
#         return query_memory