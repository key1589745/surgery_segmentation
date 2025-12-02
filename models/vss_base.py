import torch
import torch.nn as nn


class base_model(nn.Module):
    def __init__(self, image_encoder, local_memory, global_memory, mask_decoder, loss, model_name="base_model"):
        super(base_model, self).__init__()    
        self.model_name = model_name
        
        self.encoder = image_encoder
        self.local_memory = local_memory
        self.global_memory = global_memory
        self.decoder = mask_decoder
        self.loss = loss

    def forward(self, x, mask=None):
        kld = torch.tensor(0.0, device=x.device)
        if len(x.shape) < 5:
            f_curr = self.encoder(x)
        else:
            x = x.transpose(0,1)
            x_curr, x_memo = x[0], x[1:]
            f_curr = self.encoder(x_curr)
            f_memo = []
            for x in x_memo:
                with torch.no_grad():
                    f = self.encoder(x)
                    f_memo.append(f[-1].detach())
                
            f_curr[-1] = self.local_memory(f_curr[-1], f_memo)
            
            if self.training:
                f_curr[-1], kld = self.global_memory(f_curr[-1], mask)
            else:
                f_curr[-1] = self.global_memory(f_curr[-1], mask)
    
        mask_pred = self.decoder(f_curr)
        if self.training:
            return (mask_pred, kld)
        else:  
            return mask_pred





if __name__ == "__main__":
    with torch.no_grad():
        model = base_model().cuda()
        x = torch.randn(8, 32, 3, 256, 256).cuda()
        out = model(x)
        print(out.shape)