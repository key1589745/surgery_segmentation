import torch
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

class BaseMethod:
    def __init__(self, model, dataloaders, train_args, val_args):
        self.model = model.cuda()

        self.dataset_name = dataloaders.dataset_name
        self.train_loader = dataloaders.train_loader
        self.val_loader = dataloaders.val_loader
        self.test_loader = dataloaders.test_loader

        self.optimizer = self._build_callable(train_args.optimizer, params=model.parameters())
        self.scheduler = self._build_callable(train_args.scheduler, optimizer=self.optimizer)
        self.num_epochs = train_args.epochs

        self.evaluator = val_args.evaluator
        self.val_interval = val_args.val_interval
        self.save_dir = val_args.save_dir

 
    def train(self):
        best_val_dice = 0
        best_val_loss = 10000

        for _, epoch in zip(np.linspace(0.01, 0.99, num=self.num_epochs), range(self.num_epochs)):
            # Train
            train_loss = self.train_one_epoch()
            print(f'EPOCH {epoch}/{self.num_epochs}: Loss:', train_loss)
            if epoch % self.val_interval == 0:
                res = self.evaluator(self.model, self.val_loader, validation=True)
                if res['wDice_avg'] > best_val_dice:
                    best_val_dice = res['wDice_avg']

                # print(res)
                print(f'Best val wDice_avg : {best_val_dice:.4f} ')

                self.scheduler.step(res['val_loss'])
            else:
                self.scheduler.step()

        return self.model
    
    def train_one_epoch(self):
        self.model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Handle possible key mismatch
            if 'image' in batch:
                data = batch['image']
            elif 'video' in batch:
                data = batch['video']
            else:
                raise KeyError("Batch missing 'video'/'image' tensor.")
            target = batch['mask']
            
            self.optimizer.zero_grad()
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            outputs = self.model(data, target)
            loss = self.model.loss(outputs,target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        return train_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        res = self.evaluator(self.model, self.test_loader, validation=False)
        print(res)

    def save_model(self):
        dataset_name = self.dataset_name
        model_name = getattr(self.model, 'model_name', 'model')
        torch.save(self.model.state_dict(), self.save_dir + f'{dataset_name}_{model_name}.pth')
    
    @staticmethod
    def _build_callable(component, **kwargs):
        if component is None:
            return None
        if OmegaConf.is_config(component):
            return instantiate(component, **kwargs)
        if callable(component):
            return component(**kwargs) if kwargs else component()
        return component