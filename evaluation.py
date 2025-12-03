import numpy as np
import torch
import torch.nn.functional as F
import time

class Dice(object):

    def __init__(self, num_cls, smooth=1e-5):
        super(Dice, self).__init__()

        self.smooth = smooth
        self.num_cls = num_cls
        self._dices = [np.array([])]*(self.num_cls-1)
        self.cls_weight = [np.array([])] * (num_cls-1)
        self.clear()

    def clear(self):
        self._dices = [[]]*(self.num_cls-1)
    
    def convert(self,pred, target):
        if len(pred.shape)>len(target.shape):
            pred = torch.argmax(pred, dim=1)
        if isinstance(pred,torch.Tensor):
            pred = pred.numpy()
        if isinstance(target,torch.Tensor):
            target = target.numpy()
        
        return pred, target
    
    def update(self, pred, target):
        
        pred, target = self.convert(pred,target)
        
        axis = tuple(range(1,len(target.shape)))
        for i in range(1,self.num_cls):
            intersection = np.sum((pred==i)*(target==i),axis=axis)
            unionset = np.sum(pred==i,axis=axis)+np.sum(target==i,axis=axis)
            _dice_coeff = 2 * intersection.astype('float') / (unionset.astype('float') + self.smooth)
            if np.sum(target!=0,axis=axis) > 0:
                _cls_weight = np.sum(target==i,axis=axis)/ np.sum(target!=0,axis=axis)
            else:
                _cls_weight = 0
            self.cls_weight[i-1] = np.append(self.cls_weight[i-1], _cls_weight)
            self._dices[i-1] = np.append(self._dices[i-1], _dice_coeff)
        return _dice_coeff


    def avg(self):
        return [np.round(np.mean(d),4).item() for d in self._dices]
            
    def std(self):
        return [np.round(np.std(d),4).item() for d in self._dices]
    
    def mean_avg(self):
        return np.round(np.mean([np.mean(d) for d in self._dices]),4).item()
    
    def mean_std(self):
        return np.round(np.std([np.mean(d) for d in self._dices]),4).item()
    
    def wtd_avg(self):
        cls_weight = np.array(self.cls_weight)
        cls_weight = cls_weight / np.sum(cls_weight,0)
        return np.round(np.sum([np.mean(d*w) for d, w in zip(self._dices,cls_weight)]),4).item()
    
    def wtd_std(self):
        cls_weight = np.array(self.cls_weight)
        cls_weight = cls_weight / np.sum(cls_weight,0)
        return np.round(np.sum([np.std(d*w) for d, w in zip(self._dices,cls_weight)]),4).item()


    
class EvaluationMetrics(object):

    def __init__(self, num_cls=4, smooth=1e-5):
        self.smooth = smooth
        self.num_cls = num_cls
        self.clear()

    def clear(self):
        self._dices = [[] for _ in range(self.num_cls-1)]
        self._jaccards = [[] for _ in range(self.num_cls-1)]
        self._f_scores = [[] for _ in range(self.num_cls-1)]
        self._temporal_ious = [[] for _ in range(self.num_cls-1)]
        self.cls_weight = [[] for _ in range(self.num_cls-1)]
        self.history = {} # {video_id: (frame_idx, mask)}

    def convert(self, pred, target):
        if len(pred.shape)>len(target.shape):
            pred = torch.argmax(pred, dim=1)
        if isinstance(pred,torch.Tensor):
            pred = pred.numpy()
        if isinstance(target,torch.Tensor):
            target = target.numpy()
        return pred, target

    def _get_boundary(self, mask):
        # mask: (H, W) boolean or int (0/1)
        h, w = mask.shape
        # simple 4-connectivity erosion
        eroded = np.zeros((h, w), dtype=bool)
        m = mask.astype(bool)
        eroded[1:-1, 1:-1] = m[1:-1, 1:-1] & m[0:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, 0:-2] & m[1:-1, 2:]
        boundary = m & (~eroded)
        return boundary

    def update(self, pred, target, video_ids=None, frame_indices=None):
        
        pred, target = self.convert(pred,target)
        
        batch_size = pred.shape[0]
        
        for b in range(batch_size):
            p = pred[b]
            t = target[b]
            vid = video_ids[b].item() if video_ids is not None else None
            fidx = frame_indices[b].item() if frame_indices is not None else None
            
            # Identify valid pixels (not 255)
            valid_mask = (t != 255)

            for i in range(1, self.num_cls):
                # Cumulative masks, applied only on valid pixels
                p_mask = (p >= i) & valid_mask
                t_mask = (t >= i) & valid_mask
                
                # Intersection & Union
                inter = np.sum(p_mask & t_mask)
                union_dice = np.sum(p_mask) + np.sum(t_mask)
                union_iou = np.sum(p_mask | t_mask)
                
                # Dice
                dice = 2 * inter / (union_dice + self.smooth)
                if union_dice == 0: dice = 1.0
                self._dices[i-1].append(dice)
                
                # Jaccard (IoU)
                iou = inter / (union_iou + self.smooth)
                if union_iou == 0: iou = 1.0
                self._jaccards[i-1].append(iou)
                
                # F-Score
                p_bound = self._get_boundary(p_mask)
                t_bound = self._get_boundary(t_mask)
                inter_b = np.sum(p_bound & t_bound)
                prec = inter_b / (np.sum(p_bound) + 1e-8)
                rec = inter_b / (np.sum(t_bound) + 1e-8)
                f1 = 2 * prec * rec / (prec + rec + 1e-8)
                if np.sum(t_bound) == 0 and np.sum(p_bound) == 0:
                    f1 = 1.0
                self._f_scores[i-1].append(f1)

                # Temporal Consistency
                tc = np.nan
                if vid is not None and fidx is not None:
                    if vid in self.history:
                        last_fidx, last_p = self.history[vid]
                        if abs(fidx - last_fidx) == 1: # Consecutive frames
                            last_p_mask = (last_p >= i) & valid_mask
                            inter_tc = np.sum(p_mask & last_p_mask)
                            union_tc = np.sum(p_mask | last_p_mask)
                            tc = inter_tc / (union_tc + self.smooth)
                            if union_tc == 0: tc = 1.0
                    
                if not np.isnan(tc):
                    self._temporal_ious[i-1].append(tc)

                # Class weight
                w = np.sum(t_mask)
                self.cls_weight[i-1].append(w)

            if vid is not None:
                self.history[vid] = (fidx, p)

        return self._dices

    
    def wdice_avg(self):  
        # Calculate total pixels for each class across all samples
        cls_weights = [np.sum(x) for x in self.cls_weight]
        total_weight = np.sum(cls_weights)
        
        if total_weight == 0:
            return 0.0

        # Calculate mean Dice for each class
        cls_dices = [np.mean(d) if len(d) > 0 else 0.0 for d in self._dices]
        
        # Weighted average: sum(mean_dice_c * total_pixels_c) / total_all_pixels
        weighted_avg = np.sum([d * w for d, w in zip(cls_dices, cls_weights)]) / total_weight
        
        return np.round(weighted_avg, 4).item()

    def wdice_std(self):
        cw = [np.array(x) for x in self.cls_weight]
        cw_arr = np.array(cw)
        sum_w = np.sum(cw_arr, axis=0)
        norm_w = np.divide(cw_arr, sum_w, out=np.zeros_like(cw_arr, dtype=float), where=sum_w!=0)
        dices = [np.array(d) for d in self._dices]
        return np.round(np.sum([np.std(d * w) for d, w in zip(dices, norm_w)]), 4).item()

    def jaccard_avg(self):
        avgs = [np.mean(d) if len(d)>0 else 0.0 for d in self._jaccards]
        return np.round(np.mean(avgs),4).item()
    
    def jaccard_std(self):
        return np.round(np.std([np.mean(d) for d in self._jaccards if len(d)>0]),4).item() if any(len(d)>0 for d in self._jaccards) else 0.0

    def f_score_avg(self):
        avgs = [np.mean(d) if len(d)>0 else 0.0 for d in self._f_scores]
        return np.round(np.mean(avgs),4).item()

    def f_score_std(self):
        return np.round(np.std([np.mean(d) for d in self._f_scores if len(d)>0]),4).item() if any(len(d)>0 for d in self._f_scores) else 0.0

    def temporal_consistency_avg(self):
        avgs = [np.mean(d) if len(d)>0 else 0.0 for d in self._temporal_ious]
        return np.round(np.mean(avgs),4).item()

    def temporal_consistency_std(self):
        return np.round(np.std([np.mean(d) for d in self._temporal_ious if len(d)>0]),4).item() if any(len(d)>0 for d in self._temporal_ious) else 0.0


class Evaluator:
    def __init__(self, num_cls=4, smooth=1e-5):
        self.num_cls = num_cls
        self.smooth = smooth

    def __call__(self, model, dataloader, validation=True):
        metrics = EvaluationMetrics(num_cls=self.num_cls, smooth=self.smooth)
        model.eval()
        val_loss = 0
        
        start_time = time.time()
        total_frames = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle possible key mismatch
                if 'image' in batch:
                    data = batch['image']
                elif 'video' in batch:
                    data = batch['video']
                
                data = data.cuda()
                target = batch['mask']
                video_ids = batch.get('video_id', None)
                frame_indices = batch.get('frame_idx', None)
                
                total_frames += data.shape[0]

                outputs = model(data).detach().cpu()
                # print(torch.unique(target))
                if validation:
                    val_loss += F.cross_entropy(outputs, target, ignore_index=255)
                
                metrics.update(outputs, target, video_ids, frame_indices)
        
        end_time = time.time()
        fps = total_frames / (end_time - start_time) if (end_time - start_time) > 0 else 0
        
        res = {
            'val_loss': val_loss,
            'wDice_avg': metrics.wdice_avg(), 'wDice_std': metrics.wdice_std(),
            'mIoU_avg': metrics.jaccard_avg(), 'mIoU_std': metrics.jaccard_std(),
            'F1_avg': metrics.f_score_avg(), 'F1_std': metrics.f_score_std(),
            'TC_avg': metrics.temporal_consistency_avg(), 'TC_std': metrics.temporal_consistency_std(),
        }
        
        if not validation:
            res['FPS'] = fps
            
        return res


