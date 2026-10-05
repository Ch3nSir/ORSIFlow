import torch
from torch import nn
import torch.nn.functional as F



def normalize_to_01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def simple_train_val_forward(model: nn.Module, gt=None, image=None, **kwargs):
    if model.training:
        assert gt is not None and image is not None
        return model(gt, image, **kwargs)
    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        pred = model.sample(image, **kwargs)
        if time_ensemble:
            preds = torch.concat(model.history, dim=1).detach().cpu()
            pred = torch.mean(preds, dim=1, keepdim=True)

            def process(i, p, gt_size):
                p = F.interpolate(p.unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                p = normalize_to_01(p)
                ps = F.interpolate(preds[i].unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                preds_round = (ps > 0).float().mean(dim=1, keepdim=True)
                p_postion = (preds_round > 0.5).float()
                p = p_postion * p
                return p

            pred = [process(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred, gt_sizes))]
        return {
            "image": image,
            "pred": pred,
            "gt": gt if gt is not None else None,
        }


def modification_train_val_forward(model: nn.Module, gt=None, image=None, seg=None, **kwargs):
    """This is for the modification task. When diffusion model add noise, will use seg instead of gt."""
    if model.training:
        assert gt is not None and image is not None and seg is not None
        return model(gt, image, seg=seg, **kwargs)
    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        pred = model.sample(image, **kwargs).detach().cpu()
        if time_ensemble:
            """ Here is the function 3, Uncertainty based"""
            preds = torch.concat(model.history, dim=1).detach().cpu()
            pred = torch.mean(preds, dim=1, keepdim=True)

            def process(i, p, gt_size):
                p = F.interpolate(p.unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                p = normalize_to_01(p)
                ps = F.interpolate(preds[i].unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                preds_round = (ps > 0).float().mean(dim=1, keepdim=True)
                p_postion = (preds_round > 0.5).float()
                p = p_postion * p
                return p

            pred = [process(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred, gt_sizes))]
        return {
            "image": image,
            "pred": pred,
            "gt": gt if gt is not None else None,
        }


def modification_train_val_forward_e(model: nn.Module, gt=None, image=None, seg=None, **kwargs):
    """This is for the modification task. When diffusion model add noise, will use seg instead of gt."""
    if model.training:
        assert gt is not None and image is not None and seg is not None
        return model(gt, image, seg=seg, **kwargs)
    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        pred = model.sample(image, **kwargs).detach().cpu()
        if time_ensemble:
            """ Here is extend function 4, with batch extend."""
            preds = torch.concat(model.history, dim=1).detach().cpu()
            for i in range(2):
                model.sample(image, **kwargs)
                preds = torch.cat([preds, torch.concat(model.history, dim=1).detach().cpu()], dim=1)
            pred = torch.mean(preds, dim=1, keepdim=True)

            def process(i, p, gt_size):
                p = F.interpolate(p.unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                p = normalize_to_01(p)
                ps = F.interpolate(preds[i].unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                preds_round = (ps > 0).float().mean(dim=1, keepdim=True)
                p_postion = (preds_round > 0.5).float()
                p = p_postion * p
                return p

            pred = [process(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred, gt_sizes))]
        return {
            "image": image,
            "pred": pred,
            "gt": gt if gt is not None else None,
        }


def latent_flow_train_val_forward(model: nn.Module, gt=None, image=None, seg=None, **kwargs):
    """
    Training and validation forward function for Latent Rectified Flow.
    
    Handles the VAE encoding/decoding automatically through the model.
    """
    if model.training:
        assert gt is not None and image is not None
        # Model handles VAE encoding internally
        if seg is not None:
            return model(gt, image, seg=seg, **kwargs)
        else:
            return model(gt, image, **kwargs)
    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        
        # Sample returns decoded images
        pred = model.sample(image, **kwargs).detach().cpu()
        
        if time_ensemble:
            # Get history of latent predictions, decode each
            preds_list = []
            for z_hist in model.history:
                # Decode latent to image
                with torch.no_grad():
                    decoded = model.decode_from_latent(z_hist)
                    decoded = (decoded.clamp(-1, 1) + 1) / 2  # Back to [0,1]
                preds_list.append(decoded.cpu())
            
            preds = torch.cat(preds_list, dim=1)
            pred = torch.mean(preds, dim=1, keepdim=True)
            
            def process(i, p, gt_size):
                p = F.interpolate(p.unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                p = normalize_to_01(p)
                ps = F.interpolate(preds[i].unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                preds_round = (ps > 0).float().mean(dim=1, keepdim=True)
                p_postion = (preds_round > 0.5).float()
                p = p_postion * p
                return p
            
            pred = [process(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred, gt_sizes))]
        
        return {
            "image": image,
            "pred": pred,
            "gt": gt if gt is not None else None,
        }
