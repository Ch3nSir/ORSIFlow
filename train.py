from utils import init_env

import argparse

import torch
from utils.collate_utils import collate, SampleDataset
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.init_utils import add_args, config_pretty
from utils.train_utils import set_random_seed
from torch.utils.data import DataLoader
from utils.trainer import Trainer

set_random_seed(42)


def get_loader(cfg):
    train_dataset = instantiate_from_config(cfg.train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers)

    eval_dataset_names = list(getattr(cfg, 'train_eval_datasets', []) or list(cfg.test_dataset.keys()))
    eval_datasets = []
    train_eval_intervals = getattr(cfg, 'train_eval_intervals', None)

    for dataset_name in eval_dataset_names:
        if dataset_name not in cfg.test_dataset:
            raise KeyError(f'Unknown evaluation dataset: {dataset_name}')

        dataset = instantiate_from_config(cfg.test_dataset[dataset_name])
        interval = train_eval_intervals.get(dataset_name) if train_eval_intervals else None
        if interval is not None:
            dataset = SampleDataset(full_dataset=dataset, interval=interval)
        eval_datasets.append(dataset)

    if not eval_datasets:
        raise ValueError('No evaluation datasets were configured.')

    test_dataset = eval_datasets[0] if len(eval_datasets) == 1 else torch.utils.data.ConcatDataset(eval_datasets)

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    return train_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--results_folder', type=str, default=None, help='None for saving in wandb folder.')
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr_min', type=float, default=1e-6)

    cfg = add_args(parser)

    config_pretty(cfg)

    cond_uvit = instantiate_from_config(cfg.cond_uvit,
                                        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass))
    model = recurse_instantiate_from_config(cfg.model,
                                            unet=cond_uvit)
    
    # Check if VAE is configured (for latent diffusion)
    vae = None
    if hasattr(cfg, 'vae') and cfg.vae is not None:
        vae = instantiate_from_config(cfg.vae)
        print(f"Created VAE: {cfg.vae.name}")
    
    # Create diffusion model with optional VAE
    if vae is not None:
        diffusion_model = instantiate_from_config(cfg.diffusion_model,
                                                  model=model, vae=vae)
    else:
        diffusion_model = instantiate_from_config(cfg.diffusion_model,
                                                  model=model)


    train_loader, test_loader = get_loader(cfg)

    optimizer = instantiate_from_config(cfg.optimizer, params=filter(lambda p: p.requires_grad, diffusion_model.parameters()))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epoch, eta_min=cfg.lr_min)

    trainer = Trainer(
        diffusion_model, train_loader, test_loader,
        train_val_forward_fn=get_obj_from_str(cfg.train_val_forward_fn),
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer, scheduler=scheduler,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        log_with=None if cfg.num_workers == 0 else 'wandb',  # debug
        cfg=cfg,
    )
    if getattr(cfg, 'resume', None) or getattr(cfg, 'pretrained', None):
        trainer.load(resume_path=cfg.resume, pretrained_path=cfg.pretrained)
    trainer.train()
