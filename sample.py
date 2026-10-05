import sys

import torch
from utils.train_utils import set_random_seed

from utils import init_env
import os
import argparse
from pathlib import Path

from utils.collate_utils import collate
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.init_utils import add_args
from torch.utils.data import DataLoader
from utils.trainer import Trainer

set_random_seed(7)


def get_loader(cfg):
    test_loaders = {}
    for dataset_name, dataset_cfg in cfg.test_dataset.items():
        dataset = instantiate_from_config(dataset_cfg)
        test_loaders[dataset_name] = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            collate_fn=collate
        )
    return test_loaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_sample_steps', type=int, default=None)
    parser.add_argument('--target_dataset', nargs='+', type=str, default=None)
    parser.add_argument('--time_ensemble', action='store_true')
    parser.add_argument('--batch_ensemble', action='store_true')

    cfg = add_args(parser)
    assert not (cfg.time_ensemble and cfg.batch_ensemble), 'Cannot use both time_ensemble and batch_ensemble'
    """
        Hack config here.
    """
    if cfg.num_sample_steps is not None:
        cfg.diffusion_model.params.num_sample_steps = cfg.num_sample_steps

    test_loaders = get_loader(cfg)

    cond_uvit = instantiate_from_config(cfg.cond_uvit,
                                        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass))
    model = recurse_instantiate_from_config(cfg.model,
                                            unet=cond_uvit)

    vae = None
    if hasattr(cfg, 'vae') and cfg.vae is not None:
        vae = instantiate_from_config(cfg.vae)
        print(f"Created VAE: {cfg.vae.name}")

    if vae is not None:
        diffusion_model = instantiate_from_config(cfg.diffusion_model,
                                                  model=model, vae=vae)
    else:
        diffusion_model = instantiate_from_config(cfg.diffusion_model,
                                                  model=model)

    optimizer = instantiate_from_config(cfg.optimizer, params=model.parameters())

    trainer = Trainer(
        diffusion_model,
        train_loader=None, test_loader=None,
        train_val_forward_fn=get_obj_from_str(cfg.train_val_forward_fn),
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        log_with=None,
        cfg=cfg,
    )

    trainer.load(pretrained_path=cfg.checkpoint)
    prepared_test_loaders = trainer.accelerator.prepare(*test_loaders.values())
    if len(test_loaders) == 1:
        prepared_test_loaders = (prepared_test_loaders,)
    dataset_map = dict(zip(test_loaders.keys(), prepared_test_loaders))

    if cfg.target_dataset is None:
        cfg.target_dataset = list(dataset_map.keys())

    assert all(dataset_name in dataset_map for dataset_name in cfg.target_dataset), \
        f'Invalid dataset name. Available datasets: {list(dataset_map.keys())}. Your input: {cfg.target_dataset}'

    for dataset_name in cfg.target_dataset:
        dataset = dataset_map[dataset_name]
        trainer.model.eval()
        save_to = Path(cfg.results_folder) / dataset_name
        os.makedirs(save_to, exist_ok=True)
        if cfg.batch_ensemble:
            mae, _ = trainer.val_batch_ensemble(model=trainer.model,
                                                test_data_loader=dataset,
                                                accelerator=trainer.accelerator,
                                                thresholding=False,
                                                save_to=save_to)
        elif cfg.time_ensemble:
            mae, _ = trainer.val_time_ensemble(model=trainer.model,
                                               test_data_loader=dataset,
                                               accelerator=trainer.accelerator,
                                               thresholding=False,
                                               save_to=save_to)
        else:
            mae, _ = trainer.val(model=trainer.model,
                                 test_data_loader=dataset,
                                 accelerator=trainer.accelerator,
                                 thresholding=False,
                                 save_to=save_to)
        trainer.accelerator.wait_for_everyone()
        trainer.accelerator.print(f'{dataset_name} mae: {mae}')

        if trainer.accelerator.is_main_process:
            from utils.eval import eval

            eval_score = eval(
                gt_root=cfg.test_dataset[dataset_name].params.gt_root,
                pred_dir=str(save_to),
                dataset_name=dataset_name)
        trainer.accelerator.wait_for_everyone()
