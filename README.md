# ORSIFlow


This repository contains the codebase for ORSIFlow: Saliency-Guided Rectified Flow for Optical Remote Sensing Salient Object Detection.ORSIFlow reformulates ORSI-SOD as a deterministic latent rectified-flow generation problem. The model performs saliency prediction in a compact VAE latent space and uses saliency-aware conditional features for efficient few-step inference.The implementation keeps the original diffusion baseline codepaths, while the main paper configuration is now provided through config/ORSIFlow_352x352.yaml.




## Architecture

![https://i.postimg.cc/90CRkX9p/network.png](https://i.postimg.cc/k5SJKnkS/flow-draft.png)



## Requirements

- python == 3.10
- cuda == 12.4
- torch==1.12

To install requirements:

```setup
conda create -n ORSIFlow python==3.10
conda activate ORSIFlow
pip install -r requirements.txt
```



## Datasets

The paper reports results on three optical remote sensing saliency benchmarks:

- `ORSSD`
[ORSSD Github repository](https://github.com/rmcong/ORSSD-dataset)

- `EORSSD`
[EORSSD Github repository](https://github.com/rmcong/EORSSD-dataset)

- `ORSI-4199`
[ORSI-4199 Github repository](https://github.com/wchao1213/ORSI-SOD/blob/main/The%20dataset%20download%20location)

Update the dataset roots in `config/dataset_352x352.yaml` or create a local override from `config/local_ORSIFlow_352x352.yaml`.

Download the following ORSI SOD datasets and organize them as follows:

```
media/dataset/
        ├── ORSSD dataset/
        │   ├── TrainDataset/
        │   │   ├── image train/
        │   │   └── GTtrain/
        │   └── TestDataset/
        │       ├── image test/
        │       └── GTtest/
 		    ├── EORSSD/
        │   ├── train-images/
        │   ├── train-labels/
        │   ├── test-images/        
        │   └── test-labels/	
        └── ORSI-4199
```



## Training

To train the model(s) in the paper, run this command:

```shell
accelerate launch train.py \
  --config config/ORSIFlow_352x352.yaml \
  --num_epoch 150 \
  --batch_size 32 \
  --gradient_accumulate_every 1
```



## Evaluation
To test a model, run [sample.py](sample.py) with the desired model on different datasets:
```shell
accelerate launch sample.py \
  --config config/ORSIFlow_352x352.yaml \
  --results_folder ${RESULT_SAVE_PATH} \
  --checkpoint ${CHECKPOINT_PATH} \
  --num_sample_steps 3 \
  --target_dataset ORSSD EORSSD ORSI-4199 \
  --time_ensemble
```

For ease of use, we create a [eval.sh](scripts%2Feval.sh) script and a use case in the form of a shell script eval.sh.
You can edit the script to change the parameters you want to test.

```shell
bash scripts/eval.sh
```



## Citation

```bibtex
@inproceedings{orsiflow2026,
  title={ORSIFlow: Saliency-Guided Rectified Flow for Optical Remote Sensing Salient Object Detection},
  author    = {Chen, Haojing and Li, Yutong and Liu, Zhihang and Tan, Tao and Bian, Haoyu and Ma, Qiuju},
  booktitle={IEEE International Conference on Multimedia and Expo},
  year={2026}
}
```
