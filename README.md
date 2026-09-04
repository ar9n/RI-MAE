# RI-MAE for Retrieval

## RI-MAE: Rotation-Invariant Masked AutoEncoders for Self-Supervised Point Cloud Representation Learning, [ArXiv](https://arxiv.org/abs/2409.00353)

This fork adapts the Rotation-Invariant Masked AutoEncoder (RI-MAE) for retrieval and similarity assessment of CAD and FE models.

<div  align="center">    
 <img src="./figure/RI-MAE.jpg" align=center />
</div>

## 1. Requirements
Note: These instructions apply if the repository is used within a WSL-Ubuntu distribution

GCC >= 4.9
```
sudo apt install build-essential ninja-build
```

CUDA >= 9.0
https://docs.nvidia.com/cuda/wsl-user-guide/index.html

PyTorch >= 1.7.0 and Torchvision
https://pytorch.org/get-started/previous-versions/ (Make sure that the CUDA versions match)

python >= 3.7

```
pip install -r requirements.txt
```

```
# Chamfer Distance
cd ./extensions/chamfer_dist
pip install . --no-build-isolation
# Earth mover's distance
cd ./extensions/emd
pip install . --no-build-isolation
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/altaykacan/KNN_CUDA_reborn/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## 2. Datasets

We use ShapeNet, ScanObjectNN, ModelNet40 and ShapeNetPart in this work. See [DATASET.md](./DATASET.md) for details.

## 3. RI-MAE Models
|  Task | Dataset | Config | Acc. (z/z)| Acc. (SO3/SO3)| Acc. (z/SO3)|
|  ----- | ----- |-----|  -----|  -----|  -----|
|  Pre-training | ShapeNet |[pretrain.yaml](./cfgs/SSL_models/RI_MAE.yaml)| N.A. | N.A. | N.A. |
|  Classification | ScanObjectNN |[finetune_scan_objbg.yaml](./cfgs/ScanObjectNN_models/Transformer_objectbg.yaml)|91.9% |91.9% |91.9% |
| Part segmentation| ShapeNetPart| [Transformer_seg.yaml](./cfgs/ShapeNetPart_models/Transformer_seg.yaml)| 84.3% mIoU| 84.3% mIoU| 84.3% mIoU|
| Semantic segmentation| S3DIS| [Transformer_sem_seg.yaml](./cfgs/S3DIS_models/Transformer_sem_seg.yaml)| 60.3% mIoU| 60.3% mIoU| 60.3% mIoU|

|  Task | Dataset | Config | 10w10s Acc. (z/SO3)| 10w20s Acc. (z/SO3)|     
|  ----- | ----- |-----|  -----|-----|
|  Few-shot learning | ModelNet40 |[fewshot.yaml](./cfgs/Fewshot_models/Transformer_1k.yaml)| 90.2 ± 5.5| 93.7 ± 3.5| 

## 4. RI-MAE Pre-training
To pretrain RI-MAE on ShapeNet training set, run the following command. If you want to try different models or masking ratios etc., first create a new config file, and pass its path to --config.

```
CUDA_VISIBLE_DEVICES=<GPU> python main_RIMAE.py --config cfgs/SSL_models/RI_MAE.yaml --exp_name <output_file_name>
```
## 5. RI-MAE Fine-tuning

Fine-tuning on ScanObjectNN, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main_RIMAE.py --config cfgs/ScanObjectNN_models/Transformer_objectbg.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
Few-shot learning, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main_RIMAE.py --config cfgs/Fewshot_models/Transformer_1k.yaml --finetune_model \
--ckpts <path/to/pre-trained/model> --exp_name <output_file_name> --way <5 or 10> --shot <10 or 20> --fold <0-9>
```

## Acknowledgements

Our codes are built upon [Point-BERT](https://github.com/lulutang0608/Point-BERT), [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) and [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

## Reference

```
@misc{su2024ri,
    title={RI-MAE: Rotation-Invariant Masked AutoEncoders for Self-Supervised Point Cloud Representation Learning},
    author={Su, Kunming and Wu, Qiuxia and Cai, Panpan and Zhu, Xiaogang and Lu, Xuequan and Wang, Zhiyong and Hu, Kun},
    year={2024},
    eprint={2409.00353},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
