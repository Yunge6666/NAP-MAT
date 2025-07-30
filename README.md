# NAP-MAT

Neighbor-Aware Token Reduction via Hilbert Curve for Vision Transformers


## **Evaluation**

NAP (Neighbor-aware Pruning) is a token pruning method. The code is inherited from **EViT** and modified. Therefore the instructions used are the same.

To evaluate DeiT-S-16/224 on ImageNet val set. Replace `base_keep_rate` with the keeping ratio you want.

```
python3 main.py --model deit_base_patch16_shrink_base --fuse_token --base_keep_rate 0.7 --eval --resume /path/to/checkpoint --data-path /path/to/imagenet
```

MAT (Merging Adjacent Tokens) is a token merging method. The code is inherited from **ToMe** and **ViT-MAE.**


HyNAP (Hybrid NAP) is inherited from **DiffRate** and modified. The instructions used are the same as **DiffRate**.

To evaluate DeiT-S-16/224 on ImageNet val set. Replace `target_flops` with the flops you want (refer to compression_rate.json according to different models).

```
python main.py --eval --load_compression_rate --data-path /path/to/imagenet --model deit_small_patch16_224 --target_flops 2.3
```



## Finetune

## Acknowledgement

We sincerely thank the following outstanding works for providing the foundation upon which our code is built.

* [EViT](https://github.com/youweiliang/evit)
* [ToMe](https://github.com/facebookresearch/ToMe)
* [DiffRate](https://github.com/OpenGVLab/DiffRate)
* [Gilbertcurve](https://github.com/jakubcerveny/gilbert)
