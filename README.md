# HAT_RCF

This project implements a **HAT model with the RCF module incorporated as an edge guidance branch** for super-resolution. The RCF module is used to provide edge-aware features to enhance high-frequency details during training.

## Updates

- (to do) Add pretrained RCF model download url
- (to do) Add "How To Train"
- (to do) ADD "How To Test"


## Environment

PyTorch >= 1.7 (Recommend NOT using torch 1.8!!! It would cause abnormal performance.)

BasicSR == 1.3.4.9

### Installation
Install Pytorch first. Then,

```python
pip install -r requirements.txt
python setup.py develop
```

## LICENSE

This project incorporates code from two open-source projects:

1. VAGRANTLYUN, under CC BY-NC-SA 4.0 (Non-commercial use only)
2. Xiangyu Chen, under Apache 2.0

See LICENSE file for details.

## References

- HAT model by ["Activating More Pixels in Image Super-Resolution Transformer" (2023), Chen, Xiangyu, et al](https://arxiv.org/abs/2205.04437)
- EDAN module by ["Edge‑enhanced infrared image
super‑resolution reconstruction
model under transformer" (2024), Hu, Lei, Long Hu, and MingHui Chen.](https://www.nature.com/articles/s41598-024-66302-8.pdf)
- RCF module by ["Richer Convolutional Features for Edge Detection" (2017), Liu, Yun, et al.](https://arxiv.org/abs/1612.02103v3)
