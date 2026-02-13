# Notes
- The way of running the training code has changed for later support of accelerated kernel
```
python mip360/run.py --engine vmodel # Torch + Triton implementation taking ray samples
python mip360/run.py --engine tile_cuda --tile_order tilecenter # Cuda-accelerated implementation taking rays of a whole image
```