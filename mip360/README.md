# Notes
- The way of running the training code has changed for later support of accelerated kernel
```
python mip360/run.py --engine vmodel
# python mip360/run.py --engine tile_cuda --tile_order tilecenter # accelerated kernel will soon be implemented
```