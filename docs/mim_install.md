# Install mmocr
MMOCR provides a library to perform text detection and recognition with models like DBNet, that generate polygons instead of bounding boxes.

More information [here](https://mmocr.readthedocs.io/en/dev-1.x/user_guides/inference.html).
## Install mmcv for Torch 2.0.x and Cuda 11.8
```shell
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```
## Install other dependencies
```shell
pip install mmengine
pip instal mmdet
pip install mmocr
```