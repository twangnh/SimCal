
# Implementation of our ECCV 2020 paper [The Devil is in Classification: A Simple Framework for Long-tail Instance Segmentation]()

## Introduction
We strongly encourage you to also check our following up work after the LVIS challenge of
 [Balanced Group Softmax](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Overcoming_Classifier_Imbalance_for_Long-Tail_Object_Detection_With_Balanced_Group_CVPR_2020_paper.pdf)
accepted at CVPR20 that employs a more specific calibration approach with redesigned the softmax function, the calibration is
more effective without dual-head inference. Code is available
at [https://github.com/FishYuLi/BalancedGroupSoftmax](https://github.com/FishYuLi/BalancedGroupSoftmax)


## License

This project is released under the [Apache 2.0 license](LICENSE).


## Installation

Please refer to [INSTALL.md](INSTALL.md) for installation and dataset preparation.

Or run the following installation script:
###
    conda create -n simcal_mmdet python=3.7
    conda activate simcal_mmdet
    echo "python path"
    which python
    conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
    pip install cython==0.29.12 mmcv==0.2.16 matplotlib terminaltables
    pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
    pip install opencv-python-headless
    pip install Pillow==6.1
    pip install numpy==1.17.1 --no-deps
    pip install -v -e .
    
    mkdir data
    ln -s $COCO_ROOT data
Please do not install official lvis api, as we have modified with a local copy in the repository.
For [LVIS](https://www.lvisdataset.org/dataset) dataset, please arrange the data as:

```
mmdetection
├── configs
├── data
│   ├── LVIS
│   │   ├── lvis_v0.5_train.json.zip
│   │   ├── lvis_v0.5_val.json.zip
│   │   ├── images
│   │   │   ├── train2017
│   │   │   ├── val2017

```
>note for  LVIS images, you can just create a softlink for the val2017 to point to COCO val2017

## Training (Calibration)
The model is first trained under normal random sampling.

## Test

## Pre-trained models



@article{wang2019classification,
  title={Classification Calibration for Long-tail Instance Segmentation},
  author={Wang, Tao and Li, Yu and Kang, Bingyi and Li, Junnan and Liew, Jun Hao and Tang, Sheng and Hoi, Steven and Feng, Jiashi},
  journal={arXiv preprint arXiv:1910.13081},
  year={2019}
}

@article{wang2019classification,
  title={Classification Calibration for Long-tail Instance Segmentation},
  author={Wang, Tao and Li, Yu and Kang, Bingyi and Li, Junnan and Liew, Jun Hao and Tang, Sheng and Hoi, Steven and Feng, Jiashi},
  journal={ECCV},
  year={2020}
}

@inproceedings{li2020overcoming,
  title={Overcoming Classifier Imbalance for Long-Tail Object Detection With Balanced Group Softmax},
  author={Li, Yu and Wang, Tao and Kang, Bingyi and Tang, Sheng and Wang, Chunfeng and Li, Jintao and Feng, Jiashi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10991--11000},
  year={2020}
}
```
