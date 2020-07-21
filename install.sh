#!/usr/bin/env bash
conda create -n simcal_mmdet python=3.7
source ~/anaconda3/etc/profile.d/conda.sh
conda init bash
conda activate simcal_mmdet
echo "python path"
which python
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
pip install cython==0.29.12 mmcv==0.2.16 matplotlib terminaltables
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install opencv-python-headless
pip install Pillow==6.1
pip install numpy==1.17.1 --no-deps
git clone https://github.com/twangnh/SimCal
cd SimCal
pip install -v -e .