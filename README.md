<!--
Copyright (C) 2017, 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# CASENet: Deep Category-Aware Semantic Edge Detection

## Features

This source code package contains the C++/Python implementation of our CASENet based on [Caffe](http://github.com/BVLC/caffe) for multi-label semantic edge detection training/testing. There are two folders in this package:

* caffe

    Our modified Caffe (based on the official Caffe [commit](https://github.com/BVLC/caffe/commit/4efdf7ee49cffefdd7ea099c00dc5ea327640f04) on June 20, 2017), with the C++ *MultiChannelReweightedSigmoidCrossEntropyLossLayer* implementing the multi-label loss function as explained in equation (1) of the CASENet paper.

    We also modified the C++ *ImageSegDataLayer* from [DeepLab-v2](https://bitbucket.org/aquariusjay/deeplab-public-ver2) for reading multi-label edge ground truth stored as binary file.

* CASENet

    Python scripts and network configurations for training/testing/visualization.
    Note that initial and trained model weights (caffemodel files) for SBD/Cityscapes dataset can be downloaded separately from https://doi.org/10.5281/zenodo.7872003.

## Installation

### Compile

1. Clone this repository to create the following folder structure:

```
${PACKAGE_ROOT}
├── caffe
│   ├── build
│   ├── cmake
│   ├── data
│   ├── docker
│   ├── docs
│   ├── examples
│   ├── include
│   ├── matlab
│   ├── models
│   ├── python
│   ├── scripts
│   ├── src
│   └── tools
└── CASENet
    ├── cityscapes
    │   ├── config
    │   └── model
    └── sbd
        ├── config
        └── model
```

2. Follow the official Caffe's [installation guide](http://caffe.berkeleyvision.org/install_apt.html) to compile the modified Caffe in `${PACKAGE_ROOT}/caffe` (building with cuDNN is supported).

3. Make sure to build pycaffe.

## Usage

### Using Trained Weights

We have supplied trained weights for easier testing. To use them, simply download them separately from https://doi.org/10.5281/zenodo.7872003 to `${PACKAGE_ROOT}/CASENet/sbd/model`. Similarly for Cityscapes.

### Experiments

Assume pycaffe is installed in `${PACKAGE_ROOT}/caffe/build/install/python`. Following instructions use CASENet on Cityscapes dataset as an example. Baselines (Basic/DSN/CASENet-) run similarly. For SBD dataset, change all "cityscapes" to "sbd" in the following instructions.

1. If you want to train the network for other datasets, in the `${PACKAGE_ROOT}/CASENet/cityscapes/config/train_CASENet`.prototxt, modify the *root_folder* and *source* at lines 27-28 to point to your dataset.

2. Run the following commands to perform training and testing:
```
cd ${PACKAGE_ROOT}/CASENet/cityscapes
# Training
python solve.py ./config/solver_CASENet.prototxt -c ../../caffe/build/install/python

# Testing
python test.py ./config/test_CASENet.prototxt ./model/CASENet_iter_40000.caffemodel -c ../../caffe/build/install/python -l ${Cityscapes_DATASET}/val.txt -d ${Cityscapes_DATASET} -o ./result_CASENet

# Visualization (note visualization for SBD is slightly different)
python visualize_multilabel.py ${Cityscapes_DATASET}/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png
```

3. Check `${PACKAGE_ROOT}/CASENet/cityscapes/model` folder for trained weights and check `${PACKAGE_ROOT}/CASENet/cityscapes/result_CASENet` for testing results. Check `${PACKAGE_ROOT}/CASENet/cityscapes/visualize` for visualization results.

### Training Notes

If you want to train CASENet on your own dataset, you will need to generate multi-label ground truth that is readable by our modified ImageSegDataLayer, which is essentially a memory buffer dumped in binary format that stores multi-label ground truth image in row-major order, where each pixel of this multi-label image has 4 x num_label_chn **bytes**, i.e., 32 x num_label_chn **bits** (num_label_chn as specified in image_data_param in the training prototxt file).

For example, a toy multi-label ground truth image with num_label_chn=1 corresponding to a 2x3 input RGB image can be the following bits in memory:
```
1000000000000000000000000000000000 0000000000000000000000000000000001 0000000000000000000000000000000010
0000000000000000000000000000000101 0000000000000000000000000000001110 0000000000000000000000000000000000
```
which means the following pixel labels:
```
ignored,           edge-type-0,             edge-type-1
edge-type-0-and-2, edge-type-1-and-2-and-3, non-edge
```
Basically, we use a single bit to encode a single label of a pixel, and the highest bit (the 32-th bit) is used to label ignored pixels excluded from loss computation. More details can be found in line 265-273 of the image_seg_data_layer.cpp file.

BTW, our *MultiChannelReweightedSigmoidCrossEntropyLossLayer* currently only supports ignoring the 32-th bit (which is enough for the Cityscapes dataset): so if your max number of labels is more than 31 (e.g., the [ADE20K dataset](http://groups.csail.mit.edu/vision/datasets/ADE20K/)), your num_label_chn has to be larger than 1, and you will need to modify *MultiChannelReweightedSigmoidCrossEntropyLossLayer* correspondingly. More details can be found in line 54 and line 130 of the multichannel_reweighted_sigmoid_cross_entropy_loss_layer.cpp file.

### Generating Ground Truth Multi-label Edge Map
To generate such ground truth multi-label edge map from ground truth semantic segmentation in Cityscapes and SBD, we provide a separate code package downloadable at https://github.com/Chrisding (License: `MIT`).

### CASENet Python Scripts Usage

#### Training Script: solve.py

```
usage: solve.py [-h] [-c PYCAFFE_FOLDER] [-m INIT_MODEL] [-g GPU]
                solver_prototxt_file

positional arguments:
  solver_prototxt_file  path to the solver prototxt file

optional arguments:
  -h, --help            show this help message and exit
  -c PYCAFFE_FOLDER, --pycaffe_folder PYCAFFE_FOLDER
                        pycaffe folder that contains the caffe/_caffe.so file
  -m INIT_MODEL, --init_model INIT_MODEL
                        path to the initial caffemodel
  -g GPU, --gpu GPU     use which gpu device (default=0)
```

#### Testing Script: test.py

```
usage: test.py [-h] [-l IMAGE_LIST] [-f IMAGE_FILE] [-d IMAGE_DIR]
               [-o OUTPUT_DIR] [-c PYCAFFE_FOLDER] [-g GPU]
               deploy_prototxt_file model

positional arguments:
  deploy_prototxt_file  path to the deploy prototxt file
  model                 path to the caffemodel containing the trained weights

optional arguments:
  -h, --help            show this help message and exit
  -l IMAGE_LIST, --image_list IMAGE_LIST
                        list of image files to be tested
  -f IMAGE_FILE, --image_file IMAGE_FILE
                        a single image file to be tested
  -d IMAGE_DIR, --image_dir IMAGE_DIR
                        root folder of the image files in the list or the
                        single image file
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        folder to store the test results
  -c PYCAFFE_FOLDER, --pycaffe_folder PYCAFFE_FOLDER
                        pycaffe folder that contains the caffe/_caffe.so file
  -g GPU, --gpu GPU     use which gpu device (default=0)
```

#### Visualization Script: visualize_multilabel.py

```
usage: visualize_multilabel.py [-h] [-o OUTPUT_FOLDER] [-g GT_NAME]
                               [-f RESULT_FMT] [-t THRESH] [-c DO_EACH_COMP]
                               raw_name

positional arguments:
  raw_name              input rgb filename

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        visualization output folder
  -g GT_NAME, --gt_name GT_NAME
                        full path to the corresponding multi-label ground
                        truth file
  -f RESULT_FMT, --result_fmt RESULT_FMT
                        folders storing testing results for each class
  -t THRESH, --thresh THRESH
                        set any probability<=thresh to 0
  -c DO_EACH_COMP, --do_each_comp DO_EACH_COMP
                        if gt_name is not None, whether to visualize each
                        class component (1) or not (0)
```

## Modified code
We have modified Caffe, an open-source deep learning tool, to implement our CASENet paper. The modification details are listed below.

1. Newly added (for CASENet):
```
include/caffe/layers/multichannel_reweighted_sigmoid_cross_entropy_loss_layer.hpp
src/caffe/layers/multichannel_reweighted_sigmoid_cross_entropy_loss_layer.cpp
```

2. Copied/modified from the official/DeepLab-v2/HED versions of Caffe (for I/O):
```
include/caffe/layers/image_dim_prefetching_data_layer.hpp
include/caffe/layers/image_seg_data_layer.hpp
include/caffe/layers/base_data_layer.hpp
include/caffe/data_transformer.hpp
include/caffe/layers/reweighted_sigmoid_cross_entropy_loss_layer.hpp
src/caffe/layers/image_dim_prefetching_data_layer.cpp
src/caffe/layers/image_dim_prefetching_data_layer.cu
src/caffe/layers/image_seg_data_layer.cpp
src/caffe/layers/reweighted_sigmoid_cross_entropy_loss_layer.cpp
src/caffe/data_transformer.cpp
src/caffe/proto/caffe.proto
```

## Citation

If you use the software, please cite the following  ([TR2017-100](https://www.merl.com/publications/TR2017-100)):

```BibTex
@inproceedings{Yu2017jul,
  author = {Yu, Zhiding and Feng, Chen and Liu, Ming-Yu and Ramalingam, Srikumar},
  title = {CASENet: Deep Category-Aware Semantic Edge Detection},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = 2017,
  month = jul,
  doi = {10.1109/CVPR.2017.191},
  url = {https://www.merl.com/publications/TR2017-100}
}
```

## Contact

Tim K Marks (<tmarks@merl.com>)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as listed below:

```
Copyright (C) 2017, 2023 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: AGPL-3.0-or-later
```

All files in `caffe` folder (except:
`caffe/include/caffe/layers/multichannel_reweighted_sigmoid_cross_entropy_loss_layer.hpp` and
`caffe/src/caffe/layers/multichannel_reweighted_sigmoid_cross_entropy_loss_layer.cpp`)
, and `requirements.txt`:

```
Copyright:

All contributions by the University of California:
Copyright (c) 2014-2017 The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014-2017, the respective contributors
All rights reserved.

Caffe uses a shared copyright model: each contributor holds copyright over
their contributions to Caffe. The project versioning records all such
contribution and copyright details. If a contributor wants to further mark
their specific copyright on a particular contribution, they should indicate
their copyright solely in the commit message of the change when it is
committed.

License: BSD-2-Clause
```

and the files below which were copied/modified from the official/DeepLab-v2/HED versions of Caffe (for I/O):
```
caffe/include/caffe/layers/image_dim_prefetching_data_layer.hpp
caffe/include/caffe/layers/image_seg_data_layer.hpp
caffe/include/caffe/layers/base_data_layer.hpp
caffe/include/caffe/data_transformer.hpp
caffe/include/caffe/layers/reweighted_sigmoid_cross_entropy_loss_layer.hpp
caffe/src/caffe/layers/image_dim_prefetching_data_layer.cpp
caffe/src/caffe/layers/image_dim_prefetching_data_layer.cu
caffe/src/caffe/layers/image_seg_data_layer.cpp
caffe/src/caffe/layers/reweighted_sigmoid_cross_entropy_loss_layer.cpp
caffe/src/caffe/data_transformer.cpp
caffe/src/caffe/proto/caffe.proto
```

```
COPYRIGHT

All new contributions compared to the original Caffe branch:
Copyright (c) 2015, 2016, Liang-Chieh Chen (UCLA, Google), George Papandreou (Google),
Iasonas Kokkinos (CentraleSupélec /INRIA), Jonathan T. Barron(Google),
Yi Yang (Baidu), Jiang Wang (Baidu), Wei Xu (Baidu),
Kevin Murphy (Google), and Alan L. Yuille (UCLA, JHU)
All rights reserved.

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.

Caffe uses a shared copyright model: each contributor holds copyright over
their contributions to Caffe. The project versioning records all such
contribution and copyright details. If a contributor wants to further mark
their specific copyright on a particular contribution, they should indicate
their copyright solely in the commit message of the change when it is
committed.

License: BSD-2-Clause
```
