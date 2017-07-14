## What's News
* Add image data layer to support multi-label output.  
* Add more augmentation method for data transform.  
    An example usage is (3x64x64 output):  
```
layer {
  name: "data"
  type: "ImageData"
  top: "data"
  top: "label"
  top: "score"
  include {
    phase: TRAIN
  }
  transform_param {
    # scale: 0.00390625
    mirror: true
    # crop_size: 0          # crop_size and crop_pad cannot be specified at the same time
    # mean_file: "mean_file"
    mean_value: 127         # repeat this option for other channels
    # force_color: false    # Force the decoded image to have 3 color channels.
    # force_gray: false     # Force the decoded image to have 1 color channels.

    ############################Notice################################
    # If you would like to use below Augmentation
    # Please clone caffe source from https://github.com/BobLiu20/caffe

    max_rotate_angle: 2.0   # Specify the angle for doing random rotate
    min_contrast: 0.8       # contrast, brightness in random
    max_contrast: 1.2       # contrast, brightness in random
    max_brightness_shift: 5 # contrast, brightness in random
    max_smooth: 6           # random kernel size of blur
    max_color_shift: 10     # random shift R, G, B 
    crop_pad: 15            # random cutting edge in top, bottom, left and right
    crop_pad_new_size: 64   # for crop_pad. resize image to it after all pre-process
  }
  image_data_param {
    source: "train.txt"
    batch_size: 256
    shuffle: true
  }
}

layer {
  name: "data"
  type: "ImageData"
  top: "data"
  top: "label"
  top: "score"
  include {
    phase: TEST
  }
  transform_param {
    # scale: 0.00390625
    mirror: false
    # crop_size: 0          # crop_size and crop_pad cannot be specified at the same time
    # mean_file: "mean_file"
    mean_value: 127         # repeat this option for other channels

    ############################Notice################################
    # If you would like to use below Augmentation
    # Please clone caffe source from https://github.com/BobLiu20/caffe

    crop_pad: 15            # random cutting edge in top, bottom, left and right
    crop_pad_new_size: 64   # for crop_pad. resize image to it after all pre-process
  }
  image_data_param {
    source: "train.txt"
    batch_size: 256
  }
}
```




# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by Berkeley AI Research ([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning Center (BVLC) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BAIR reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

## Custom distributions

 - [Intel Caffe](https://github.com/BVLC/caffe/tree/intel) (Optimized for CPU and support for multi-node), in particular Xeon processors (HSW, BDW, Xeon Phi).
- [OpenCL Caffe](https://github.com/BVLC/caffe/tree/opencl) e.g. for AMD or Intel devices.
- [Windows Caffe](https://github.com/BVLC/caffe/tree/windows)

## Community

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
