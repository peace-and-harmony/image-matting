# Image matting
* Automatically extracting the alpha matte via deep learning

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [The Story so Far](#the-story-so-far)
  * [Notebooks](#notebooks)
    * [U2Net](#u2net)
    * [MODNet](#modnet)
    * [TensorRT](#tensorrt)
    * [Data Prep](#data-prep)
  * [Scripts and Tools](#scripts-and-tools)
  * [Research](#research)
  * [Reporting](#reporting)
  * [Supplementary Data](#supplementary-data)
* [Proposed Updates](#proposed-updates)
* [License](#license)
* [Contact](#contact)

<!-- ABOUT THE PROJECT -->
## About The Project

Image matting is the research area that algorithms can accurately extract the foreground object of the corresponding photos or videos. The produced alpha matte (right sub-figure in Figure 1) can further used for extracting the alpha matte (middle in Figure 1) from the original image(left in Figure 1).

![product-screenshot-tbc](images/download.png)

### Built With

* [ONNX](https://onnx.ai/)
* [OpenCV](https://opencv.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [PyTorch](https://pytorch.org/)
* [TesnorRT](https://developer.nvidia.com/tensorrt)
* [MODNet](https://github.com/ZHKKKe/MODNet)
<!-- * [U-2-Net](https://github.com/xuebinqin/U-2-Net) -->



<!-- GETTING STARTED -->
## Getting Started

### The Story So Far

MODNet was designed for portrait matting. Here, we adapted the architectures of MODNet to the domain of clothing matting.

#### MODNet


1. [MODNet demo - Benchmark edition ](notebooks/modnet_demo_benchmark_edition.ipynb) - An initial on the rails demo
of the original U2Net implementation including a Pyrtorch versus ONNX comparison benchmark. A guide as as to conversion to TensorRT and a subsequent benchmark.

 **Note** Insert a table for the resulting benchmark.

2. [MODNet sandboxed training](notebooks/MODNet_train.ipynb) - A brief runtrhough of what a training run looks like whether via an Ipython interactive environment, local machine or vm instance. The notebook guides the user through merging both the Fashionpedia and Pronti datsets, allows a user to switch between the original augs and or custom augs, along with allowing a user to continue training a stalled session from the command line.

- Guided meging of both the Fashionpedia and Pronti dataset.
- Implementation of custom PyTorch based transforms.
- Initialisation from a pretrained backbone.
- The ability to resume after a stalled or stopped session.

3. [MODNet eval](notebooks/MODNet_eval.ipynb) - As the label describes on the tin the, follow the notebook's colab badge in order to receive an on the rails guide as to evaluation procedure with an active checkpoint. Further functionality includes:

 - The ability to strip a training checkpoint ready for serving.


4. [MODNet ONNX conversion](notebooks/u2net/u2net_onnx_throughput.ipynb) - A sandboxed implementation of `u2net_onnx_mp.py` allowing the following:

 - Adjustment of a prepared onnx model to a chosen static batch.
 - Conversion of a provided datset to a base64 encoded string tagged by unique uuid, alternatively a prepared json is available via our shared [folder](https://drive.google.com/drive/folders/168swtLLjG722I2nzggNr2FCbVOS-1KqN?usp=sharing) based on the datset that's currently being labelled.
 - Mock implementation and streaming of incoming data. Allowing a user to either view per-batch benchmarks as they're available or visualise incoming cropped images decoded from the ouput base64 json.

 **Note** For demo purposes this implementation has been stripped from flask and simulates the scenario with a seperate thread feeding random chunks of the overall json. The concept still applies however should you wish to implement the module with Flask and a filesystem listener etc.

 Please also note that a breakdown of the module itself along with advanced usage can also be found at the U2Net landing [page](scripts/u2net/README.md) once again.

#### TensorRT

1. [Resnet50 - onnx - TensorRT](notebooks/tensorrt/resnet50_tensorrt.ipynb) - A full Tesnorflow/ Keras based run through of the huge speedup that can be yielded by utlising some of the emerging intermediate representation platforms. The notebooks provides before and after latency benchmarks, guides the user through conversion and ultimately builds a tensort engine for device specific deployment.

    **Note** that while ResNet50 is used here all available models have been verified within this woorkflow, it's therefore possoble to used advanced architectures suchas Efficientnet B7 for a fraction of the inference cost.

    It's also worth noting the findings here also apply to Pytorch with a few tweaks.


### Scripts and Tools

Given that the the u2net scripts section has become so populated please see the u2net landing [page](scripts/u2net/README.md) for further instruction and functionality.


### Supplementary Data

While we're hoping to version control as best we can, given the nature of our work we can expect data to exceed the confines of this repo. Check here regularly or within any notebooks themselves for training data or pretrained models. As it stands all checkpoints, saved image data and results can be found at the following private drive link:

[Supporting data](https://drive.google.com/drive/folders/168swtLLjG722I2nzggNr2FCbVOS-1KqN?usp=sharing)


## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/peace-and-harmony/image-matting/edit/main/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/peace-and-harmony/image-matting/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
