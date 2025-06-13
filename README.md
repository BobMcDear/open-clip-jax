# CLIP in JAX/Flax

• <strong>[Introduction](#introduction)</strong><br>
• <strong>[Installation](#installation)</strong><br>
• <strong>[Usage](#usage)</strong><br>
• <strong>[Training](#training)</strong><br>
&nbsp;&nbsp;&nbsp;&nbsp;• <strong>[Dataset Preparation](#dataset-preparation)</strong><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• <strong>[CSV](#csv)</strong><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• <strong>[TFRecord](#tfrecord)</strong><br>
&nbsp;&nbsp;&nbsp;&nbsp;• <strong>[Single-Worker Training](#single-worker-training)</strong><br>
&nbsp;&nbsp;&nbsp;&nbsp;• <strong>[Multi-Worker Training](#multi-worker-training)</strong><br>
• <strong>[Available Models](#available-models)</strong><br>
• <strong>[Acknowledgements](#acknowledgements)</strong><br>
• <strong>[Citations](#citations)</strong><br>


## Introduction

```open_clip_jax``` is an open source JAX/Flax implementation of OpenAI's [CLIP](https://arxiv.org/abs/2103.00020), including image and text towers, pre-trained parameters, training utilities, and more. It is inspired by but not affiliated with [OpenCLIP](https://github.com/mlfoundations/open_clip) and aims to deliver similar functionalities with a JAX backend.

## Installation

The JAX installation process may differ depending on one's machine, so JAX needs to be [installed manually](https://github.com/google/jax#installation) by the user. Afterwards, ```open_clip_jax``` can be installed through ```pip install git+https://github.com/BobMcDear/open-clip-jax.git```.

## Usage

```CLIPInference``` is a convenience class for conducting inference, which can be called on raw images and texts to compute their similarity scores, as demonstrated below.

```python
import jax
from PIL import Image
from open_clip_jax import CLIPInference


clip = CLIPInference(
    'vit-base-patch32',
    softmax_temp=100.,
    pretrained='laion2b-s34b-b79k',
    )
image = Image.open('CLIP.png').convert('RGB')
text = ['A diagram', 'A dog', 'A cat']

# image and text can be single data points or lists.
probs, _ = clip(image, text)
print(probs)
```

Under the hood, ```CLIPInference``` utilizes ```create_model_with_params``` to create the CLIP model, ```create_image_transforms``` to pre-process the image(s), and ```tokenize``` to tokenize the text(s). A sample usage of these functions, equivalent to the code above, is exhibited in the following snippet. Breaking ```CLIPInference``` into these smaller components can offer greater flexibility.

```python
from typing import Dict

import jax
from PIL import Image
from jax import Array
from open_clip_jax import create_image_transforms, create_model_with_params, tokenize


model, vars = create_model_with_params(
    'vit-base-patch32',
    pretrained='laion2b-s34b-b79k',
    )
image_transforms = create_image_transforms(
    train=False,
    input_format='image',
    do_batch_transforms=False,
    )

image = image_transforms(Image.open('CLIP.png').convert('RGB'))._numpy()
image = np.expand_dims(image, axis=0)
text = tokenize(['A diagram', 'A dog', 'A cat'])._numpy()

def calculate_similarity(vars: Dict, image: Array, text: Array) -> Array:
    # CLIP returns L2-normalized image and text features.
    image_proj, text_proj = model.apply(vars, image, text)
    return nn.softmax(100 * image_proj @ text_proj.T)

probs = jax.jit(calculate_similarity)(vars, image, text)
print(probs)
```

## Training

This repository also supports training CLIP models from scratch, using either the utilities supplied by ```open_clip_jax.training``` for more fine-grained control or ```main.py``` for a fully-featured training script. The ensuing sections elaborate on training with ```main.py```.

### Dataset Preparation

```main.py``` accepts two data formats, CSV files or TFRecords. The latter should generally be preferred as ```tf.data``` pipelines constructed around TFRecords are quite efficient, especially if the data is stored remotely, but training using CSV files can be more convenient and should not be an issue when dealing with smaller datasets. Dataset preparation instructions for each case are outlined below.

#### CSV

To prepare a dataset for CSV training, a CSV file needs to be created with one column containing image paths and another holding text captions corresponding to each image. Other columns may be included as well, but they are not read, and the order of the columns is also ignored. For instance, the table below displays how such a file may be structured.

| caption                   |image_path |
|--------------------------------|------------|
| Diagram of OpenAI's CLIP model | clip.jpg   |
| A Siamese cat                  |  cat.jpg    |
| Dog running on grass           |  dog.jpg    |
| ...                            | ...        |

TensorFlow integrates seamlessly with Google Cloud Storage (GCS), so the CSV file or images may be stored in a GCS bucket, as can be seen below. However, doing so would slow down data loading since GCS has a high time to first byte (TTFB), and therefore TFRecords would be the appropriate option if storing data in the cloud.

| caption                   |image_path |
|--------------------------------|------------|
| Diagram of OpenAI's CLIP model | gs://open_clip_jax/clip.jpg   |
| A Siamese cat                  |  gs://open_clip_jax/cat.jpg    |
| Dog running on grass           |  gs://open_clip_jax/dog.jpg    |
| ...                            | ...        |



#### TFRecord

To prepare a dataset for TFRecord training, every image-text pair must be written as a ```tf.train.Example```/Protobuf message (images as JPEG-encoded bytes, text captions as strings) to TFRecord files stored in a single directory (local or in a GCS bucket), with ideally 100+ MB of data, or 10,000 samples, per file. [img2dataset](https://github.com/rom1504/img2dataset) can automatically convert image URLs to such datasets by setting the output format to TFRecord via ```--output_format tfrecord``` and supports many popular datasets, e.g., COCO, LAION-400M, etc.

### Single-Worker Training

In single-worker settings, assuming JAX and ```open_clip_jax``` have been installed, ```main.py``` simply needs to be downloaded and executed to begin training. If using cloud computing, the remote server should be logged into first, the data optionally transferred to it, and finally the following commands can be run, ideally in a [```tmux```](https://github.com/tmux/tmux/wiki) or [```screen```](https://www.gnu.org/software/screen/) session.

```bash
wget https://raw.githubusercontent.com/BobMcDear/open-clip-jax/main/main.py -q
python3 open-clip-jax/main.py \
    --train-path train.csv \
    --valid-path valid.csv \
    --image-key image_path \
    --text-key caption \
    --global-batch-size 128 \
    --model-name vit-base-patch32 \
    --learning-rate 1e-3 \
    --n-epochs 30
```

### Multi-Worker Training

Multi-worker training has been tested only for TPUs, although the process should remain largely identical for GPU clusters. To train on a pod slice, the same commands are sent to every worker in parallel to install the necessary packages and start training. The data must also be in a GCS bucket in the same zone as the VM. An end-to-end minimal example, with TPU creation and deletion commands included, can be seen below.

```bash
NAME=open_clip_jax
ZONE=us-central1-a
TYPE=v3-32
VERSION=v2-alpha

# Create pod slice
gcloud compute tpus tpu-vm create $NAME \
    --zone=$ZONE \
    --accelerator-type=$TYPE \
    --version=$VERSION

# Connect to TPUs and train
gcloud compute tpus tpu-vm ssh $NAME \
    --zone $ZONE \
    --worker=all \
    --command "
        pip install -U pip &&
        pip install -U jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html &&
        pip install git+https://github.com/BobMcDear/open-clip-jax.git &&
        wget https://raw.githubusercontent.com/BobMcDear/open-clip-jax/main/main.py -q &&
        python3 main.py \
            --train-path gs://open_clip_jax/train_tfrecords/ \
            --valid-path gs://open_clip_jax/valid_tfrecords/ \
            --image-key jpg \
            --text-key caption \
            --global-batch-size 1024 \
            --model-name vit-base-patch32 \
            --learning-rate 3e-3 \
            --n-epochs 30 \
            --checkpoint-dir gs://open_clip_jax/checkpoints/
        "

# Delete VM
gcloud compute tpus tpu-vm delete $NAME \
    --zone $ZONE
```

An important caveat that should be borne in mind is that epoch boundaries become blurry if the number of batches assigned to each worker varies. For example, suppose a dataset consists of two TFRecord files, one containing 32 samples and the other 64, and training is being performed on two workers (each receiving one TFRecord file) with a per-worker batch size of 16: In this scenario, the number of steps per epoch is calculated to be 3 = (64 + 32) / 32 (the 32 in the denominator is the global batch size), so 3 batches, or 48 = 16 * 3 samples, are taken from each file in every epoch. Consequently, during the initial epoch, half the samples from the first file are iterated over twice, whereas there are samples in the second file that are not seen at all. In the subsequent epoch, however, the remaining samples from the second file will be the first to be grabbed, and the process restarts once the entire file has been consumed. Such extreme discrepancies should be rare as the number of files increases and are unlikely to pose a problem in practice, but one should be cognizant of them nevertheless.

## Available Models

There are three functions related to listing available models and pre-trained parameters:

* ```list_models```: Returns the name of every model, but some, such as ViT-Small, do not have associated pre-trained parameters.
* ```list_pretrained```: Returns tuples of (name of model, name of pre-trained parameters). A model may have several groups of pre-trained parameters, so
there may be multiple entries with identical model names but different pre-trained parameters.
* ```list_pretrained_by_model```: Returns a particular model's pre-trained parameters.

```python
>>> import open_clip_jax
>>> open_clip_jax.list_models()
('convnext-base-w',
 'convnext-base',
 'convnext-large-d',
 'vit-base-patch16',
 'vit-base-patch32',
 'vit-huge-patch14',
 'vit-huge-patch16',
 'vit-large-patch14',
 'vit-large-patch16',
 'vit-nano-patch32',
 'vit-small-patch16',
 'vit-small-patch32')
>>> open_clip_jax.list_pretrained()
(('convnext-base', 'laion400m-s13b-b51k'),
 ('convnext-base-w', 'laion-aesthetic-s13b-b82k'),
 ('convnext-base-w', 'laion-aesthetic-s13b-b82k-320'),
 ('convnext-base-w', 'laion-aesthetic-s13b-b82k-augreg-320'),
 ('convnext-base-w', 'laion2b-s13b-b82k'),
 ('convnext-base-w', 'laion2b-s13b-b82k-augreg'),
 ('convnext-large-d', 'laion2b-s26b-b102k-augreg'),
 ('convnext-large-d', 'laion2b-s29b-b131k-ft-320'),
 ('convnext-large-d', 'laion2b-s29b-b131k-ft-soup-320'),
 ('vit-base-patch32', 'laion400m-e31'),
 ('vit-base-patch32', 'laion400m-e32'),
 ('vit-base-patch32', 'laion2b-e16'),
 ('vit-base-patch32', 'laion2b-s34b-b79k'),
 ('vit-base-patch32', 'datacomp-m-s128m-b4k'),
 ('vit-base-patch32', 'datacomp-xl-s13b-b90k'),
 ('vit-base-patch16', 'laion400m-e31'),
 ('vit-base-patch16', 'laion400m-e32'),
 ('vit-base-patch16', 'laion2b-s34b-b88k'),
 ('vit-base-patch16', 'dfn2b'),
 ('vit-large-patch14', 'laion400m-e31'),
 ('vit-large-patch14', 'laion400m-e32'),
 ('vit-large-patch14', 'laion2b-s32b-b82k'),
 ('vit-huge-patch14', 'laion2b-s32b-b79k'))
>>> open_clip_jax.list_pretrained_by_model('vit-base-patch32')
('laion400m-e31', 'laion400m-e32', 'laion2b-e16', 'laion2b-s34b-b79k')
```

The pre-trained parameters have been ported from OpenCLIP, and more information regarding them, such as their training recipes or zero-shot performance, can be found in the OpenCLIP repository or as model cards on [Hugging Face Hub](https://huggingface.co/models?library=open_clip).


## Acknowledgements

Thanks to Google's [TPU Research Cloud (TRC) program](https://sites.research.google/trc/about/) for providing hardware used to accelerate the development of this project.

## Citations

```bibtex
@software{ilharco_gabriel_2021_5143773,
  author       = {Ilharco, Gabriel and
                  Wortsman, Mitchell and
                  Wightman, Ross and
                  Gordon, Cade and
                  Carlini, Nicholas and
                  Taori, Rohan and
                  Dave, Achal and
                  Shankar, Vaishaal and
                  Namkoong, Hongseok and
                  Miller, John and
                  Hajishirzi, Hannaneh and
                  Farhadi, Ali and
                  Schmidt, Ludwig},
  title        = {OpenCLIP},
  month        = jul,
  year         = 2021,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5143773},
  url          = {https://doi.org/10.5281/zenodo.5143773}
}
```

```bibtex
@inproceedings{Radford2021LearningTV,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and A. Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={ICML},
  year={2021}
}
```

```bibtex
@inproceedings{schuhmann2022laionb,
  title={{LAION}-5B: An open large-scale dataset for training next generation image-text models},
  author={Christoph Schuhmann and
          Romain Beaumont and
          Richard Vencu and
          Cade W Gordon and
          Ross Wightman and
          Mehdi Cherti and
          Theo Coombes and
          Aarush Katta and
          Clayton Mullis and
          Mitchell Wortsman and
          Patrick Schramowski and
          Srivatsa R Kundurthy and
          Katherine Crowson and
          Ludwig Schmidt and
          Robert Kaczmarczyk and
          Jenia Jitsev},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  url={https://openreview.net/forum?id=M3Y74vmsMcY}
}
```