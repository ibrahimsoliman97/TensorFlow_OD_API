# TensorFlow Object Detection API - Training Tutorial

# Installation
1-  `git clone https://github.com/ibrahimsoliman97/TensorFlow_OD_API.git`

2- `cd TensorFlow_OD_API`

3- `pip install tensorflow-gpu==1.15`

4- `protoc object_detection/protos/*.proto --python_out=.` 

# Dataset Preparation

TensorFlow Object Detection API reads data using the TFRecord file format. a sample scripts (`create_pascal_tf_record.py`) is
provided to convert from the PASCAL VOC dataset to TFRecords.

## Generating the PASCAL VOC TFRecord files.

The raw 2012 PASCAL VOC data set is located
[here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).
To download, extract and convert it to TFRecords, run the following commands
below:

```bash
# From TensorFlow_OD_API/
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
python object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=train \
    --output_path=pascal_train.record
python object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=val \
    --output_path=pascal_val.record
```

You should end up with two TFRecord files named `pascal_train.record` and
`pascal_val.record` in the `tensorflow/models/research/` directory.

The label map for the PASCAL VOC data set can be found at
`object_detection/data/pascal_label_map.pbtxt`.


# Configuring the Object Detection Training Pipeline

## Overview

The TensorFlow Object Detection API uses protobuf files to configure the
training and evaluation process. The schema for the training pipeline can be
found in object_detection/protos/pipeline.proto. At a high level, the config
file is split into 5 parts:

1. The `model` configuration. This defines what type of model will be trained
(ie. meta-architecture, feature extractor).
2. The `train_config`, which decides what parameters should be used to train
model parameters (ie. SGD parameters, input preprocessing and feature extractor
initialization values).
3. The `eval_config`, which determines what set of metrics will be reported for
evaluation.
4. The `train_input_config`, which defines what dataset the model should be
trained on.
5. The `eval_input_config`, which defines what dataset the model will be
evaluated on. Typically this should be different than the training input
dataset.

A skeleton configuration file is shown below:

```
model {
(... Add model config here...)
}

train_config : {
(... Add train_config here...)
}

train_input_reader: {
(... Add train_input configuration here...)
}

eval_config: {
}

eval_input_reader: {
(... Add eval_input configuration here...)
}
```
## Picking Model Parameters

There are a large number of model parameters to configure. The best settings
will depend on your given application. Faster R-CNN models are better suited to
cases where high accuracy is desired and latency is of lower priority.
Conversely, if processing time is the most important factor, SSD models are
recommended. Read [our paper](https://arxiv.org/abs/1611.10012) for a more
detailed discussion on the speed vs accuracy tradeoff.

To help new users get started, sample model configurations have been provided
in the object_detection/samples/configs folder. The contents of these
configuration files can be pasted into `model` field of the skeleton
configuration. Users should note that the `num_classes` field should be changed
to a value suited for the dataset the user is training on.

## Defining Inputs

The TensorFlow Object Detection API accepts inputs in the TFRecord file format.
Users must specify the locations of both the training and evaluation files.
Additionally, users should also specify a label map, which define the mapping
between a class id and class name. The label map should be identical between
training and evaluation datasets.

An example input configuration looks as follows:

```
tf_record_input_reader {
  input_path: "/usr/home/username/data/train.record"
}
label_map_path: "/usr/home/username/data/label_map.pbtxt"
```

Users should substitute the `input_path` and `label_map_path` arguments and
insert the input configuration into the `train_input_reader` and
`eval_input_reader` fields in the skeleton configuration.

# Model Training
After stetting up your dataset and model configuration file, you can run below command to start the training:
```bash
python3 train.py --logtostderr --train_dir=/path/to_output_model/ --pipeline_config_path=/path_to_training_config_.json

```
## Running Tensorboard for Training Progress

Progress for training and eval jobs can be inspected using Tensorboard. If using
the recommended directory structure, Tensorboard can be run using the following
command:

```bash
tensorboard --logdir=${MODEL_DIR}
```

where `${MODEL_DIR}` points to the directory that contains the train and eval
directories. Please note it may take Tensorboard a couple minutes to populate
with data.

# Model Deployment 
After your model has been trained, you should export it to a TensorFlow graph proto (Frozen Model) for inference or OpenVINO optimization deployment.
```bash
python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path training-config/ssdMv2-512/pipeline.config --trained_checkpoint_prefix training-config/ssdMv2-512/model.ckpt-300000 --output_directory .
```

# Model Inference
You can use below notebook on colab or locally for model testing and detection visualization :
https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb