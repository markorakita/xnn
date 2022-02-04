## Overview

XNN supports several modes of operation:
- [Data preparation](#data-preparation)
- [Model training](#model-training)
- [Model testing](#model-testing)
- [Model training and testing combined](#model-training-and-testing-combined)
- [Features extraction for transfer learning](#features-extraction-for-transfer-learning)
- [Model conversion](#model-conversion)
- [Model analysis](#model-analysis)
- [Running XNN dev tests](#running-xnn-dev-tests)

## Data preparation

This mode prepares data sets for training of various networks.

**Arguments**: `-makedata NetworkArchitectureName`

_NetworkArchitectureName_ should be replaced with the name of the network architecture you need to prepare dataset for. Right now only these architecture names are supported:
- `alexnet` To prepare images from imagenet data set for training and testing of AlexNet model architecture. It expects to find in input folder two folders named _train_ and _test_, and in each of them file named _labels.txt_ containing in each line one image name and label separated by space character.
- `centralcrops` To prepare images for training and testing of various model architectures which only require images to be resized and centrally cropped.

**Additional arguments**:
- `-inputfolder FolderPath` Path to the folder with original data. (It can be omitted if full input data paths are provided through `-inputdatalist` argument)
- `-inputdatalist FilePath` Path to the file containing list of input data paths, which can be relative to the input folder or absolute paths in case `-inputfolder` argument is omitted.
- `-outputfolder FolderPath` Path to the folder where prepared data will be generated.
- `-imagesize S` Output images will be of size _SxS_. (It can be omitted, default value is 256)
- `-numchannels C` Number of channels _C_ in input and output images. (It can be omitted, default value is 3)

**Example command**:
```
XNN.exe -makedata centralcrops -inputfolder "D:\\OriginalData" -inputdatalist "D:\\datalist.txt" -outputfolder "D:\\PreparedData" -imagesize 32 -numchannels 3
```

## Model training

This is a mode for training of neural network model based on a specified network architecture. After each epoch of training, model checkpoint is created so that training could be resumed from it in case of unexpected failures. Number of GPUs to use for training will be determined from network architecture file.

**Arguments**: `-train`

**Additional arguments**:
- `-continue` If this argument is present, model training will be resumed from last saved checkpoint.
- `-configfile FilePath` Path to the file containing desired network architecture to be trained.
- `-traindata FolderPath` Path to the folder containing training data set.
    - In case of image data, it is expected that this folder contains training images and the file named _labels.txt_ containing in each line one image name and its label separated by space character.
    - In case of data features, it is expected that this folder contains file named _trainSet.txt_ containing in each line label and then features of one data, all separated by space characters.
- `-workfolder FolderPath` Path to the folder where trained model and checkpoints will be saved.
- `-numepochs E` Number of epochs to train model for.
- `-batchsize B` Batch size to use.
- `-gpu G` Zero based index of GPU to use for training operations. (If omitted default GPU or GPUs will be used)

**Example command**:
```
XNN.exe -train -configfile "D:\\AlexNet-1gpu.xnn" -traindata "D:\\Data\\train" -workfolder "D:\\WorkFolder" -numepochs 75 -batchsize 128
```

## Model testing

This is a mode for testing accuracy of trained neural network model. Testing always runs on a single GPU.

**Arguments**: `-test`

**Additional arguments**:
- `-configfile FilePath` Path to the file containing network architecture of trained model.
- `-modelfile FilePath` Path to the trained model file.
- `-testdata FolderPath` Path to the folder containing test data set.
    - In case of image data, it is expected that this folder contains test images and the file named _labels.txt_ containing in each line one image name and its label separated by space character.
    - In case of data features, it is expected that this folder contains file named _testSet.txt_ containing in each line label and then features of one data, all separated by space characters.
- `-workfolder FolderPath` Path to the folder where test results will be saved.
- `-batchsize B` Batch size to use.
- `-gpu G` Zero based index of GPU to use for testing operations. (If omitted default GPU will be used)

**Example command**:
```
XNN.exe -test -configfile "D:\\AlexNet-1gpu.xnn" -modelfile "D:\\alexnet-model.xnnm" -testdata "D:\\Data\\test" -workfolder "D:\\WorkFolder" -batchsize 128 -gpu 3
```

## Model training and testing combined

This is a mode for training of neural network model based on a specified network architecture, and then immediately afterwards testing accuracy of that trained neural network model. It simply combines previous two modes. After each epoch of training, model checkpoint is created so that training could be resumed from it in case of unexpected failures. Number of GPUs to use for training will be determined from network architecture file. Testing always runs on a single GPU.

**Arguments**: `-traintest`

**Additional arguments**:
- `-continue` If this argument is present, model training will be resumed from last saved checkpoint.
- `-configfile FilePath` Path to the file containing desired network architecture to be trained.
- `-traindata FolderPath` Path to the folder containing training data set.
    - In case of image data, it is expected that this folder contains training images and the file named _labels.txt_ containing in each line one image name and its label separated by space character.
    - In case of data features, it is expected that this folder contains file named _trainSet.txt_ containing in each line label and then features of one data, all separated by space characters.
- `-testdata FolderPath` Path to the folder containing test data set.
    - In case of image data, it is expected that this folder contains test images and the file named _labels.txt_ containing in each line one image name and its label separated by space character.
    - In case of data features, it is expected that this folder contains file named _testSet.txt_ containing in each line label and then features of one data, all separated by space characters.
- `-workfolder FolderPath` Path to the folder where trained model and checkpoints will be saved.
- `-numepochs E` Number of epochs to train model for.
- `-batchsize B` Batch size to use for training, testing will use double of this batch size.
- `-gpu G` Zero based index of GPU to use for training and testing operations. (If omitted default GPU or GPUs will be used)

**Example command**:
```
XNN.exe -traintest -configfile "D:\\AlexNet-1gpu.xnn" -traindata "D:\\Data\\train" -testdata "D:\\Data\\test" -workfolder "D:\\WorkFolder" -numepochs 75 -batchsize 128
```

## Features extraction for transfer learning

This mode extracts features from data set using trained network model. It runs model inference on the data set and dumps output activations from specified network layer. This is mainly used for transfer learning where you use large pretrained network to extract features and then you train a smaller network to classify those features.

**Arguments**: `-featurize`

**Additional arguments**:
- `-inputfolder FolderPath` Path to the folder with data set.
- `-inputdatalist FilePath` Path to the file containing list of input data paths relative to the input folder.
- `-configfile FilePath` Path to the file containing network architecture of trained model.
- `-modelfile FilePath` Path to the trained model file.
- `-layer L` Zero based index of network layer to extract features from.
- `-batchsize B` Batch size to use.

**Example command**:
```
XNN.exe -featurize -inputfolder "D:\\Data" -inputdatalist "D:\\Data\\datalist.txt" -configfile "D:\\AlexNet-1gpu.xnn" -modelfile "D:\\alexnet-model.xnnm" -layer 11 -batchsize 128
```

## Model conversion

This mode converts trained network model to various supported formats.

**Arguments**: `-convert FormatName`

_FormatName_ should be replaced with the name of the desired output format. Right now only these format names are supported:
- `bigendian` Converts model weights to big endian format so that model can be used on big endian architectures.
- `axnn` Converts model to format suitable for [AXNN](/../../../axnn) to be used for inference on Android mobile devices.

**Additional arguments**:
- `-inmodelfile FilePath` Path to the trained model file to convert.
- `-configfile FilePath` Path to the file containing network architecture of trained model.
- `-outmodelfile FilePath` Path to the converted output model file.

**Example command**:
```
XNN.exe -convert axnn -inmodelfile "D:\\alexnet-model.xnnm" -configfile "D:\\AlexNet-1gpu.xnn" -outmodelfile "D:\\alexnet-axnn-model.axnnm"
```

## Model analysis

This is a mode for analyzing characteristics of the trained model file. It goes through model layers and for each layer outputs characteristics like minimal weight value, average weight value, maximal weight value, etc.

**Arguments**: `-explore`

**Additional arguments**:
- `-configfile FilePath` Path to the file containing network architecture of trained model.
- `-modelfile FilePath` Path to the trained model file to analyze.

**Example command**:
```
XNN.exe -explore -configfile "D:\\AlexNet-1gpu.xnn" -modelfile "D:\\alexnet-model.xnnm"
```

## Running XNN dev tests

This is a mode for developers working on XNN. It runs all or specific solution tests to verify that code changes are not breaking any XNN functionality. It is also useful to run these tests to verify that XNN is properly compiled and working on your machine.

**Arguments**: `-runtests`

**Additional arguments**:
- `-component Name` Name of the component whose tests to run. If omitted and `-testname` argument is also omitted, then all solution tests will be run.
- `-testname Name` Name of the specific component test to run. If omitted and `-component` argument is present, then all solution tests for the specified component will be run.
- `-outputfolder FolderPath` Path to the folder to be used for tests output. This is needed only for some tests, for components that manipulate image files.

**Example command**:
```
XNN.exe -runtests -component softmaxlayer -testname doforwardprop
```
