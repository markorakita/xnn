# XNN

XNN is an open source machine learning framework for deep neural networks training on multiple GPUs on Windows platform.

This project was started back in 2014 when there were no machine learning frameworks available for Windows platform. It was inspired by cuda-convnet2 project from [Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/).

## Getting Started

After you build the solution, in `XNN/target/{Plaftorm}/{Configuration}/` folder you will have XNN.exe built for your platform and configuration. It is a command line program and in [Command line parameters](Docs/Command%20line%20parameters.md) document you can find full list of commands and parameters to run it.

Desired network architecture needs to be specified in a textual file following semantic rules specified in [Architecture parameters](Docs/Architecture%20parameters.md) document. You can find example architecture files in [Example architectures](/Example%20architectures) folder. In [Supported layers](Docs/Supported%20layers.md) document you can find detailed description of all supported layers and their parameters.

Models trained with XNN can be exported in format suitable for [AXNN](/../../../axnn) to be used for inference on Android mobile devices.
