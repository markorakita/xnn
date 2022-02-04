## Overview

These are the neural network layers supported by XNN. To see how to specify these layers and their parameters in the network architecture file, check [Architecture parameters](Architecture%20parameters.md) document.

- [Input layer](#input-layer)
- [Convolutional layer](#convolutional-layer)
- [Response Normalization layer](#response-normalization-layer)
- [Max Pool layer](#max-pool-layer)
- [Standard layer](#standard-layer)
- [Dropout layer](#dropout-layer)
- [SoftMax layer](#softmax-layer)
- [Output layer](#output-layer)

Additionally, for layers with weights such as Convolutional and Standard layers, here is more about some common parameters:
- [Weights initialization options](#weights-initialization-options)
- [Weights update parameters](#weights-update-parameters)
- [Activation functions](#activation-functions)

## Input layer

Input layer is required as a first layer in the network. Its purpose is to load input data from disk, resize and normalize it if necessary, and serve it to the network. Data loading from disk is done in parallel with network propagation in order to save time.

**Parameters:**
- `dataType` Type of input data. Right now only image data and data features in a textual file are supported.
- `numChannels` Number of input data channels.
- `originalDataWidth` Width of the original data from disk.
- `originalDataHeight` Height of the original data from disk.
- `inputDataWidth` Width of the final input data. If it is different from `originalDataWidth`, data will be randomly cropped.
- `inputDataHeight` Height of the final input data. If it is different from `originalDataHeight`, data will be randomly cropped.
- `doRandomFlips` Should input data be randomly flipped.
- `normalizeInputs` Should input data be normalized to specified mean and standard deviation.
- `inputMeans` List of means to which to normalize each channel of input data.
- `inputStDevs` List of standard deviations to which to normalize each channel of input data.
- `numTestPatches` Number of test data patches to take for generating test predictions. Final prediction will be average of predictions on each of the patches.
- `testOnFlips` Should flips of test data patches also be included into generation of test predictions.

## Convolutional layer

Convolutional layer is the core building block of a CNN. The layers parameters consist of a set of learnable filters (or kernels), which have a small receptive field, but extend through the full depth of the input volume. During the forward pass, each filter is convolved across the width and height of the input volume, computing the dot product between the entries of the filter and the input and producing a 2-dimensional activation map of that filter. As a result, the network learns filters that activate when they see some specific type of feature at some spatial position in the input.

**Parameters:**
- `numFilters` Number of filters.
- `filterWidth` Filter width.
- `filterHeight` Filter height.
- `paddingX` Horizontal padding to apply to input.
- `paddingY` Vertical padding to apply to input.
- `stride` Stride controls for how much should filters shift through the input during fordard propagation.
- [Weights initialization parameters](#weights-initialization-options)
- [Weights update parameters](#weights-update-parameters)
- [Activation function parameters](#activation-functions)

## Response Normalization layer

Response normalization layer implements a form of lateral inhibition inspired by the type found in real neurons, creating competition for big activities amongst neuron outputs computed using different kernels. For details, see [Krizhevsky et al., ImageNet classification with deep convolutional neural networks (NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

It is implemented by a following formula:

<img src="https://render.githubusercontent.com/render/math?math=\displaystyle A_{i} = P_{i} / (bias %2B \frac{alphaCoeff}{depth}(\sum_{j=max(0,i-depth/2)}^{min(N-1,i%2Bdepth/2)}P_{j}^2))^{betaCoeff}">

Where `A[i]` is activation with index `i`, `P[i]` is preactivation with index `i`, and `N` is number of channels.

**Parameters:**
- `depth` Depth of normalization.
- `bias` Normalization bias.
- `alphaCoeff` Normalization alpha coefficient (see the formula above).
- `betaCoeff` Normalization beta coefficient (see the formula above).

## Max Pool layer

Max pool layer partitions the input image into a set of regions, and for each such region outputs the maximum value of input activity. It helps to reduce dimensionality, but also teaches model to be more invariant to translation.

**Parameters:**
- `filterWidth` Pooling region width.
- `filterHeight` Pooling region height.
- `paddingX` Horizontal padding to apply to input.
- `paddingY` Vertical padding to apply to input.
- `stride` Stride controls for how much should pooling region shift through the input during fordard propagation.

## Standard layer

Standard fully connected neural network layer.

**Parameters:**
- `numNeurons` Number of neurons.
- [Weights initialization parameters](#weights-initialization-options)
- [Weights update parameters](#weights-update-parameters)
- [Activation function parameters](#activation-functions)

## Dropout layer

Dropout layer provides efficient way to simulate combining multiple trained models to reduce test error and prevent overfitting. It works by dropping each neuron activity with certain probability, preventing complex coadaptations between neurons.

**Parameters:**
- `dropProbability` Probability to drop some neuron activation.

## SoftMax layer

Soft Max layer calculates soft maximums of input activations, so they sum to 1 and can be used as probabilities of prediction.

This layer has no additional parameters.

## Output layer

Output layer is required as a last layer in the network. It calculates training loss and training/testing accuracy.

These loss functions are supported:
- **Logistic regression** - This loss function should be used for binary classification. When it is used it is expected to have before this layer one Standard layer with linear activation function and one neuron, in order to produce single activation on which to calculate logistic regression loss and accuracy.
- **Cross entropy** - This loss function should be used for classification when there are more than 2 classes. When it is used it is expected to have before this layer one SoftMax layer, and before it one Standard layer with linear activation function and number of neurons equal to number of classes.

**Parameters:**
- `lossFunction` Loss function to use.
- `numGuesses` Number of guesses K network is allowed to make when calculating top-K accuracy.

## Weights initialization options

These weights/biases initialization options and their parameters are supported by all layers with weights such as Convolutional and Standard layers:
- **Constant initialization** - Initializes all weights/biases to constant value specified by parameters `weightsInitialValue` and `biasesInitialValue`.
- **Normal (Gaussian) initialization** - Initializes all weights/biases to values taken from Normal (Gaussian) distribution with mean and standard deviation specified by parameters `weightsMean`, `weightsStdDev`, `biasesMean` and `biasesStdDev`.
- **Uniform initialization** - Initializes all weights/biases to values taken from Uniform distribution in range specified by parameters `weightsRangeStart`, `weightsRangeEnd`, `biasesRangeStart` and `biasesRangeEnd`.
- **Xavier initialization** - Initializes all weights to values taken from Normal (Gaussian) distribution with mean `0` and standard deviation calculated as `sqrt(6.0 / (NumberOfActivationsInThisLayer + NumberOfActivationsInPreviousLayer))`.
- **He initialization** - Initializes all weights to values taken from Normal (Gaussian) distribution with mean `0` and standard deviation calculated as `sqrt((ActivationType == ReLU ? 2.0 : 1.0) / NumberOfActivationsInPreviousLayer)`.

## Weights update parameters

These parameters are supported by all layers with weights such as Convolutional and Standard layers and they control the weights/biases update after each backpropagation pass:
- `weightsMomentum` Momentum to apply to weights updates.
- `weightsDecay` Decay to apply to weights updates.
- `weightsStartingLR` Weights updates starting learning rate.
- `weightsLRStep` Fraction of epochs after which to multiply weights learning rate with `weightsLRFactor`.
- `weightsLRFactor` Factor with which to multiply weights learning rate after specified fraction of epochs.
- `biasesMomentum` Momentum to apply to biases updates.
- `biasesDecay` Decay to apply to biases updates.
- `biasesStartingLR` Biases updates starting learning rate.
- `biasesLRStep` Fraction of epochs after which to multiply biases learning rate with `weightsLRFactor`.
- `biasesLRFactor` Factor with which to multiply biases learning rate after specified fraction of epochs.

## Activation functions

These activation functions and their parameters are supported by all layers with weights such as Convolutional and Standard layers:
- **Linear** - Applies linear activation, i.e. just passes through the preactivation values.
- **Sigmoid** - Applies [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) activation.
- **Tanh** - Applies [tanh](https://en.wikipedia.org/wiki/Hyperbolic_functions) (hyperbolic tangent) activation.
- **ReLU** - Applies [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation.
- **ELU** - Applies [ELU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation. `activationAlpha` parameter gets multiplied with preactivation in case preactivation is smaller than `0`.
- **LeakyReLU** - Applies [LeakyReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation. `activationAlpha` parameter gets multiplied with `exp(Preactivation) - 1`, in case preactivation is smaller than `0`.
