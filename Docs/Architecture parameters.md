## Overview

Neural network architecture for XNN needs to be specified in parameter blocks, one block per layer. Each parameter is specified by parameter name, followed by `:` and then parameter value (with optional spaces in between). Casing is important for parameter names but it is ignored for parameter values. Block starts with `layer:` parameter and ends with a blank line, everything before and after is ignored until next block is found. Additionally, you can write comments inside the blocks by starting the line with `//`, and those lines will be ignored too.

Multi GPU support is enabled via parameters `tierSize` and `parallelism`. Each layer in network constitutes a _tier_ where inside a tier there can be multiple instances of same layer, working on different GPUs. `tierSize` parameter simply defines how many layer instances there are in a tier, and `parallelism` parameter defines are they going to share weights and work on different data (data parallelism), or share data and train different weights (model parallelism). `prevLayers` parameter defines if each layer instance in tier will be connected to all layer instances in previous tier, or only to layer instance with the same index in previous tier (in case when previous tier has only one layer instance then all layer instances in current tier will be connected to that one).

You can find example architecture files in [Example architectures](/Example%20architectures) folder.

These are the neural network layers supported by XNN and their supported parameters. To see details about these layers and what their parameters are for, check [Supported layers](Supported%20layers.md) document.

- [Input layer](#input-layer)
- [Convolutional layer](#convolutional-layer)
- [Response Normalization layer](#response-normalization-layer)
- [Max Pool layer](#max-pool-layer)
- [Standard layer](#standard-layer)
- [Dropout layer](#dropout-layer)
- [SoftMax layer](#softmax-layer)
- [Output layer](#output-layer)

## Input layer

`layer: input`

**Parameters:**
- `dataType` Value can be `image` or `text`.
- `numChannels` Positive integer value. Required only if `dataType` is `image`.
- `originalDataWidth` Positive integer value. Optional, defaults to `inputDataWidth`.
- `originalDataHeight` Positive integer value. Optional, defaults to `inputDataHeight`.
- `inputDataWidth` Positive integer value.
- `inputDataHeight` Positive integer value. Required only if `dataType` is `image`.
- `doRandomFlips` Value can be `yes` or `no`. Optional, default value is `no`.
- `normalizeInputs` Value can be `yes` or `no`. Optional, default value is `no`.
- `inputMeans` List of positive integer values, separated by `,` character. List can have `numChannels` values, or one value which will be applied to each channel. Required only if `normalizeInputs` is `yes`.
- `inputStDevs` List of positive integer values, separated by `,` character. List can have `numChannels` values, or one value which will be applied to each channel. Required only if `normalizeInputs` is `yes`.
- `numTestPatches` Positive integer value. Optional, default value is `1`.
- `testOnFlips` Value can be `yes` or `no`. Optional, default value is `no`.

## Convolutional layer

`layer: convolutional`

**Parameters:**
- `tierSize` Positive integer value. Optional, default value is `1`.
- `parallelism` Value can be `data` or `model`. Optional, default value is `model`.
- `prevLayers` Value can be `all`. Optional, if not present layer instances in this tier will be connected only to layer instance with the same index in previous tier (in case when previous tier has only one layer instance then all instances in this tier will be connected to that one).
- `numFilters` Positive integer value.
- `filterWidth` Positive integer value.
- `filterHeight` Positive integer value.
- `paddingX` Positive integer value, including zero. Optional, default value is `0`.
- `paddingY` Positive integer value, including zero. Optional, default value is `0`.
- `stride` Positive integer value. Optional, default value is `1`.
- `weightsInitialization` Value can be `constant`, `normal`, `gaussian` (treated same as `normal`), `uniform`, `xavier` or `he`. Optional, default value is `constant`.
- `weightsInitialValue` Float value. Required only if `weightsInitialization` is `constant`.
- `weightsMean` Float value. Required only if `weightsInitialization` is `normal` or `gaussian`.
- `weightsStdDev` Float value. Required only if `weightsInitialization` is `normal` or `gaussian`.
- `weightsRangeStart` Float value. Required only if `weightsInitialization` is `uniform`.
- `weightsRangeEnd` Float value. Required only if `weightsInitialization` is `uniform`.
- `biasesInitialization` Value can be `constant`, `normal`, `gaussian` (treated same as `normal`), `uniform`. Optional, default value is `constant`.
- `biasesInitialValue` Float value. Required only if `biasesInitialization` is `constant`.
- `biasesMean` Float value. Required only if `biasesInitialization` is `normal` or `gaussian`.
- `biasesStdDev` Float value. Required only if `biasesInitialization` is `normal` or `gaussian`.
- `biasesRangeStart` Float value. Required only if `biasesInitialization` is `uniform`.
- `biasesRangeEnd` Float value. Required only if `biasesInitialization` is `uniform`.
- `weightsMomentum` Float value. Optional, default value is `0`.
- `weightsDecay` Float value. Optional, default value is `0`.
- `weightsStartingLR` Float value.
- `weightsLRStep` Float value.
- `weightsLRFactor` Float value.
- `biasesMomentum` Float value. Optional, default value is `0`.
- `biasesDecay` Float value. Optional, default value is `0`.
- `biasesStartingLR` Float value.
- `biasesLRStep` Float value.
- `biasesLRFactor` Float value.
- `activationType` Value can be `linear`, `relu`, `elu`, `leakyrelu`, `lrelu` (treated same as `leakyrelu`), `sigmoid` or `tanh`.
- `activationAlpha` Float value. Required only if `activationType` is `elu`, `leakyrelu` or `lrelu`.

## Response Normalization layer

`layer: responsenormalization`

**Parameters:**
- `tierSize` Positive integer value. Optional, default value is `1`.
- `parallelism` Value can be `data` or `model`. Optional, default value is `model`.
- `prevLayers` Value can be `all`. Optional, if not present layer instances in this tier will be connected only to layer instance with the same index in previous tier (in case when previous tier has only one layer instance then all instances in this tier will be connected to that one).
- `depth` Positive integer value.
- `bias` Positive integer value, including zero.
- `alphaCoeff` Float value.
- `betaCoeff` Float value.

## Max Pool layer

`layer: maxpool`

**Parameters:**
- `tierSize` Positive integer value. Optional, default value is `1`.
- `parallelism` Value can be `data` or `model`. Optional, default value is `model`.
- `prevLayers` Value can be `all`. Optional, if not present layer instances in this tier will be connected only to layer instance with the same index in previous tier (in case when previous tier has only one layer instance then all instances in this tier will be connected to that one).
- `filterWidth` Positive integer value.
- `filterHeight` Positive integer value.
- `paddingX` Positive integer value, including zero.
- `paddingY` Positive integer value, including zero.
- `stride` Positive integer value.

## Standard layer

`layer: standard`

**Parameters:**
- `tierSize` Positive integer value. Optional, default value is `1`.
- `parallelism` Value can be `data` or `model`. Optional, default value is `model`.
- `prevLayers` Value can be `all`. Optional, if not present layer instances in this tier will be connected only to layer instance with the same index in previous tier (in case when previous tier has only one layer instance then all instances in this tier will be connected to that one).
- `numNeurons` Positive integer value.
- `weightsInitialization` Value can be `constant`, `normal`, `gaussian` (treated same as `normal`), `uniform`, `xavier` or `he`. Optional, default value is `constant`.
- `weightsInitialValue` Float value. Required only if `weightsInitialization` is `constant`.
- `weightsMean` Float value. Required only if `weightsInitialization` is `normal` or `gaussian`.
- `weightsStdDev` Float value. Required only if `weightsInitialization` is `normal` or `gaussian`.
- `weightsRangeStart` Float value. Required only if `weightsInitialization` is `uniform`.
- `weightsRangeEnd` Float value. Required only if `weightsInitialization` is `uniform`.
- `biasesInitialization` Value can be `constant`, `normal`, `gaussian` (treated same as `normal`), `uniform`. Optional, default value is `constant`.
- `biasesInitialValue` Float value. Required only if `biasesInitialization` is `constant`.
- `biasesMean` Float value. Required only if `biasesInitialization` is `normal` or `gaussian`.
- `biasesStdDev` Float value. Required only if `biasesInitialization` is `normal` or `gaussian`.
- `biasesRangeStart` Float value. Required only if `biasesInitialization` is `uniform`.
- `biasesRangeEnd` Float value. Required only if `biasesInitialization` is `uniform`.
- `weightsMomentum` Float value. Optional, default value is `0`.
- `weightsDecay` Float value. Optional, default value is `0`.
- `weightsStartingLR` Float value.
- `weightsLRStep` Float value.
- `weightsLRFactor` Float value.
- `biasesMomentum` Float value. Optional, default value is `0`.
- `biasesDecay` Float value. Optional, default value is `0`.
- `biasesStartingLR` Float value.
- `biasesLRStep` Float value.
- `biasesLRFactor` Float value.
- `activationType` Value can be `linear`, `relu`, `elu`, `leakyrelu`, `lrelu` (treated same as `leakyrelu`), `sigmoid` or `tanh`.
- `activationAlpha` Float value. Required only if `activationType` is `elu`, `leakyrelu` or `lrelu`.

## Dropout layer

`layer: dropout`

**Parameters:**
- `tierSize` Positive integer value. Optional, default value is `1`.
- `parallelism` Value can be `data` or `model`. Optional, default value is `model`.
- `prevLayers` Value can be `all`. Optional, if not present layer instances in this tier will be connected only to layer instance with the same index in previous tier (in case when previous tier has only one layer instance then all instances in this tier will be connected to that one).
- `dropProbability` Float value. Optional, default value is `0.5`.

## SoftMax layer

`layer: softmax`

**Parameters:**
- `prevLayers` Value can be `all`. Optional, if not present layer instances in this tier will be connected only to layer instance with the same index in previous tier (in case when previous tier has only one layer instance then all instances in this tier will be connected to that one).

## Output layer

`layer: output`

**Parameters:**
- `prevLayers` Value can be `all`. Optional, if not present layer instances in this tier will be connected only to layer instance with the same index in previous tier (in case when previous tier has only one layer instance then all instances in this tier will be connected to that one).
- `lossFunction` Value can be `crossentropy` or `logisticregression`.
- `numGuesses` Positive integer value. Optional, default value is `1`.
