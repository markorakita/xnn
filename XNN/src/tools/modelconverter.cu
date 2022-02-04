// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Class for converting trained XNN neural network models to various formats.
// Created: 06/12/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/modelconverter.cuh"

#include <fstream>

#include "../neuralnetwork/include/configurationparser.cuh"
#include "../neuralnetwork/include/neuralnet.cuh"
#include "../neuralnetwork/layers/include/convolutionallayer.cuh"
#include "../neuralnetwork/layers/include/standardlayer.cuh"
#include "../utils/include/asserts.cuh"
#include "../utils/include/utils.cuh"

const string ModelConverter::c_trainedModelSignature = "-inmodelfile";
const string ModelConverter::c_configurationSignature = "-configfile";
const string ModelConverter::c_outputModelSignature = "-outmodelfile";

bool ModelConverter::ParseArguments(int argc, char* argv[])
{
    if (!ParseArgument(argc, argv, c_trainedModelSignature, m_inputModelPath) ||
        !ParseArgument(argc, argv, c_outputModelSignature, m_outputModelPath))
    {
        return false;
    }

    m_networkConfigurationFile = "";
    ParseArgument(argc, argv, c_configurationSignature, m_networkConfigurationFile);

    return true;
}

void ModelConverter::ConvertModelToBigEndian()
{
    ifstream inputModelStream(m_inputModelPath, ios::binary | ios::ate);
    size_t bufferSize = inputModelStream.tellg();
    inputModelStream.seekg(0, ios::beg);

    ofstream outputModelStream = ofstream(m_outputModelPath, ios::binary);

    ConvertContiguousBufferToBigEndian(inputModelStream, bufferSize, outputModelStream);

    inputModelStream.close();
    outputModelStream.close();
}

void ModelConverter::ConvertModelForAXNN()
{
    ShipAssert(m_networkConfigurationFile != "", "Missing network configuration file path!");

    ConfigurationParser configurationParser;
    NeuralNet* neuralNet = configurationParser.ParseNetworkFromConfiguration(ParsingMode::Prediction, m_networkConfigurationFile, "", 1, false);

    ifstream inputModelStream = ifstream(m_inputModelPath, ios::binary);
    ofstream outputModelStream = ofstream(m_outputModelPath, ios::binary);

    bool prevLayerConv = false;
    for (const vector<Layer*>& layerTier : neuralNet->GetLayerTiers())
    {
        int tierSize = layerTier[0]->GetParallelismMode() == ParallelismMode::Data ? 1 : (int)layerTier.size();

        if (layerTier[0]->GetLayerType() == LayerType::Convolutional)
        {
            ConvolutionalLayer* convLayer = static_cast<ConvolutionalLayer*>(layerTier[0]);

            for (int i = 0; i < tierSize; ++i)
            {
                ConvertConvolutionalLayerForAXNN(inputModelStream, convLayer->GetNumberOfFilters(), convLayer->GetFilterWidth(), convLayer->GetFilterHeight(),
                    convLayer->GetNumberOfFilterChannels(), outputModelStream);
            }

            prevLayerConv = true;
        }
        else if (layerTier[0]->GetLayerType() == LayerType::Standard)
        {
            StandardLayer* standardLayer = static_cast<StandardLayer*>(layerTier[0]);

            if (prevLayerConv)
            {
                ConvertStandardLayerForAXNN(inputModelStream, standardLayer->GetNumberOfNeurons() * tierSize, standardLayer->GetInputDataWidth(),
                    standardLayer->GetInputDataHeight(), standardLayer->GetInputNumChannels(), outputModelStream);
            }
            else
            {
                ConvertContiguousBufferToBigEndian(inputModelStream, tierSize * (standardLayer->GetWeightsBufferSize() + standardLayer->GetBiasesBufferSize()), outputModelStream);
            }
        }
    }

    inputModelStream.close();
    outputModelStream.close();

    delete neuralNet;
}

void ModelConverter::ConvertConvolutionalLayerForAXNN(ifstream& inputModelStream, uint numFilters, int filterWidth, int filterHeight, int numFilterChannels, ofstream& outputModelStream)
{
    // Converting weights.
    size_t weightsBufferSize = (size_t)numFilters * filterWidth * filterHeight * numFilterChannels * sizeof(float);
    if (weightsBufferSize < sizeof(float))
    {
        return;
    }

    uchar* weightsBuffer = (uchar*)malloc(weightsBufferSize);
    ShipAssert(weightsBuffer != NULL, "Can't allocate necessary buffers!");
    uchar* convertedWeightsBuffer = (uchar*)malloc(weightsBufferSize);
    ShipAssert(convertedWeightsBuffer != NULL, "Can't allocate necessary buffers!");

    inputModelStream.read(reinterpret_cast<char*>(weightsBuffer), weightsBufferSize);
    ShipAssert(!inputModelStream.eof(), "Unexpectedly short size of input model!");

    for (size_t filterIndex = 0; filterIndex < numFilters; ++filterIndex)
    {
        size_t filterOffset = filterIndex * filterHeight * filterWidth * numFilterChannels;
        for (size_t filterY = 0; filterY < filterHeight; ++filterY)
        {
            size_t filterYOffset = filterOffset + filterY * filterWidth * numFilterChannels;
            for (size_t filterX = 0; filterX < filterWidth; ++filterX)
            {
                size_t filterXOffset = filterYOffset + filterX * numFilterChannels;
                for (size_t channelIndex = 0; channelIndex < numFilterChannels; ++channelIndex)
                {
                    size_t channelOffset = (filterXOffset + channelIndex) * sizeof(float);
                    size_t inputChannelOffset = (channelIndex * filterHeight * filterWidth * numFilters + filterY * filterWidth * numFilters +
                        filterX * numFilters + filterIndex) * sizeof(float);
                    for (size_t i = 0; i < sizeof(float); ++i)
                    {
                        convertedWeightsBuffer[channelOffset + i] = weightsBuffer[inputChannelOffset + sizeof(float) - 1 - i];
                    }
                }
            }
        }
    }

    outputModelStream.write(reinterpret_cast<const char*>(convertedWeightsBuffer), weightsBufferSize);

    free(weightsBuffer);
    free(convertedWeightsBuffer);

    // Converting biases.
    ConvertContiguousBufferToBigEndian(inputModelStream, numFilters * sizeof(float), outputModelStream);
}

void ModelConverter::ConvertStandardLayerForAXNN(ifstream& inputModelStream, int numNeurons, int inputWidth, int inputHeight, int inputNumChannels, ofstream& outputModelStream)
{
    // Converting weights.
    size_t weightsBufferSize = (size_t)numNeurons * inputWidth * inputHeight * inputNumChannels * sizeof(float);
    if (weightsBufferSize < sizeof(float))
    {
        return;
    }

    uchar* weightsBuffer = (uchar*)malloc(weightsBufferSize);
    ShipAssert(weightsBuffer != NULL, "Can't allocate necessary buffers!");
    uchar* convertedWeightsBuffer = (uchar*)malloc(weightsBufferSize);
    ShipAssert(convertedWeightsBuffer != NULL, "Can't allocate necessary buffers!");

    inputModelStream.read(reinterpret_cast<char*>(weightsBuffer), weightsBufferSize);
    ShipAssert(!inputModelStream.eof(), "Unexpectedly short size of input model!");

    for (size_t neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex)
    {
        size_t neuronOffset = neuronIndex * inputWidth * inputHeight * inputNumChannels * sizeof(float);
        for (size_t pixelY = 0; pixelY < inputHeight; ++pixelY)
        {
            size_t pixelYOffset = pixelY * inputWidth * inputNumChannels;
            for (size_t pixelX = 0; pixelX < inputWidth; ++pixelX)
            {
                size_t pixelXOffset = pixelYOffset + pixelX * inputNumChannels;
                for (size_t channelIndex = 0; channelIndex < inputNumChannels; ++channelIndex)
                {
                    size_t channelOffset = (pixelXOffset + channelIndex) * sizeof(float);
                    size_t inputChannelOffset = (channelIndex * inputHeight * inputWidth + pixelY * inputWidth + pixelX) * sizeof(float);
                    for (size_t i = 0; i < sizeof(float); ++i)
                    {
                        convertedWeightsBuffer[neuronOffset + channelOffset + i] = weightsBuffer[neuronOffset + inputChannelOffset + sizeof(float) - 1 - i];
                    }
                }
            }
        }
    }

    outputModelStream.write(reinterpret_cast<const char*>(convertedWeightsBuffer), weightsBufferSize);

    free(weightsBuffer);
    free(convertedWeightsBuffer);

    // Converting biases.
    ConvertContiguousBufferToBigEndian(inputModelStream, numNeurons * sizeof(float), outputModelStream);
}

void ModelConverter::ConvertContiguousBufferToBigEndian(ifstream& inputModelStream, size_t bufferSize, ofstream& outputModelStream)
{
    if (bufferSize < sizeof(float))
    {
        return;
    }

    uchar* buffer = (uchar*)malloc(bufferSize);
    ShipAssert(buffer != NULL, "Can't allocate necessary buffers!");
    uchar* convertedBuffer = (uchar*)malloc(bufferSize);
    ShipAssert(convertedBuffer != NULL, "Can't allocate necessary buffers!");

    inputModelStream.read(reinterpret_cast<char*>(buffer), bufferSize);
    ShipAssert(!inputModelStream.eof(), "Unexpectedly short size of input model!");

    for (size_t bufferOffset = 0; bufferOffset < bufferSize; bufferOffset += sizeof(float))
    {
        for (size_t i = 0; i < sizeof(float); ++i)
        {
            convertedBuffer[bufferOffset + i] = buffer[bufferOffset + sizeof(float) - 1 - i];
        }
    }

    outputModelStream.write(reinterpret_cast<const char*>(convertedBuffer), bufferSize);

    free(buffer);
    free(convertedBuffer);
}