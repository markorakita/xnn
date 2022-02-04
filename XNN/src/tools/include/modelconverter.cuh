// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Class for converting trained XNN neural network models to various formats.
// Created: 06/12/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <iosfwd>
#include <string>

#include "../../utils/include/deftypes.cuh"

using namespace std;

class ModelConverter
{
private:
    // Input model path.
    string m_inputModelPath;

    // Network configuration file path.
    string m_networkConfigurationFile;

    // Output model path.
    string m_outputModelPath;

    // Converts convolutional layer weights and biases from model to AXNN format.
    void ConvertConvolutionalLayerForAXNN(ifstream& inputModelStream, uint numFilters, int filterWidth, int filterHeight, int numFilterChannels, ofstream& outputModelStream);

    // Converts standard layer weights and biases from model to AXNN format.
    void ConvertStandardLayerForAXNN(ifstream& inputModelStream, int numNeurons, int inputWidth, int inputHeight, int inputNumChannels, ofstream& outputModelStream);

    // Converts contiguous buffer to big endian format.
    void ConvertContiguousBufferToBigEndian(ifstream& inputModelStream, size_t bufferLength, ofstream& outputModelStream);

public:
    // Parameters signatures.
    static const string c_trainedModelSignature;
    static const string c_configurationSignature;
    static const string c_outputModelSignature;

    // Parses arguments for explorer.
    bool ParseArguments(int argc, char* argv[]);

    // Converts model to big endian format.
    void ConvertModelToBigEndian();

    // Converts model to AXNN format.
    void ConvertModelForAXNN();
};