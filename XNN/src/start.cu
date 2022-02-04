// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Program for training neural networks.
// Created: 11/24/2015.
// ----------------------------------------------------------------------------------------------------

#include <iostream>

#include <cuda_runtime.h>

#include "tests/include/testsdriver.cuh"
#include "tools/include/datamaker.cuh"
#include "tools/include/featurizer.cuh"
#include "tools/include/modelconverter.cuh"
#include "tools/include/modelexplorer.cuh"
#include "tools/include/trainer.cuh"
#include "utils/include/cudaasserts.cuh"
#include "utils/include/consolehelper.cuh"
#include "utils/include/utils.cuh"

const string c_argMakeData = "-makedata";
const string c_argMakeDataForAlexNet = "alexnet";
const string c_argMakeDataCentralCrops = "centralcrops";
const string c_argTrain = "-train";
const string c_argTest = "-test";
const string c_argTrainWithTest = "-traintest";
const string c_argFeaturize = "-featurize";
const string c_argConvert = "-convert";
const string c_argConvertToBigEndian = "bigendian";
const string c_argConvertForAXNN = "axnn";
const string c_argExploreModel = "-explore";
const string c_argRunXnnTests = "-runtests";

/*
    Prints usage of this program.
*/
void PrintUsage()
{
    cout << endl;
    cout << "----------------------------------------------------------------------------------------------------" << endl;
    cout << "Usage:" << endl;
    cout << "----------------------------------------------------------------------------------------------------" << endl;
    cout << "    " << c_argMakeData << " NetworkArchitectureName    (prepares data for training of various network architectures, for example alexnet.)" << endl;
    cout << "        " << DataMaker::c_inputFolderSignature << " \"...\"    (folder with original data)" << endl;
    cout << "        " << DataMaker::c_inputDataListSignature << " \"...\"    (path to the list of input data names, or full paths if input folder is omitted)" << endl;
    cout << "        " << DataMaker::c_outputFolderSignature << " \"...\"    (folder in which to output prepared data)" << endl;
    cout << "        " << DataMaker::c_imageSizeSignature << " S    (output images will be size of SxS, default value: " << DataMaker::c_defaultImageSize << ")" << endl;
    cout << "        " << DataMaker::c_numImageChannelsSignature << " C    (input and output images will have C channels, default value: " << DataMaker::c_defaultNumOfImageChannels << ")" << endl;
    cout << endl;
    cout << "    " << c_argTrain << "    (runs training of network)" << endl;
    cout << "        " << Trainer::c_loadFromCheckpointSignature << "    (if present, training of model will be continued from saved checkpoint)" << endl;
    cout << "        " << Trainer::c_configurationSignature << " \"...\"    (network configuration file path, in case we are training new model)" << endl;
    cout << "        " << Trainer::c_trainDataFolderSignature << " \"...\"    (folder with data for training)" << endl;
    cout << "        " << Trainer::c_workFolderSignature << " \"...\"    (folder where trained models and checkpoints will be saved)" << endl;
    cout << "        " << Trainer::c_numEpochsSignature << " E    (number of epochs to train will be E)" << endl;
    cout << "        " << Trainer::c_batchSizeSignature << " B    (batch size will be B)" << endl;
    cout << "        " << Trainer::c_defaultGpuSignature << " G    (zero based index of GPU to use by default)" << endl;
    cout << endl;
    cout << "    " << c_argTest << "    (runs testing of network)" << endl;
    cout << "        " << Trainer::c_configurationSignature << " \"...\"    (network configuration file path)" << endl;
    cout << "        " << Trainer::c_trainedModelSignature << " \"...\"    (trained network model path)" << endl;
    cout << "        " << Trainer::c_testDataFolderSignature << " \"...\"    (folder with data for testing)" << endl;
    cout << "        " << Trainer::c_workFolderSignature << " \"...\"    (folder where test results will be saved)" << endl;
    cout << "        " << Trainer::c_batchSizeSignature << " B    (batch size will be B)" << endl;
    cout << "        " << Trainer::c_defaultGpuSignature << " G    (zero based index of GPU to use by default)" << endl;
    cout << endl;
    cout << "    " << c_argTrainWithTest << "    (runs training and testing of network)" << endl;
    cout << "        (all arguments for -train and -test combined)" << endl;
    cout << endl;
    cout << "    " << c_argFeaturize << "    (runs features extraction from data set)" << endl;
    cout << "        " << Featurizer::c_inputFolderSignature << " \"...\"    (folder with input data set)" << endl;
    cout << "        " << Featurizer::c_inputDataNamesListSignature << " \"...\"    (path to the list of input data names)" << endl;
    cout << "        " << Featurizer::c_configurationSignature << " \"...\"    (network configuration file path)" << endl;
    cout << "        " << Featurizer::c_modelFileSignature << " \"...\"    (trained network model file path)" << endl;
    cout << "        " << Featurizer::c_targetLayerSignature << " L    (zero based index of network layer to extract features from)" << endl;
    cout << "        " << Featurizer::c_batchSizeSignature << " B    (batch size will be N)" << endl;
    cout << endl;
    cout << "    " << c_argConvert << " FormatName    (converts trained model to specified format)" << endl;
    cout << "        " << ModelConverter::c_trainedModelSignature << " \"...\"    (trained model path)" << endl;
    cout << "        " << ModelConverter::c_configurationSignature << " \"...\"    (network configuration file path)" << endl;
    cout << "        " << ModelConverter::c_outputModelSignature << " \"...\"    (converted model path)" << endl;
    cout << endl;
    cout << "    " << c_argExploreModel << "    (runs trained model exploration to analyze model characteristics)" << endl;
    cout << "        " << ModelExplorer::c_configurationSignature << " \"...\"    (network configuration file path)" << endl;
    cout << "        " << ModelExplorer::c_trainedModelSignature << " \"...\"    (trained network model path)" << endl;
    cout << "        " << ModelExplorer::c_isModelFromCheckpointSignature << "    (if present, it means model file is from saved checkpoint during training)" << endl;
    cout << endl;
    cout << "    " << c_argRunXnnTests << "    (runs tests for solution, if no other argument specified then it runs all tests)" << endl;
    cout << "        " << TestsDriver::c_outputFolderSignature << " \"...\"    (folder for test output, if not specified some tests might not be run!)" << endl;
    cout << "        " << TestsDriver::c_componentToRunSignature << " ComponentName    (runs all tests in component with name equals to ComponentName)" << endl;
    cout << "        " << TestsDriver::c_testToRunSignature << " TestName    (runs only tests with name equal to TestName, and only if they are in component if component is specified)" << endl;
    cout << "----------------------------------------------------------------------------------------------------" << endl << endl << endl;
}

int PrintUsageForBadArguments()
{
    cout << endl << "Bad arguments format." << endl << endl;
    PrintUsage();
    return 1;
}

/*
    Main function.
*/
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        PrintUsage();
        return 0;
    }

    string firstArg = argv[1];

    if (firstArg == c_argMakeData)
    {
        string makeDataArg;
        if (!ParseArgument(argc, argv, c_argMakeData, makeDataArg))
        {
            return PrintUsageForBadArguments();
        }

        DataMaker dataMaker;
        if (dataMaker.ParseArguments(argc, argv))
        {
            if (makeDataArg == c_argMakeDataForAlexNet)
            {
                dataMaker.MakeDataForAlexNet();
            }
            else if (makeDataArg == c_argMakeDataCentralCrops)
            {
                dataMaker.MakeDataCentralCrops();
            }
            else
            {
                return PrintUsageForBadArguments();
            }
        }
        else
        {
            return PrintUsageForBadArguments();
        }
    }
    else if (firstArg == c_argTrain || firstArg == c_argTest || firstArg == c_argTrainWithTest)
    {
        Trainer trainer;
        if (trainer.ParseArguments(argc, argv))
        {
            if (firstArg == c_argTrain)
            {
                trainer.RunTraining();
            }
            else if (firstArg == c_argTest)
            {
                trainer.RunTesting();
            }
            else
            {
                trainer.RunTrainingWithTesting();
            }
        }
        else
        {
            return PrintUsageForBadArguments();
        }
    }
    else if (firstArg == c_argFeaturize)
    {
        Featurizer featurizer;
        if (featurizer.ParseArguments(argc, argv))
        {
            featurizer.RunExtraction();
        }
        else
        {
            return PrintUsageForBadArguments();
        }
    }
    else if (firstArg == c_argConvert)
    {
        string convertArg;
        if (!ParseArgument(argc, argv, c_argConvert, convertArg))
        {
            return PrintUsageForBadArguments();
        }

        ModelConverter modelConverter;
        if (modelConverter.ParseArguments(argc, argv))
        {
            if (convertArg == c_argConvertToBigEndian)
            {
                modelConverter.ConvertModelToBigEndian();
            }
            else if (convertArg == c_argConvertForAXNN)
            {
                modelConverter.ConvertModelForAXNN();
            }
            else
            {
                return PrintUsageForBadArguments();
            }
        }
        else
        {
            return PrintUsageForBadArguments();
        }
    }
    else if (firstArg == c_argExploreModel)
    {
        ModelExplorer modelExplorer;
        if (modelExplorer.ParseArguments(argc, argv))
        {
            modelExplorer.ExploreModel();
        }
        else
        {
            return PrintUsageForBadArguments();
        }
    }
    else if (firstArg == c_argRunXnnTests)
    {
        TestsDriver testsDriver;
        if (testsDriver.ParseArguments(argc, argv))
        {
            testsDriver.RunTests();
        }
        else
        {
            return PrintUsageForBadArguments();
        }
    }
    else
    {
        cout << endl << "Unknown command." << endl << endl;
        PrintUsage();
        return 1;
    }

    CudaAssert(cudaDeviceReset());

    ConsoleHelper::RevertConsoleColors();

    return 0;
}