// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Driver for all tests.
// Created: 12/12/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/testsdriver.cuh"

#include <iostream>

#include "dataparsers/image/include/testjpegdataparser.cuh"
#include "neuralnetwork/include/testactivationfunctions.cuh"
#include "neuralnetwork/include/testconfigurationparser.cuh"
#include "neuralnetwork/layers/include/testconvolutionallayer.cuh"
#include "neuralnetwork/layers/include/testdropoutlayer.cuh"
#include "neuralnetwork/layers/include/testlayer.cuh"
#include "neuralnetwork/layers/include/testmaxpoollayer.cuh"
#include "neuralnetwork/layers/include/testoutputlayer.cuh"
#include "neuralnetwork/layers/include/testresponsenormalizationlayer.cuh"
#include "neuralnetwork/layers/include/testsoftmaxlayer.cuh"
#include "neuralnetwork/layers/include/teststandardlayer.cuh"
#include "tools/include/testtrainer.cuh"
#include "../utils/include/asserts.cuh"
#include "../utils/include/consolehelper.cuh"
#include "../utils/include/utils.cuh"

const string TestsDriver::c_outputFolderSignature = "-outputfolder";
const string TestsDriver::c_componentToRunSignature = "-component";
const string TestsDriver::c_testToRunSignature = "-testname";

void TestsDriver::RegisterTesters()
{
	m_testers["jpegdataparser"] = new TestJpegDataParser(m_inputFolder, m_outputFolder);
	m_testers["configurationparser"] = new TestConfigurationParser();
	m_testers["convolutionallayer"] = new TestConvolutionalLayer();
	m_testers["maxpoollayer"] = new TestMaxPoolLayer();
	m_testers["responsenormalizationlayer"] = new TestResponseNormalizationLayer();
	m_testers["standardlayer"] = new TestStandardLayer();
	m_testers["dropoutlayer"] = new TestDropoutLayer();
	m_testers["softmaxlayer"] = new TestSoftMaxLayer();
	m_testers["outputlayer"] = new TestOutputLayer();
	m_testers["activationfunctions"] = new TestActivationFunctions();
	m_testers["trainer"] = new TestTrainer();
	m_testers["layer"] = new TestLayer();
}

TestsDriver::~TestsDriver()
{
	for (auto tester = m_testers.begin(); tester != m_testers.end(); ++tester)
	{
		delete tester->second;
	}
}

bool TestsDriver::ParseArguments(int argc, char *argv[])
{
	string exePath = (string)argv[0];
	size_t slashPosition = exePath.length();
	for (int i = 0; i < 4; ++i)
	{
		// Backtracking through \target\$(Platform)\$(Configuration)\ structure.
		slashPosition = exePath.find_last_of('\\', slashPosition - 1);
		ShipAssert(slashPosition != string::npos, "Tests can only be run from build target folder!");
	}
	m_inputFolder = exePath.substr(0, slashPosition) + "\\resources\\";

	m_outputFolder = "";
	m_componentToRun = "";
	m_testToRun = "";

	switch (argc)
	{
		case 2:
			return true;
		case 4:
		case 6:
		case 8:
		{
			bool argumentsParsed = ParseArgument(argc, argv, c_outputFolderSignature, m_outputFolder);
			argumentsParsed = ParseArgument(argc, argv, c_componentToRunSignature, m_componentToRun, true) || argumentsParsed;
			argumentsParsed = ParseArgument(argc, argv, c_testToRunSignature, m_testToRun, true) || argumentsParsed;
			return argumentsParsed;
		}
		default:
			return false;
	}
}

void TestsDriver::RunTests()
{
	RegisterTesters();
	
	cout << endl;

	if (m_componentToRun == "" && m_testToRun == "")
	{
		bool allTestsPassed = true;
		for (auto tester = m_testers.begin(); tester != m_testers.end(); ++tester)
		{
			cout << "Running tests from component " << tester->first << endl << endl;
			bool componentTestsPassed = tester->second->RunAllTests();
			allTestsPassed = allTestsPassed && componentTestsPassed;
		}

		if (allTestsPassed)
		{
			ConsoleHelper::SetConsoleForeground(ConsoleColor::GREEN);
			cout << endl << "All tests passed!" << endl << endl;
		}
		else
		{
			ConsoleHelper::SetConsoleForeground(ConsoleColor::RED);
			cout << endl << "Some of the tests failed!" << endl << endl;
		}
	}
	else
	{
		bool noTestsRun = true;

		for (auto tester = m_testers.begin(); tester != m_testers.end(); ++tester)
		{
			if (tester->first == m_componentToRun)
			{
				if (m_testToRun == "")
				{
					cout << "Running tests from component " << tester->first << endl << endl;
					if (tester->second->RunAllTests())
					{
						ConsoleHelper::SetConsoleForeground(ConsoleColor::GREEN);
						cout << endl << "All tests from component " << tester->first << " passed!" << endl << endl;
					}
					else
					{
						ConsoleHelper::SetConsoleForeground(ConsoleColor::RED);
						cout << endl << "Some of the tests from component " << tester->first << " failed!" << endl << endl;
					}
				}
				else
				{
					if (tester->second->HasTest(m_testToRun))
					{
						cout << "Running test " << m_testToRun << " in component " << m_componentToRun << endl << endl;
						tester->second->RunTest(m_testToRun);
					}
				}

				return;
			}
			else if (m_componentToRun == "" && tester->second->HasTest(m_testToRun))
			{
				cout << "Running test " << m_testToRun << " in component " << tester->first << endl << endl;
				tester->second->RunTest(m_testToRun);
				cout << endl;
				noTestsRun = false;
			}
		}

		if (noTestsRun)
		{
			ConsoleHelper::SetConsoleForeground(ConsoleColor::RED);
			cout << "No test has been run!" << endl << endl;
		}
	}
}