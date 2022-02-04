// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for configuration parser.
// Created: 10/09/2020.
// ----------------------------------------------------------------------------------------------------

#include "include/testconfigurationparser.cuh"

#include "../../neuralnetwork/include/configurationparser.cuh"

TestConfigurationParser::TestConfigurationParser()
{
    m_tests["testtrimline"] = bind(&TestConfigurationParser::TestTrimLine, this);
    m_tests["testgetparametervaluestrfromline"] = bind(&TestConfigurationParser::TestGetParameterValueStrFromLine, this);
    m_tests["testparseparameteruint"] = bind(&TestConfigurationParser::TestParseParameterUint, this);
    m_tests["testparseparameterfloat"] = bind(&TestConfigurationParser::TestParseParameterFloat, this);
    m_tests["testparseparameterbool"] = bind(&TestConfigurationParser::TestParseParameterBool, this);
    m_tests["testparseparameterstring"] = bind(&TestConfigurationParser::TestParseParameterString, this);
}

bool TestConfigurationParser::TestTrimLine()
{
    ConfigurationParser configurationParser;

    string line = "test trim line";

    return configurationParser.TrimLine("") == "" &&
        configurationParser.TrimLine(line) == line &&
        configurationParser.TrimLine(" " + line) == line &&
        configurationParser.TrimLine("\t" + line) == line &&
        configurationParser.TrimLine(line + " ") == line &&
        configurationParser.TrimLine(line + "\t") == line &&
        configurationParser.TrimLine(" " + line + " ") == line &&
        configurationParser.TrimLine("\t" + line + "\t") == line &&
        configurationParser.TrimLine("  " + line + "    ") == line &&
        configurationParser.TrimLine("\t\t" + line + "\t\t\t") == line &&
        configurationParser.TrimLine("  \t \t  " + line + " \t  \t  ") == line;
}

bool TestConfigurationParser::TestGetParameterValueStrFromLine()
{
    ConfigurationParser configurationParser;

    string parameterName = "dataType";
    string parameterValue = "image";
    string parameterValueStr;

    if (!configurationParser.GetParameterValueStrFromLine(parameterName + ":" + parameterValue, parameterName, parameterValueStr) || parameterValueStr != parameterValue ||
        !configurationParser.GetParameterValueStrFromLine(parameterName + ": " + parameterValue, parameterName, parameterValueStr) || parameterValueStr != parameterValue ||
        !configurationParser.GetParameterValueStrFromLine(parameterName + " :" + parameterValue, parameterName, parameterValueStr) || parameterValueStr != parameterValue ||
        !configurationParser.GetParameterValueStrFromLine(parameterName + " : " + parameterValue, parameterName, parameterValueStr) || parameterValueStr != parameterValue ||
        configurationParser.GetParameterValueStrFromLine("   " + parameterName + ": " + parameterValue, parameterName, parameterValueStr) ||
        configurationParser.GetParameterValueStrFromLine(parameterName + "r: " + parameterValue, parameterName, parameterValueStr))
    {
        return false;
    }

    return true;
}

bool TestConfigurationParser::TestParseParameterUint()
{
    ConfigurationParser configurationParser;

    string parameterName = "numChannels";
    uint parameterValue = 3;
    uint parsedParameterValue;

    return configurationParser.ParseParameterUint(parameterName + ": " + to_string(parameterValue), parameterName, parsedParameterValue) &&
        parsedParameterValue == parameterValue;
}

bool TestConfigurationParser::TestParseParameterFloat()
{
    ConfigurationParser configurationParser;

    string parameterName = "weightsDecay";
    float parameterValue = 0.01f;
    float parsedParameterValue;

    return configurationParser.ParseParameterFloat(parameterName + ": " + to_string(parameterValue), parameterName, parsedParameterValue) &&
        parsedParameterValue == parameterValue;
}

bool TestConfigurationParser::TestParseParameterBool()
{
    ConfigurationParser configurationParser;

    string parameterName = "testOnFlips";
    bool parsedParameterValue;

    return configurationParser.ParseParameterBool(parameterName + ": yes", parameterName, parsedParameterValue) && parsedParameterValue &&
        configurationParser.ParseParameterBool(parameterName + ": no", parameterName, parsedParameterValue) && !parsedParameterValue;
}

bool TestConfigurationParser::TestParseParameterString()
{
    ConfigurationParser configurationParser;

    string parameterName = "dataType";
    string parameterValue = "image";
    string parameterValueUppercase = "Image";
    string parameterValueStr;

    return configurationParser.ParseParameterString(parameterName + ":" + parameterValue, parameterName, parameterValueStr) && parameterValueStr == parameterValue &&
        configurationParser.ParseParameterString(parameterName + ":" + parameterValueUppercase, parameterName, parameterValueStr) && parameterValueStr == parameterValue;
}