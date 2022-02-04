// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Factory for creating data parsers.
// Created: 11/29/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/dataparserfactory.cuh"

#include "image/include/jpegdataparser.cuh"
#include "../utils/include/asserts.cuh"

DataParserType DataParserFactory::GetDataParserType(string dataExtension)
{
	if (dataExtension == "jpg" || dataExtension == "jpeg")
	{
		return DataParserType::JPEG;
	}

	return DataParserType::Unknown;
}

bool DataParserFactory::CreateDataParser(DataParserType dataParserType, DataParser** outDataParser)
{
	if (dataParserType == DataParserType::JPEG)
	{
		*outDataParser = new JpegDataParser();
		return true;
	}

	return false;
}

DataParserFactory::~DataParserFactory()
{
	for (auto it = m_dataParsers.begin(); it != m_dataParsers.end(); ++it)
	{
		delete it->second;
	}
}

DataParser* DataParserFactory::GetDataParser(string dataExtension)
{
	DataParserType dataParserType = GetDataParserType(dataExtension);
	if (dataParserType == DataParserType::Unknown)
	{
		return NULL;
	}

	auto parserIt = m_dataParsers.find(dataParserType);
	if (parserIt == m_dataParsers.end())
	{
		DataParser* newDataParser;
		ShipAssert(CreateDataParser(dataParserType, &newDataParser), "Couldn't create appropriate data parser! (data extension: " + dataExtension + ")");
		m_dataParsers.insert(make_pair(dataParserType, newDataParser));
		return newDataParser;
	}
	else
	{
		return parserIt->second;
	}
}