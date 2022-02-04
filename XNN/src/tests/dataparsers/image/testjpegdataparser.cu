// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for jpeg data parser.
// Created: 12/12/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/testjpegdataparser.cuh"

#include <iostream>
#include <vector>

#include "../../../data/include/imagedata.cuh"
#include "../../../dataparsers/image/include/jpegdataparser.cuh"
#include "../../../utils/include/consolehelper.cuh"
#include "../../../utils/include/utils.cuh"

TestJpegDataParser::TestJpegDataParser(string inputFolder, string outputFolder)
{
	m_inputFolder = inputFolder;
	m_outputFolder = outputFolder;

	// Registering tests.
	m_tests["resizeimagecu"] = bind(&TestJpegDataParser::TestResizeImageCu, this);
	m_tests["cropimage"] = bind(&TestJpegDataParser::TestCropImage, this);
	m_tests["resizeimagewithcropcu"] = bind(&TestJpegDataParser::TestResizeImageWithCropCu, this);
}

bool TestJpegDataParser::CheckInputOutputFolders()
{
	if (m_inputFolder == "" || m_outputFolder == "")
	{
		ConsoleHelper::SetConsoleForeground(ConsoleColor::YELLOW);
		if (m_inputFolder == "")
		{
			cout << "No input folder defined, JpegDataParser tests will not be run!" << endl;
		}
		else if (m_outputFolder == "")
		{
			cout << "No output folder defined, JpegDataParser tests will not be run!" << endl;
		}
		ConsoleHelper::RevertConsoleColors();

		return false;
	}

	return true;
}

bool TestJpegDataParser::RunTest(string testName)
{
	if (!CheckInputOutputFolders())
	{
		return false;
	}

	return AbstractTester::RunTest(testName);
}

bool TestJpegDataParser::RunAllTests()
{
	if (!CheckInputOutputFolders())
	{
		return false;
	}

	return AbstractTester::RunAllTests();
}

//******************************************************************************************************
// Tests
//******************************************************************************************************

bool TestJpegDataParser::TestResizeImageCu()
{
	vector<string> testImages;
	testImages.push_back("testImageH.JPEG");
	testImages.push_back("testImageV.JPEG");

	JpegDataParser dataParser;
	const uint imageSize = 224;
	const uint mediumImageSize = 2560;
	const uint largeImageSize = 4800;
	ImageData* image;
	ImageData* resizedImage;
	
	for (string testImage : testImages)
	{
		image = dataParser.LoadImage(m_inputFolder + testImage);
		string imageName = GetFileNameWithoutExtension(testImage);

		// Test resizing to smaller.
		resizedImage = dataParser.ResizeImageCu(*image, imageSize, imageSize, ResizeMode::ResizeToSmaller);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToSmaller.jpg");
		delete resizedImage;
		// Test resizing to larger.
		resizedImage = dataParser.ResizeImageCu(*image, imageSize, imageSize, ResizeMode::ResizeToLarger);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToLarger.jpg");
		delete resizedImage;
		// Test resizing to fit.
		resizedImage = dataParser.ResizeImageCu(*image, imageSize, imageSize, ResizeMode::ResizeToFit);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToFit.jpg");
		delete resizedImage;

		// Test resizing with medium image size.
		resizedImage = dataParser.ResizeImageCu(*image, mediumImageSize, mediumImageSize, ResizeMode::ResizeToLarger);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToLargerMediumSize.jpg");
		delete resizedImage;

		// Test resizing with large image size.
		resizedImage = dataParser.ResizeImageCu(*image, largeImageSize, largeImageSize, ResizeMode::ResizeToLarger);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToLargerLargeSize.jpg");
		delete resizedImage;

		delete image;
	}

	return true;
}

bool TestJpegDataParser::TestCropImage()
{
	vector<string> testImages;
	testImages.push_back("testImageH.JPEG");
	testImages.push_back("testImageV.JPEG");

	JpegDataParser dataParser;
	const uint imageSize = 224;
	ImageData* image;
	ImageData* croppedImage;

	for (string testImage : testImages)
	{
		image = dataParser.LoadImage(m_inputFolder + testImage);
		string imageName = GetFileNameWithoutExtension(testImage);

		// Test crop of upper left corner, no flip.
		croppedImage = dataParser.CropImage(*image, 0, 0, imageSize, imageSize, false);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedUpperLeft.jpg");
		delete croppedImage;
		// Test crop of upper right corner, no flip.
		croppedImage = dataParser.CropImage(*image, image->GetWidth() - imageSize, 0, imageSize, imageSize, false);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedUpperRight.jpg");
		delete croppedImage;
		// Test crop center, no flip.
		croppedImage = dataParser.CropImage(*image, (image->GetWidth() - imageSize) /2, (image->GetHeight() - imageSize) / 2, imageSize, imageSize, false);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedCenter.jpg");
		delete croppedImage;
		// Test crop of lower left corner, no flip.
		croppedImage = dataParser.CropImage(*image, 0, image->GetHeight() - imageSize, imageSize, imageSize, false);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedLowerLeft.jpg");
		delete croppedImage;
		// Test crop of lower right corner, no flip.
		croppedImage = dataParser.CropImage(*image, image->GetWidth() - imageSize, image->GetHeight() - imageSize, imageSize, imageSize, false);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedLowerRight.jpg");
		delete croppedImage;

		// Test crop of upper left corner, with flip.
		croppedImage = dataParser.CropImage(*image, 0, 0, imageSize, imageSize, true);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedUpperLeftFlipped.jpg");
		delete croppedImage;
		// Test crop of upper right corner, with flip.
		croppedImage = dataParser.CropImage(*image, image->GetWidth() - imageSize, 0, imageSize, imageSize, true);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedUpperRightFlipped.jpg");
		delete croppedImage;
		// Test crop center, with flip.
		croppedImage = dataParser.CropImage(*image, (image->GetWidth() - imageSize) / 2, (image->GetHeight() - imageSize) / 2, imageSize, imageSize, true);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedCenterFlipped.jpg");
		delete croppedImage;
		// Test crop of lower left corner, with flip.
		croppedImage = dataParser.CropImage(*image, 0, image->GetHeight() - imageSize, imageSize, imageSize, true);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedLowerLeftFlipped.jpg");
		delete croppedImage;
		// Test crop of lower right corner, with flip.
		croppedImage = dataParser.CropImage(*image, image->GetWidth() - imageSize, image->GetHeight() - imageSize, imageSize, imageSize, true);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedLowerRightFlipped.jpg");
		delete croppedImage;


		delete image;
	}

	return true;
}

bool TestJpegDataParser::TestResizeImageWithCropCu()
{
	vector<string> testImages;
	testImages.push_back("testImageH.JPEG");
	testImages.push_back("testImageV.JPEG");
	
	JpegDataParser dataParser;
	const uint imageSize = 224;
	const uint mediumImageSize = 2560;
	const uint largeImageSize = 4800;
	ImageData* image;
	ImageData* resizedImage;

	for (string testImage : testImages)
	{
		image = dataParser.LoadImage(m_inputFolder + testImage);
		string imageName = GetFileNameWithoutExtension(testImage);

		// Test resizing to smaller, cropping left.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, imageSize, ResizeMode::ResizeToSmaller, CropMode::CropLeft);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToSmallerCroppedLeft.jpg");
		delete resizedImage;
		// Test resizing to smaller, cropping top.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, imageSize, ResizeMode::ResizeToSmaller, CropMode::CropTop);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToSmallerCroppedTop.jpg");
		delete resizedImage;
		// Test resizing to smaller, cropping right.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, imageSize, ResizeMode::ResizeToSmaller, CropMode::CropRight);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToSmallerCroppedRight.jpg");
		delete resizedImage;
		// Test resizing to smaller, cropping bottom.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, imageSize, ResizeMode::ResizeToSmaller, CropMode::CropBottom);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToSmallerCroppedBottom.jpg");
		delete resizedImage;
		// Test resizing to smaller, cropping center.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, imageSize, ResizeMode::ResizeToSmaller, CropMode::CropCenter);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToSmallerCroppedCenter.jpg");
		delete resizedImage;

		// Test resizing to larger, cropping left.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, imageSize, ResizeMode::ResizeToLarger, CropMode::CropLeft);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToLargerCroppedLeft.jpg");
		delete resizedImage;
		// Test resizing to larger, cropping top.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, imageSize, ResizeMode::ResizeToLarger, CropMode::CropTop);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToLargerCroppedTop.jpg");
		delete resizedImage;
		// Test resizing to larger, cropping right.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, imageSize, ResizeMode::ResizeToLarger, CropMode::CropRight);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToLargerCroppedRight.jpg");
		delete resizedImage;
		// Test resizing to larger, cropping bottom.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, imageSize, ResizeMode::ResizeToLarger, CropMode::CropBottom);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToLargerCroppedBottom.jpg");
		delete resizedImage;
		// Test resizing to larger, cropping center.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, imageSize, ResizeMode::ResizeToLarger, CropMode::CropCenter);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToLargerCroppedCenter.jpg");
		delete resizedImage;

		// Test resizing to fit, cropping left.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, 2 * imageSize, ResizeMode::ResizeToFit, CropMode::CropLeft);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToFitCroppedLeft.jpg");
		delete resizedImage;
		// Test resizing to fit, cropping top.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, 2 * imageSize, ResizeMode::ResizeToFit, CropMode::CropTop);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToFitCroppedTop.jpg");
		delete resizedImage;
		// Test resizing to fit, cropping right.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, 2 * imageSize, ResizeMode::ResizeToFit, CropMode::CropRight);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToFitCroppedRight.jpg");
		delete resizedImage;
		// Test resizing to fit, cropping bottom.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, 2 * imageSize, ResizeMode::ResizeToFit, CropMode::CropBottom);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToFitCroppedBottom.jpg");
		delete resizedImage;
		// Test resizing to fit, cropping center.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, 2 * imageSize, ResizeMode::ResizeToFit, CropMode::CropCenter);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToFitCroppedCenter.jpg");
		delete resizedImage;

		// Test cropping with medium image size.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, mediumImageSize, mediumImageSize, ResizeMode::ResizeToLarger, CropMode::CropCenter);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToLargerCroppedCenterMediumSize.jpg");
		delete resizedImage;

		// Test cropping with large image size.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, largeImageSize, largeImageSize, ResizeMode::ResizeToLarger, CropMode::CropCenter);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToLargerCroppedCenterLargeSize.jpg");
		delete resizedImage;

		delete image;
	}

	return true;
}