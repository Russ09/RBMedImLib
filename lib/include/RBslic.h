#pragma once

#include "RBMedImLib.h"
#include <vector>

class SLICsegmenter
{
private:

	//	Image parameters
	Image *baseImage;
	Image labelImage;
	std::vector<std::vector<int>> > SLICcentres;
	std::vector<int> imageDimensions;
	int numDimensions;

	//SLIC algorithm parameters
	int numCentres= 100;
	double compactness = 1;
	double sigma = 0;
	int spacing;

	//	Flags
	bool flag_LoadedImage = 0;
	bool flag_DoneSLIC = 0;
	bool flag_LabelChanges = 1;

	void updateCentres();
	void updateLabels();
	void initCentres();

public:
	SLICsegmenter();
	SLICsegmenter(Image *inputImage);

	~SLICsegmenter

	void setImage(Image *inputImage);

	void setNumCentres(int newNumCentres);
	void setCompactness(double newCompactness);
	void setSigma(double newSigma);

	void doSLIC();

	Image *returnLabelImage();
};


SLICsegmenter() {

}

SLICsegmenter(Image *baseImage) {

	Image *baseImage = inputImage.getPtr();
	Image labelImage = Image(inputImage);

	std::vector<std::vector<int>> > SLICcentres;
	std::vector<int> imageDimensions = baseImage.getDimensions();

	flag_LoadedImage = 1;
}

~SLICsegmenter() {
	~labelImage;
	delete SLICcentres;
	delete imageDimensions;
	delete numDimensions;
}

void setSigma(double newSigma)
{
	if (newSigma >= 0) {
		sigma = newSigma;
	}
	else {
		throw std::runtime_error("Must have sigma >= 0.");
	}
		
}

void setNumCentres(double newNumCentres)
{
	if (newNumCentres >= 5) {
		numCentres = newNumCentres;
	}
	else {
		throw std::runtime_error("Must have numCentres >= 5.");
	}

}

void setCompactness(double newCompactness)
{
	if (newCompactness >= 0) {
		compactness = newCompactness;
	}
	else {
		throw std::runtime_error("Must have compactness >= 0.");
	}

}

void doSLIC() {

	if (flag_LoadedImage == 0) {
		throw std::runtime_error("No image loaded.");
	}

	//SLIC algorithm
	initCentres();
	updateLabels();
	while (flag_LabelChanges == 1) {
		updateCentres();
		updateLabels();
	}

}

void initCentres() {

}