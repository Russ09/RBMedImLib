//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//			Author: Russell Bates				     //
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
// 			C++ image analysis library 			     //
//				RBMedImLib.h				     //
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//


#include <iostream>
#include <string>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <map>

#include "nifti1.h"
#include "nifti1_io.h"

#define MIN_HEADER_SIZE 348
#define NII_HEADER_SIZE 352


//------------------Class 1: ScalarField----------------------------------------


class ScalarField{

private:
	//size of the field
	int NXNY; // size of slice
	int NXNYNZ; // size of volume
	
	// X,Y,Z,T voxel sizes (mm/seconds);
	int dX;
	int dY;
	int dZ;
	int dT;
	
	//scalar field
	float *field;
	
public:
	
	// X,Y,Z,T dimensions in voxels	
	int NX;
	int NY;
	int NZ;
	int NT;

	float im2wld[4][4]; //Image to World coordinates transformation
	float wld2im[4][4]; //World to Image coordinates transformation
	
	ScalarField(); //constructor
	ScalarField(const ScalarField& ScalField); //copy constructor
	
	~ScalarField(); //destructor

	void CreateVoidField(int NX,int NY, int NZ);
	
	inline float get(int x,int y, int z=0,int t=0) const; //retrieves given voxel value
	inline float get(int i)const;
	void set(float value,int x,int y, int z=0,int t=0); //sets given voxel to value
	void set(float value,int i); //set by linear indexing
	
	virtual float getDim(int dim)const; // gets the size in the selected dimension
	
	virtual void setAll(float value,int t); //sets all voxels in a specific frame to value
	
	virtual void multiplyFrameByScalar(float scalar,int t=0); //multiplies a specific frame by a scalar
	
	virtual float getMax(void); // retrieves the maximum value from the image volume
	virtual float getMin(void); // retrieves the minimum value from the image volume

	virtual float getRegionMax(int *indexList, int regionSize); // Gets maximum value of the scalar field in the described region
	virtual float getRegionMin(int *indexList, int regionSize); // Gets minimum value of the scalar field in the described region
	virtual float getRegionMedian(int *indexList, int regionSize); // Gets median value of the scalar field in the described region

    virtual void padArray(float padValue,int xPad,int yPad, int zPad=0,int tPad=0); //pads image with a set value
    virtual void padArraySymmetric(int xPad,int yPad, int zPad=0,int tPad=0); // pads array with symmetric boundary conditions

    virtual void unpadArray(int xPad, int yPad, int zPad, int tPad); // removes padding from edge of image

    //NIfTI I/O functions
	virtual void readNifti(char* imageName);
	virtual void writeNifti(char *imageName);

    //Swap function for = operator overload
	friend void swap(ScalarField& first, ScalarField& second);
	
	
	//---------------Overloaded unary operators---------------------
	ScalarField& operator=(ScalarField other);
	
	ScalarField& operator+=(const float& scalar);
	ScalarField& operator+=(const ScalarField& ScalField);
	ScalarField& operator-=(const float& scalar);
	ScalarField& operator-=(const ScalarField& ScalField);
	ScalarField& operator*=(const float& scalar);
	ScalarField& operator*=(const ScalarField& ScalField);
	ScalarField& operator/=(const float& scalar);
	ScalarField& operator/=(const ScalarField& ScalField);
	//---------------------------------------------------------------
};


	//---------------Overloaded binary operators------------------------------------
	ScalarField operator*(ScalarField ScalField,const float scalar); 		// multiplies the entire image volume by scalar 
	ScalarField operator*(ScalarField ScalField1,const ScalarField& ScalField2);	// element-wise multiplication between scalar fields (must have same dimensions)
	
	ScalarField operator+(ScalarField ScalField,const float scalar);		// increments entire image volume by scalar
	ScalarField operator+(ScalarField ScalField1,const ScalarField& ScalField2);	// element-wise addition between scalar fields (must have same dimensions)	
	
	ScalarField operator-(ScalarField ScalField,const float scalar);		// subtracts scalar from entire image volume
	ScalarField operator-(ScalarField ScalField1,const ScalarField& ScalField2);	// element-wise subtraction between scalar fields (must have same dimensions)
	
	ScalarField operator/(ScalarField ScalField,const float scalar);		// divides image volume by scalar
	ScalarField operator/(ScalarField ScalField1,const ScalarField& ScalField2);	// element-wise division between scalar fields (must have same dimensions)
	
	//--------------------------------------------------------------------------------
	
	
	
//-----------Class 2: Point---------------------------------------------------------

class Point{

private:

	int nDims;
	float *coords;
	
public:

	Point(); //constructor
	Point(int nDims);
	Point(const Point& P1); //copy constructor
	Point(float x, float y, float z); //initialise 3D point
	~Point(); //destructor
	
	
	
	int getDims(void)const{return nDims;}; //return the number of dimensions
	
	virtual float get(const int i) const; //retrieve value in ith dimension
	virtual void set(const int i,const float value); //set value in ith dimension
	
	friend void swap(Point& P1, Point& P2); //swaps Point P1 for Point P2
	friend float dot(Point& P1, Point& P2); //performs dot product between P1 and P2
	friend float abs(Point P1);
	friend float prod(Point P1);
	
	float& operator()(const int& i); //allows for indexing using round brackets P1(2) etc.
	
	//Overloaded unary operators
	Point& operator+=(const Point& P2); 
	Point& operator+=(const float& value); 
	Point& operator-=(const Point& P2);
	Point& operator-=(const float& value); 
	Point& operator/=(const Point& P2); 
	Point& operator/=(const float& value); 
	Point& operator*=(const Point& P2);
	Point& operator*=(const float& value); 
	
	Point& operator=(Point other);
};
	//Overloaded binary operators
	Point operator+(Point P1, const Point& P2);
	Point operator+(Point P1, const float& value);
	Point operator-(Point P1, const Point& P2);
	Point operator-(Point P1, const float& value);
	Point operator/(Point P1, const Point& P2);
	Point operator/(Point P1, const float& value);
	Point operator*(Point P1, const Point& P2);
	Point operator*(Point P1, const float& value);

	
//---------------Class 3: LineSegment----------------------------------------

class LineSegment{


private:
	int nPoints;
	int nDims;
	
	//float tortuosity;
	float length;
	float CLR;

	Point *Points;
	
public:

	LineSegment();
	LineSegment(const Point& P1);


    //Retrieves points from LineSegment
	virtual Point getPoint(int i);
	virtual Point getLastPoint(void);

    //Adds point to end of LineSegment
	virtual void addPoint(Point P1);

    //Removes points from LineSegment
    virtual void removePoint(int i);
	virtual void removeLastPoint(void); 
	
	//virtual void updateProperties(void);
	//virtual void calculateTortuosity(void);
	virtual void calculateLength(void);
	virtual void calculateCLR(void);

	virtual float getLength(void); //Calculates length of LineSegment in Voxel lengths, currently assumes voxel isotropy
	virtual float getCLR(void); //Calcaultes Chord-Length ratio of LineSegment (used as measure of tortuosity in vessel skeletons)
	
	LineSegment& operator+=(const LineSegment& LineSeg);
};


//-------------------Class 4: Kernel ------------------------------------------------

class Kernel{

private:
	int nDims;
	Point dims;
	Point centre;
	float *mData;
    int NX;
    int NY;
    int NZ;
    int NT;
	
public:

	Kernel();
	Kernel(Point dims);
	~Kernel();
	
	virtual float getValue(int i, int j, int k=0, int t=0);
	virtual void setValue(float value, int i, int j, int k=0, int t=0);

    Point getCentre(void){return centre;};
	int getCentre(int nDim){return centre(nDim);};
    Point getDims(void){return dims;};
	int getDims(int nDim){return (int)dims(nDim);};
	
	virtual void GenerateUniform(Point dims); //Generates uniformly valued Kernel for mean filtering
	virtual void GenerateGaussian(Point dims, float sd); //Generates Gaussian Kernel for used in Gaussian blur filtering
	//virtual void GenerateSumOfGaussians(Point dims, Point sds); // Generates sum of Gaussians Kernel with SD's of Gaussians specified by N-dimensional Point sds
    //virtual void GenerateLoG(Point dims); // Generates Laplacian of Gaussian Kernel for edge detection
};

//---------------------Class 5: ImageFilter------------------------------------------------

class ImageFilter{

public:
	ScalarField *image;
	Kernel loadedKernel;
	char *filterType;

    ImageFilter();
	~ImageFilter();

    void loadImage(ScalarField * inputImage);
    void loadKernel(Kernel inputKernel);
	//void loadImage(char *inputImageName);
	
	//void setFilterType(char *filterType);
	
	ScalarField * output(void);
	
	//virtual void update(void);
	
	void ConvolveWithKernel(Kernel inputKernel);




	
};


class FFTFilter: public ImageFilter{

private:
	ScalarField RealSignalForFFT;
	ScalarField ImagSignalForFFT;

	ScalarField RealFilterForFFT;
	ScalarField ImagFilterForFFT;

	ScalarField ImageTemp;

	int imageX,imageY,imageZ;
	int NXfft,NYfft,NZfft;

public:

	FFTFilter();
	~FFTFilter();


	void ConvolveWithKernel(void);

	void InitialiseConvolver(void);
	void AdjustKernelForFFT(void);

	//	Functions for FFT Convolution
	void ConvertKernel(void);
	void FFT1D(float *data, unsigned long nn, int isign);
	void DirectFFT(ScalarField *RealSignal, ScalarField *ImaginaryField);
	void InverseFFT(ScalarField *RealSignal, ScalarField *ImaginaryField);

};
	


//-------------------------Class 5: Supervoxels -----------------------------------------------

class SuperVoxel{

private:
	int nSuperVoxels;
	int nDims;
	int *dims;

	int **indices;
	std::map <std::pair<int,int>,int> connectivity;
	int *svSizes;

	int bConnectivity;
	int bIndices;

public:

	SuperVoxel(void);
	~SuperVoxel(void);

	void readSegmentationImage(ScalarField inputImage);
	int *getSuperVoxelIndices(int i);
	int getNumSuperVoxels(void);
	int getSizeSuperVoxel(int i);
	void getConnectivity(ScalarField inputImage, int **listOne=NULL, int **listTwo = NULL,int *nConn = NULL, int **sizeInterface = NULL);

};



//--------------------------Miscellaneous Functions ---------------------------------------------------


//Median array value:

float kth_smallest(float a[], uint16_t n, uint16_t k);
float quick_select_median(float arr[], uint16_t n);






