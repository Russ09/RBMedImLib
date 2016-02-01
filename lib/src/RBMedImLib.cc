//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//			Author: Russell Bates				     //
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
// 			C++ image analysis library 			     //
//				RBMedImLib.h				     //
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//


#include <algorithm>
#include <assert.h>
#include <omp.h>
#include <math.h>
#include "RBMedImLib.h"
#include <map>
#include <stdint.h>

# define M_PI           3.14159265358979323846  /* pi */

#define COMPILE_WITH_OPENMP 1

//------------------Class 1: ScalarField----------------------------------------


//constructor
ScalarField::ScalarField(void){
	this->NX=0;
	this->NY=0;
	this->NZ=0;
	this->NT=0;
	
	this->field = new float[10];
}

ScalarField::ScalarField(const ScalarField& ScalField){
	int x,y,z,t;
	float thisVal;
	
	this->NX = ScalField.NX;
	this->NY = ScalField.NY;
	this->NZ = ScalField.NZ;
	this->NT = ScalField.NT;
	
	this->field = new float[this->NX*this->NY*this->NZ*this->NT];
	
	for(t=0;t<this->NT;t++){
	for(z=0;z<this->NZ;z++){
	for(y=0;y<this->NY;y++){
	for(x=0;x<this->NX;x++){
		thisVal = ScalField.get(x,y,z,t);
		this->set(thisVal,x,y,z,t);
	}}}}
}

//destructor
ScalarField::~ScalarField(void){
	delete[] this->field;
}

void ScalarField::CreateVoidField(int NX, int NY, int NZ) {
    this->NX = NX;
    this->NY = NY;
    this->NZ = NZ;
    this->NT = 0;

    this->field = new float[NX*NY*NZ];
    for(int i=0;i<this->NX*this->NY*this->NZ;i++){
        this->set(0,i);
    }

}

inline float ScalarField::get(int x,int y, int z,int t)const{
	return field[t*NX*NY*NZ + z*this->NX*NY + y*NX + x];
}

inline float ScalarField::get(int i)const{
	return field[i];
}

void ScalarField::set(float value,int x,int y, int z,int t){
	field[t*NX*NY*NZ + z*NX*NY + y*NX + x] = value;
}

void ScalarField::set(float value,int i){
	field[i] = value;
}

float ScalarField::getDim(int dim)const{
	switch(dim){
		case 1:
			return this->NX;
		case 2:
			return this->NY;
		case 3:
			return this->NZ;
		case 4:
			return this->NT;
	}
}

void ScalarField::setAll(float value,int t){
	int i;
	for(i=0;i<NX*NY*NZ;i++){
	field[t*NX*NY*NZ+i] = value;
	};
}

void ScalarField::multiplyFrameByScalar(float scalar,int t){
	int i;
	for(i=0;i<NX*NY*NZ;i++) field[t*NX*NY*NZ+i]*=scalar;
}

float ScalarField::getMax(void){
	int i;
	float maxVal;
	float thisVal;
	
	maxVal = this->get(0);

    for(i=0;i<this->NX*this->NY*this->NZ*this->NT;i++){
		thisVal = this->get(i);
		if(thisVal > maxVal){
			maxVal = thisVal;
		}
	}
    return maxVal;
			
}

float ScalarField::getMin(void){
	int x,y,z,t;
	float minVal;
	float thisVal;
	
	minVal = this->get(0,0,0,0);

    #pragma omp parallel for collapse(4)
    for(t=0;t<this->NT;t++){
    for(z=0;z<this->NZ;z++){
    for(y=0;y<this->NY;y++){
    for(x=0;x<this->NX;x++){
		thisVal = this->get(x,y,z,t);
		if(thisVal < minVal){
			minVal = thisVal;
		}
	}}}}

    return minVal;
			
}

float ScalarField::getRegionMax(int *indexList, int regionSize){

    float tmpMax = this->get(indexList[0]);
    int i;

    for(i=0;i<regionSize;i++){
        if(this->get(indexList[i]) > tmpMax){
            tmpMax = this->get(indexList[i]);
        }
    }

    return tmpMax;

}

float ScalarField::getRegionMin(int *indexList, int regionSize){

    float tmpMin = this->get(indexList[0]);
    int i;

    for(i=0;i<regionSize;i++){
        if(this->get(indexList[i]) < tmpMin){
            tmpMin = this->get(indexList[i]);
        }
    }

    return tmpMin;

}

float ScalarField::getRegionMedian(int *indexList, int regionSize){

    float *regionVals = new float[regionSize];
    int i;
    float medianVal;

    for(i=0;i<regionSize;i++){
        regionVals[i] = this->get(indexList[i]);
    }

    medianVal = quick_select_median(regionVals,regionSize);

    delete(regionVals);
    return medianVal;

}

void ScalarField::padArray(float padValue,int xPad,int yPad, int zPad,int tPad) {
    bool xrange,yrange,zrange,trange,allRange;
    int newNX,newNY,newNZ,newNT;
    int x,y,z,t;
    int newPadX,newPadY,newPadZ,newPadT;
    float *newImage;

    newPadX = xPad;
    newPadY = yPad;
    newPadZ = zPad;
    newPadT = tPad;

    newNX = NX + 2*newPadX;
    newNY = NY + 2*newPadY;
    newNZ = NZ + 2*newPadZ;
    newNT = NT + 2*newPadT;

    newImage = new float[newNX*newNY*newNZ*newNT];

    #pragma omp parallel for
    for(t=0;t<newNT;t++){
    for(z=0;z<newNZ;z++){
    for(y=0;y<newNY;y++){
    for(x=0;x<newNX;x++){
        xrange = (x>=newPadX) && x<(NX + newPadX);
        yrange = (y>=newPadY) && y<(NY + newPadY);
        zrange = (z>=newPadZ) && z<(NZ + newPadZ);
        trange = (t>=newPadT) && t<(NT + newPadT);
        allRange = xrange && yrange && zrange && trange;
        if(allRange){
            newImage[newNX*newNY*newNZ*t+newNX*newNY*z + newNX*y + x]= field[(int)((t -newPadT)*NX*NY*NZ + (z-newPadZ)*NX*NY + (y-newPadY)*NX + (x-newPadX))];
        }
        else{
            newImage[newNX*newNY*newNZ*t+newNX*newNY*z + newNX*y + x] = padValue;
        }
    }}}}

    NX = newNX;
    NY = newNY;
    NZ = newNZ;
    NT = newNT;

    field = newImage;
}


void ScalarField::padArraySymmetric(int xPad, int yPad, int zPad, int tPad) {
    bool xrange,yrange,zrange,trange,allRange;
    int newNX,newNY,newNZ,newNT;
    int x,y,z,t;
    int newPadX,newPadY,newPadZ,newPadT;
    int deltaX,deltaY,deltaZ,deltaT;
    bool xrangeU,xrangeL,yrangeU,yrangeL,zrangeU,zrangeL,trangeU,trangeL;
    float *newImage;

    newPadX = xPad;
    newPadY = yPad;
    newPadZ = zPad;
    newPadT = tPad;

    newNX = NX + 2*newPadX;
    newNY = NY + 2*newPadY;
    newNZ = NZ + 2*newPadZ;
    newNT = NT + 2*newPadT;

    newImage = new float[newNX*newNY*newNZ*newNT];

    for(t=0;t<newNT;t++){
        for(z=0;z<newNZ;z++){
            for(y=0;y<newNY;y++){
                for(x=0;x<newNX;x++){
                    xrangeL = (x>=newPadX);
                    xrangeU = x<(NX + newPadX);
                    xrange = xrangeU && xrangeL;

                    yrangeL = (y>=newPadY);
                    yrangeU = y<(NY + newPadY);
                    yrange = yrangeL && yrangeU;

                    zrangeL = (z>=newPadZ);
                    zrangeU = z<(NZ + newPadZ);
                    zrange = zrangeL && zrangeU;

                    trangeL = (t>=newPadT);
                    trangeU = t<(NT + newPadT);
                    trange = trangeL && trangeU;

                    allRange = xrange && yrange && zrange && trange;

                    if(allRange){
                        newImage[newNX*newNY*newNZ*t+newNX*newNY*z + newNX*y + x] = field[(int)((t -newPadT)*NX*NY*NZ + (z-newPadZ)*NX*NY + (y-newPadY)*NX + (x-newPadX))];
                    }
                    else{
                        if(!xrangeL) {deltaX = newPadX-x;}else if(!xrangeU){deltaX = 2*NX + newPadX - x;}else{deltaX = x-newPadX;}
                        if(!yrangeL) {deltaY = newPadY-y;}else if(!yrangeU){deltaY = 2*NY + newPadY - y;}else{deltaY = y-newPadY;}
                        if(!zrangeL) {deltaZ = newPadZ-z;}else if(!zrangeU){deltaZ = 2*NZ + newPadZ - z;}else{deltaZ = z-newPadZ;}
                        if(!trangeL) {deltaT = newPadT-t;}else if(!trangeU){deltaT = 2*NT + newPadT - t;}else{deltaT = t-newPadT;}

                        newImage[newNX*newNY*newNZ*t+newNX*newNY*z + newNX*y + x] = field[(int)(deltaT*NX*NY*NZ + deltaZ*NX*NY + deltaY*NX + deltaX)];
                    }
                }}}}

    NX = newNX;
    NY = newNY;
    NZ = newNZ;
    NT = newNT;

    field = newImage;
}

void ScalarField::unpadArray(int xPad, int yPad, int zPad, int tPad){

    int newPadX,newPadY,newPadZ,newPadT;
    int newNX,newNY,newNZ,newNT;
    int x,y,z,t;

    newPadX = xPad;
    newPadY = yPad;
    newPadZ = zPad;
    newPadT = tPad;

    assert(this->NX > 2*newPadX);
    assert(this->NY > 2*newPadY);
    assert(this->NZ > 2*newPadZ);
    assert(this->NT > 2*newPadT);

    newNX = this->NX - 2*newPadX;
    newNY = this->NY - 2*newPadY;
    newNZ = this->NZ - 2*newPadZ;
    newNT = this->NT - 2*newPadT;

    float *newImage;

    newImage = new float[newNX*newNY*newNZ*newNT];



    for(t=newPadT;t<(this->NT-newPadT);t++){
    for(z=newPadZ;z<(this->NZ-newPadZ);z++){
    for(y=newPadY;y<(this->NY-newPadY);y++){
    for(x=newPadX;x<(this->NX-newPadX);x++){
        newImage[(t-newPadT)*newNX*newNY*newNZ + (z-newPadZ)*newNX*newNY + (y-newPadY)*newNX + (x-newPadX)] = this->field[t*this->NX*this->NY*this->NZ + z*this->NX*this->NY + y*this->NX + x];
    }}}}

    this->field = newImage;
    this->NX = newNX;
    this->NY = newNY;
    this->NZ = newNZ;
    this->NT = newNT;

}

void ScalarField::readNifti(char *imageName){

	nifti_1_header hdr;
	FILE *fp;
	int tmp;
	int x,y,z,t;
	unsigned char data2;
	signed short data4;
	signed int data8;
	float data16;
	double data64;
	signed char data256;
	unsigned short data512;
	unsigned int data768;

	//open nifti image
	fp = fopen(imageName,"r");

    assert(fp!=NULL);

	//read header information into nifti header class
	tmp = fread(&hdr,MIN_HEADER_SIZE,1,fp);

	//to open some multichannel 3D images which have the channels in the 5th dimension instead of the 4th dimension
	if ((hdr.dim[4]==1)&&(hdr.dim[5]>1)){
	    hdr.dim[4]=hdr.dim[5];
	    hdr.dim[5]=1;
	}

	this->NX=static_cast<int>(hdr.dim[1]);
	this->NY=static_cast<int>(hdr.dim[2]);
	this->NZ=static_cast<int>(hdr.dim[3]);
	this->NT=static_cast<int>(hdr.dim[4]);
	
	this->dX=hdr.pixdim[1];
	this->dY=hdr.pixdim[2];
	this->dZ=hdr.pixdim[3];
	this->dT=hdr.pixdim[4];
	
	if (this->NT<1) {this->NT=1; hdr.dim[4]=1;}
  	this->NXNY=this->NX*this->NY;
  	this->NXNYNZ=this->NXNY*this->NZ;
	
	this->field = new float[this->NXNYNZ * this->NT];

	
	// jump to data offset
  	tmp = fseek(fp, (long)(hdr.vox_offset), SEEK_SET);
  	
  	if(hdr.scl_slope == 0){
  		hdr.scl_slope = 1;
  	}
  	
	  	//load the image
	  if (hdr.datatype==2){ //2  ->  unsigned char +++++++++++++++++++++++++++++++++++++++++++++++++
	    for(t=0;t<this->NT;t++) for(z=0;z<this->NZ;z++) for(y=0;y<this->NY;y++) for(x=0;x<this->NX;x++){
	      tmp = fread(&data2, sizeof(unsigned char), 1, fp);
	      this->set(static_cast<float>((data2 * hdr.scl_slope) + hdr.scl_inter),x,y,z,t);
	    }
	  }
	  else  if (hdr.datatype==4){ //4  ->  signed short +++++++++++++++++++++++++++++++++++++++++++++++++
	    for(t=0;t<this->NT;t++) for(z=0;z<this->NZ;z++) for(y=0;y<this->NY;y++) for(x=0;x<this->NX;x++){
	      tmp = fread(&data4, sizeof(signed short), 1, fp);
	      this->set(static_cast<float>((data4 * hdr.scl_slope) + hdr.scl_inter),x,y,z,t);
	    }
	  }
	  else  if (hdr.datatype==8){  //8  ->  signed int +++++++++++++++++++++++++++++++++++++++++++++++++
	    for(t=0;t<this->NT;t++) for(z=0;z<this->NZ;z++) for(y=0;y<this->NY;y++) for(x=0;x<this->NX;x++){
	      tmp = fread(&data8, sizeof(signed int), 1, fp);
	      this->set(static_cast<float>((data8 * hdr.scl_slope) + hdr.scl_inter),x,y,z,t);
	    }
	  }
	  else  if (hdr.datatype==16){  //16  ->  float +++++++++++++++++++++++++++++++++++++++++++++++++
	    for(t=0;t<this->NT;t++) for(z=0;z<this->NZ;z++) for(y=0;y<this->NY;y++) for(x=0;x<this->NX;x++){
	      tmp = fread(&data16, sizeof(float), 1, fp);
	      this->set(static_cast<float>((data16 * hdr.scl_slope) + hdr.scl_inter),x,y,z,t);
	    }
	  }
	  else  if (hdr.datatype==64){  //64  ->  double +++++++++++++++++++++++++++++++++++++++++++++++++
	    for(t=0;t<this->NT;t++) for(z=0;z<this->NZ;z++) for(y=0;y<this->NY;y++) for(x=0;x<this->NX;x++){
	      tmp = fread(&data64, sizeof(double), 1, fp);
	      this->set(static_cast<float>((data64 * hdr.scl_slope) + hdr.scl_inter),x,y,z,t);
	    }
	  }
	  else  if (hdr.datatype==256){  //256  ->  signed char +++++++++++++++++++++++++++++++++++++++++++++++++
	    for(t=0;t<this->NT;t++) for(z=0;z<this->NZ;z++) for(y=0;y<this->NY;y++) for(x=0;x<this->NX;x++){
	      tmp = fread(&data256, sizeof(signed char), 1, fp);
	      this->set(static_cast<float>((data256 * hdr.scl_slope) + hdr.scl_inter),x,y,z,t);
	    }
	  }
	  else  if (hdr.datatype==512){  //512  ->  unsigned short +++++++++++++++++++++++++++++++++++++++++++++++++
	    for(t=0;t<this->NT;t++) for(z=0;z<this->NZ;z++) for(y=0;y<this->NY;y++) for(x=0;x<this->NX;x++){
	      tmp = fread(&data512, sizeof(unsigned short), 1, fp);
	      this->set(static_cast<float>((data512 * hdr.scl_slope) + hdr.scl_inter),x,y,z,t);
	    }
	  }
	  else  if (hdr.datatype==768){  //768  ->  unsigned int +++++++++++++++++++++++++++++++++++++++++++++++++
	    for(t=0;t<this->NT;t++) for(z=0;z<this->NZ;z++) for(y=0;y<this->NY;y++) for(x=0;x<this->NX;x++){
	      tmp = fread(&data768, sizeof(unsigned int), 1, fp);
	      this->set(static_cast<float>((data768 * hdr.scl_slope) + hdr.scl_inter),x,y,z,t);
	    }
	  }
	  else{
	    std::cout << "Image contains an unsupported graylevel type" << std::endl;
	  }
	  
	  fclose(fp);
	 

}

///write a scalar field in a nifti image
void ScalarField::writeNifti(char *imageName){
  int i;
  int x,y,z,t;
  nifti_1_header hdr;
  nifti1_extender pad={0,0,0,0};
  FILE *fp;
  int ret;
  float *data=NULL;
  
  
  //1) create the header
  memset((void *)&hdr,0, sizeof(hdr));
  hdr.sizeof_hdr = MIN_HEADER_SIZE;
  hdr.dim[0] = 4;
  hdr.dim[1] = (short) this->NX;
  hdr.dim[2] = (short) this->NY;
  hdr.dim[3] = (short) this->NZ;
  hdr.dim[4] = (short) this->NT;
  hdr.datatype = NIFTI_TYPE_FLOAT32;
  hdr.bitpix = 32; 
  hdr.qform_code=0; // should ideally be set to 1 but I don't set the values of 'quatern_b', 'quatern_c' and 'quatern_d'
  hdr.pixdim[1] = this->dX;
  hdr.pixdim[2] = this->dY;
  hdr.pixdim[3] = this->dZ;
  hdr.pixdim[4] = this->dT;
  hdr.qoffset_x=this->im2wld[0][3];
  hdr.qoffset_y=this->im2wld[1][3];
  hdr.qoffset_z=this->im2wld[2][3];
  hdr.sform_code=1;
  hdr.srow_x[0]=this->im2wld[0][0];  hdr.srow_x[1]=this->im2wld[0][1];  hdr.srow_x[2]=this->im2wld[0][2];  hdr.srow_x[3]=this->im2wld[0][3];
  hdr.srow_y[0]=this->im2wld[1][0];  hdr.srow_y[1]=this->im2wld[1][1];  hdr.srow_y[2]=this->im2wld[1][2];  hdr.srow_y[3]=this->im2wld[1][3];
  hdr.srow_z[0]=this->im2wld[2][0];  hdr.srow_z[1]=this->im2wld[2][1];  hdr.srow_z[2]=this->im2wld[2][2];  hdr.srow_z[3]=this->im2wld[2][3];
  hdr.vox_offset = (float) NII_HEADER_SIZE;
  hdr.scl_inter = 0.0;
  hdr.scl_slope = 1.0;
  hdr.xyzt_units = NIFTI_UNITS_MM | NIFTI_UNITS_SEC;
  strncpy(hdr.magic, "n+1\0", 4);

  //2) save the image OutputImageName
  //allocate and fill the buffer 
  data = new float [hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4]];
  
  i=0;
  for(t=0;t<this->NT;t++) for(z=0;z<this->NZ;z++) for(y=0;y<this->NY;y++) for(x=0;x<this->NX;x++){
    data[i] = this->get(x,y,z,t);
    i++;
  }
  
  // write first 348 bytes of header  
  fp = fopen(imageName,"wb");
  ret = fwrite(&hdr, MIN_HEADER_SIZE, 1, fp);
  
  // write extender pad and image data  
  ret = fwrite(&pad, 4, 1, fp);
  ret = fwrite(data, (size_t)(hdr.bitpix/8), hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4], fp);
  
  fclose(fp);
  delete(data);

}

void swap(ScalarField& first, ScalarField& second){
        
        using std::swap; 

        swap(first.NX, second.NX);
        swap(first.NY, second.NY);
        swap(first.NZ, second.NZ);
        swap(first.NT, second.NT);
        
        swap(first.dX,second.dX);
        swap(first.dY,second.dY);
        swap(first.dZ,second.dZ);
        swap(first.dT,second.dT);
        
        std::swap(first.field, second.field);
        
        
}

ScalarField& ScalarField::operator=(ScalarField other){
	
	swap(*this,other);	
	return *this;
}

ScalarField& ScalarField::operator+=(const float& scalar){
	int i;
	for(i=0;i<NX*NY*NZ*NT;i++) field[i] += scalar;
	return *this;
}

ScalarField& ScalarField::operator+=(const ScalarField& ScalField){
	int i;
	if(NX==ScalField.getDim(1)&&NY==ScalField.getDim(2)&&NZ==ScalField.getDim(3)&&NT==ScalField.getDim(4)){
		for(i=0;i<NX*NY*NZ*NT;i++) field[i] += ScalField.get(i);
		return *this;
	}
	else{
		std::cout<<"Dimensional mismatch when attempting ScalarField addition";
	}
}

ScalarField& ScalarField::operator-=(const float& scalar){
	int i;
	for(i=0;i<NX*NY*NZ*NT;i++) field[i] -= scalar;
	return *this;
}

ScalarField& ScalarField::operator-=(const ScalarField& ScalField){
	int i;
	if(NX==ScalField.getDim(1)&&NY==ScalField.getDim(2)&&NZ==ScalField.getDim(3)&&NT==ScalField.getDim(4)){
		for(i=0;i<NX*NY*NZ*NT;i++) field[i] -= ScalField.get(i);
		return *this;
	}
	else{
		std::cout<<"Dimensional mismatch when attempting ScalarField subtraction";
	}
}

ScalarField& ScalarField::operator*=(const float& scalar){
	int i;
	for(i=0;i<NX*NY*NZ*NT;i++) field[i] *= scalar;
	return *this;
}

ScalarField& ScalarField::operator*=(const ScalarField& ScalField){
	int i;
	if(NX==ScalField.getDim(1)&&NY==ScalField.getDim(2)&&NZ==ScalField.getDim(3)&&NT==ScalField.getDim(4)){
		for(i=0;i<NX*NY*NZ*NT;i++) field[i] *= ScalField.get(i);
		return *this;
	}
	else{
		std::cout<<"Dimensional mismatch when attempting ScalarField multiplication";
	}
}

ScalarField& ScalarField::operator/=(const float& scalar){
	int i;
	for(i=0;i<NX*NY*NZ*NT;i++) field[i] /= scalar;
	return *this;
}

ScalarField& ScalarField::operator/=(const ScalarField& ScalField){
	int i;
	if(NX==ScalField.getDim(1)&&NY==ScalField.getDim(2)&&NZ==ScalField.getDim(3)&&NT==ScalField.getDim(4)){
		for(i=0;i<NX*NY*NZ*NT;i++) field[i] /= ScalField.get(i);
		return *this;
	}
	else{
		std::cout<<"Dimensional mismatch when attempting ScalarField division";
	}
}

ScalarField operator+(ScalarField ScalField,const float scalar){
	ScalField += scalar;
	return ScalField;
}

ScalarField operator+(ScalarField ScalField1,const ScalarField& ScalField2){
	ScalField1 += ScalField2;
	return ScalField1;
}

ScalarField operator-(ScalarField ScalField,const float scalar){
	ScalField -= scalar;
	return ScalField;
}

ScalarField operator-(ScalarField ScalField1,const ScalarField& ScalField2){
	ScalField1 -= ScalField2;
	return ScalField1;
}

ScalarField operator*(ScalarField ScalField,const float scalar){
	ScalField *= scalar;
	return ScalField;
}

ScalarField operator*(ScalarField ScalField1,const ScalarField& ScalField2){
	ScalField1 *= ScalField2;
	return ScalField1;
}

ScalarField operator/(ScalarField ScalField,const float scalar){
	ScalField /= scalar;
	return ScalField;
}

ScalarField operator/(ScalarField ScalField1,const ScalarField& ScalField2){
	ScalField1 /= ScalField2;
	return ScalField1;
}



//------------------Class 2: Point---------------------------------------------

Point::Point(void){
	nDims = 3;
	coords = new float[3];
	coords[0] = 0;
	coords[1] = 0;
	coords[2] = 0;
}

Point::Point(int inputDims){

	int i;
	nDims = inputDims;
	coords = new float[nDims];
	
	for(i=0;i<nDims;i++){
		coords[i] = 0;
	}
}

Point::Point(const Point& P1){
	int i;
	nDims = P1.getDims();
	coords = new float[nDims];
	for(i=0;i<nDims;i++){
		coords[i] = P1.get(i);
	}
}

Point::Point(float x, float y, float z){
	nDims = 3;
	coords = new float[3];
	coords[0] = x;
	coords[1] = y;
	coords[2] = z;
}

Point::~Point(){
	delete coords;
}

float Point::get(const int i)const{
	if(i>=0 && i<nDims){
		return coords[i];
	}
	else{
		std::cout<<"Assigned dimension was outside the number of possible dimensions\n"; 
	}
}

void Point::set(const int i,const float value){
	if(i>=0 && i<nDims){
		coords[i] = value;
	}
	else{
		std::cout<<"Assigned dimension was outside the number of possible dimensions\n"; 
	}
}

void swap(Point& P1, Point& P2){

        using std::swap; 
        swap(P1.nDims, P2.nDims);
        swap(P1.coords,P2.coords);
}

float dot(Point& P1, Point& P2){
	int i;
	float dotVal;
	dotVal = 0;
	
	if(P1.nDims == P2.nDims){
		for(i=0;i<P1.nDims;i++){
			dotVal +=P1(i)*P2(i);
		}
	}
	return dotVal;
}

float abs(Point P1){
	int i;
	float absVal;
	
	absVal = 0;
	
	for(i=0;i<P1.getDims();i++){
		absVal += P1.get(i)*P1.get(i);
	}
	
	return (float) sqrt(absVal);
}

float prod(Point P1){
	int i;
	float prodVal;
	
	prodVal = 1;
	
	for(i=0;i<P1.getDims();i++){
		prodVal *= P1.get(i);
	}
	
	return prodVal;
}

float& Point::operator()(const int& i){
	if(i <= nDims && i>0){
		return this->coords[i-1];
	}
	else{
		std::cout<<"Coordinate specified is higher than the number of dimensions\n";
	}
}
	
Point& Point::operator+=(const Point& P2){
	int i;
	if(this->nDims == P2.getDims()){
		for(i=0;i<this->nDims;i++){
			coords[i] += P2.get(i);
		}
	}
}

Point& Point::operator+=(const float& value){
	int i;
	for(i=0;i<this->nDims;i++){
		coords[i] += value;
	}
}

Point& Point::operator-=(const Point& P2){
	int i;
	if(this->nDims == P2.getDims()){
		Point POut;
		for(i=0;i<this->nDims;i++){
			coords[i] -= P2.get(i);
		}
	}
}

Point& Point::operator-=(const float& value){
	int i;
	for(i=0;i<this->nDims;i++){
		coords[i] -= value;
	}
}

Point& Point::operator/=(const Point& P2){
	int i;
	if(this->nDims == P2.getDims()){
		Point POut;
		for(i=0;i<this->nDims;i++){
			coords[i] /= P2.get(i);
		}
	}
}

Point& Point::operator/=(const float& value){
	int i;
	for(i=0;i<this->nDims;i++){
		coords[i] /= value;
	}
}

Point& Point::operator*=(const Point& P2){
	int i;
	if(this->nDims == P2.getDims()){
		Point POut;
		for(i=0;i<this->nDims;i++){
			coords[i] *= P2.get(i);
		}
	}
}

Point& Point::operator*=(const float& value){
	int i;
	for(i=0;i<this->nDims;i++){
		coords[i] *= value;
	}
}

Point& Point::operator=(Point other){
	swap(*this,other);
	return *this;
}

Point operator+(Point P1, const Point& P2){
	P1 += P2;
	return P1;
}

Point operator+(Point P1, const float& value){
	P1 += value;
	return P1;
}

Point operator-(Point P1, const Point& P2){
	P1 -= P2;
	return P1;
}

Point operator-(Point P1, const float& value){
	P1 -= value;
	return P1;
}

Point operator/(Point P1, const Point& P2){
	P1 /= P2;
	return P1;
}

Point operator/(Point P1, const float& value){
	P1 /= value;
	return P1;
}

Point operator*(Point P1, const Point& P2){
	P1 *= P2;
	return P1;
}

Point operator*(Point P1, const float& value){
	P1 *= value;
	return P1;
}

//-----------------Class 3: Line Segment---------------------------------------


LineSegment::LineSegment(){
	nPoints = 0;
}

LineSegment::LineSegment(const Point& P1){
	nPoints = 1;
	nDims = P1.getDims();
	Points = new Point[1];
	Points[0] = P1;
}

Point LineSegment::getPoint(int i){
	assert(i>0 && i<=nPoints);
	return Points[i-1];
}

Point LineSegment::getLastPoint(void){
	return Points[nPoints-1];
}

void LineSegment::addPoint(Point P1){

	int i;
	assert(P1.getDims() == nDims);
	nPoints += 1;
	Point *newPoints;
	newPoints = new Point[nPoints];
	for(i=0;i<(nPoints-1);i++){
		newPoints[i] = Points[i];
	}
	newPoints[nPoints-1] = P1;
	Points = newPoints;
	
}

void LineSegment::removePoint(int i){
	
	int j;
	assert(i<= nPoints);
	nPoints -= 1;
	Point *newPoints;
	
	newPoints = new Point[nPoints];
	
	
	if(i==nPoints+1){
		for(j=0;j<(nPoints);j++){
			newPoints[j] = Points[j];	
		}
	}
	else if(i == 1){
		for(j=1;j<(nPoints+1);j++){
			newPoints[j-1] = Points[j];	
		}
	}
	
	else if(i>1 && i<nPoints){
		for(j=0;j<(i-1);j++){
			newPoints[j] = Points[j];	
		}
		for(j=(i);j<(nPoints+1);j++){
			newPoints[j-1] = Points[j];	
		}
	}
	
	Points = newPoints;	
}

void LineSegment::removeLastPoint(void){

	this->removePoint(nPoints);

}

void LineSegment::calculateLength(void){
	
	int i;
	float lengthVal;
	lengthVal = 0;
	
	for(i=0;i<(nPoints-1);i++){
		lengthVal += abs(Points[i+1] - Points[i]);
	}
	
	length = lengthVal;
}

void LineSegment::calculateCLR(void){

	float chordLength;
	this->calculateLength();
	
	chordLength = abs(getPoint(1) - getLastPoint());
	CLR = length/chordLength;
}

float LineSegment::getLength(void){
	return length;
}

float LineSegment::getCLR(void){
	return CLR;
}


//------------------Class 4: Kernel ----------------------------------------------

Kernel::Kernel(void){

	nDims = 2;
	Point dims(nDims);

}

Kernel::Kernel(Point inputDims){

	nDims = inputDims.getDims();
	dims = inputDims;
	mData = new float[(int)prod(inputDims)];
    NX = (int) inputDims(1);
    NY = (int) inputDims(2);
    if(inputDims.getDims() > 2){NZ = (int) inputDims(3);}else{NZ = 1;}
    if(inputDims.getDims() > 3){NT = (int) inputDims(4);}else{NT = 1;}

}

Kernel::~Kernel(){
}

float Kernel::getValue(int i, int j, int k, int t){

		return mData[(int)(t*NX*NY*NZ + k*NX*NY + j*NX + i)];
}

void Kernel::setValue(float value, int i, int j, int k, int t){

	if(nDims == 2){
		mData[(int)(j*dims(1) + i)] = value;
	}	
	if(nDims == 3){
		mData[(int)(k*dims(1)*dims(2) + j*dims(1) + i)] = value;
	}
	if(nDims == 4){
		mData[(int)(t*dims(1)*dims(2)*dims(3) + k*dims(1)*dims(2) + j*dims(1) + i)] = value;
	}

}

void Kernel::GenerateUniform(Point inputDims){

	int i;
	assert((int)(inputDims(1))%2 == 1);
	assert((int)(inputDims(2))%2 == 1);
	assert((int)(inputDims(3))%2 == 1);

    NX = (int) inputDims(1);
    NY = (int) inputDims(2);
    if(inputDims.getDims() > 2){NZ = (int) inputDims(3);}else{NZ = 1;}
    if(inputDims.getDims() > 3){NT = (int) inputDims(4);}else{NT = 1;}
	
	nDims = inputDims.getDims();
	dims = inputDims;
	mData = new float[(int)(prod(inputDims))];
	
	Point centre(inputDims.getDims());
	centre = (inputDims + 1)/2;
	
	for(i=0;i<prod(inputDims);i++){
		mData[i] = (float)1.0/prod(inputDims);
	}
	
}

void Kernel::GenerateGaussian(Point inputDims, float sd){
	
	int i,j,k,l,x,y,z,t;
	float gaussVal;
	double num, denom;
	assert((int)(inputDims(1))%2 == 1);
	assert((int)(inputDims(2))%2 == 1);
	assert((int)(inputDims(3))%2 == 1);
	
	nDims = inputDims.getDims();
	dims = inputDims;
	mData = new float[(int)(prod(inputDims))];

    NX = (int) inputDims(1);
    NY = (int) inputDims(2);
    if(inputDims.getDims() > 2){NZ = (int) inputDims(3);}else{NZ = 1;}
    if(inputDims.getDims() > 3){NT = (int) inputDims(4);}else{NT = 1;}
	
	gaussVal = 0;
	
	if(inputDims.getDims() == 2){
        for(j=0;j<inputDims(2);j++){
		for(i=0;i<inputDims(1);i++){
			x = i-centre(1)+1;
			y = j-centre(2)+1;
			num = (double)(-(x*x + y*y));
			denom = (double)(2*sd*sd);
			gaussVal = exp(num/denom);
			gaussVal = gaussVal/(sd*sqrt(2*M_PI));
			this->setValue(gaussVal,i,j);
		}}
	}
	if(inputDims.getDims() == 3){
	
		
		assert((int)(inputDims(3))%2 == 1);
        for(k=0;k<inputDims(3);k++){
        for(j=0;j<inputDims(2);j++){
		for(i=0;i<inputDims(1);i++){
			x = i-centre(1)+1;
			y = j-centre(2)+1;
			z = k-centre(3)+1;
			num = (double)(-(x*x + y*y + z*z));
			denom = (double)(2*sd*sd);
			gaussVal = exp(num/denom);
			gaussVal = gaussVal/(sd*sqrt(2*M_PI));
			this->setValue(gaussVal,i,j,k);
		}}}
	}
	if(inputDims.getDims() == 4){
		assert((int)(inputDims(3))%2 == 1);
		assert((int)(inputDims(4))%2 == 1);

        #pragma omp parallel for

        for(l=0;l<inputDims(4);l++){
        for(k=0;k<inputDims(3);k++){
        for(j=0;j<inputDims(2);j++){
		for(i=0;i<inputDims(1);i++){
			x = i-centre(1)+1;
			y = j-centre(2)+1;
			z = k-centre(3)+1;
			t = l-centre(4)+1;
			num = (double)(-(x*x + y*y + z*z + t*t));
			denom = (double)(2*sd*sd);
			gaussVal = exp(num/denom);
			gaussVal = gaussVal/(sd*sqrt(2*M_PI));
			this->setValue(gaussVal,i,j,k,l);
		}}}}
	}
	
	//normalise such that sum of the kernel is equal to 1
	float sum;
	sum = 0;
	for(i=0;i<prod(inputDims);i++){
		sum += mData[i]; 
	}
	for(i=0;i<prod(inputDims);i++){
		mData[i] /= sum; 
	}
}



//--------------Class 5: ImageFilter---------------------------------------------------


ImageFilter::ImageFilter() {
    ScalarField inputImage;
    Kernel loadedKernel;
}

ImageFilter::~ImageFilter(){

}

void ImageFilter::loadImage(ScalarField * inputImage){
	this->image = inputImage;
}

void ImageFilter::loadKernel(Kernel inputKernel) {
    this->loadedKernel = inputKernel;
}

ScalarField * ImageFilter::output(void){
    return this->image;
}

void ImageFilter::ConvolveWithKernel(Kernel inputKernel) {

    int pX,pY,pZ,pT;
    int i,j,k,l;
    int x,y,z,t;
    int Ix,Iy,Iz,It;
    int PXU,PYU,PZU,PTU;
    float convVal;
    Point Pdims = inputKernel.getDims();
    Point centre = inputKernel.getCentre();
    ScalarField * copyImage = this->image;

    pX = (int) Pdims(1);
    pY = (int) Pdims(2);
    if(Pdims.getDims()>2){pZ = (int) Pdims(3);}else{pZ = 1;}
    if(Pdims.getDims()>3){pT = (int) Pdims(4);}else{pT = 1;}

    //Add symmetric padding for convolution boundary conditions
    copyImage->padArraySymmetric(pX,pY,pZ,pT);
    std::cout << "Done Padding" << std::endl;
    Ix = (int) copyImage->getDim(1);
    Iy = (int) copyImage->getDim(2);
    Iz = (int) copyImage->getDim(3);
    It = (int) copyImage->getDim(4);

    PXU = (pX - 1)/2;
    PYU = (pY - 1)/2;
    PZU = (pZ - 1)/2;
    PTU = (pT - 1)/2;

    #pragma omp parallel for
    for(l=pT;l<It-pT;l++){
    for(k=pZ;k<Iz-pZ;k++){
    for(j=pY;j<Iy-pY;j++){
    for(i=pX;i<Ix-pX;i++){
        convVal = 0;

        for(t=-PTU;t<=PTU;t++){
        for(z=-PZU;z<=PZU;z++){
        for(y=-PYU;y<=PYU;y++){
        for(x=-PXU;x<=PXU;x++){
            convVal += copyImage->get(i+x, j+y, k+z, l+t)*inputKernel.getValue(x+PXU,y+PYU,z+PZU,t+PTU);
        }}}}

        copyImage->set(convVal,i,j,k,l);

    }}}}
    std::cout << "Done Convolving" << std::endl;

    copyImage->unpadArray(pX, pY, pZ, pT);
    std::cout << "Done Un-Padding" << std::endl;
    this->image = copyImage;

}


//-------------Class: FFTFilter ---------------------------------------------------------------------------------------------------------------------------------

FFTFilter::FFTFilter(){

}

FFTFilter::~FFTFilter(){

}

void FFTFilter::InitialiseConvolver(void){

    this->imageX = image->NX;
    this->imageY = image->NY;
    this->imageZ = image->NZ;

    this->NXfft=(int)(pow(2.,floor((log((double)this->imageX)/log(2.))+0.99999))+0.00001);
    this->NYfft=(int)(pow(2.,floor((log((double)this->imageY)/log(2.))+0.99999))+0.00001);
    this->NZfft=(int)(pow(2.,floor((log((double)this->imageZ)/log(2.))+0.99999))+0.00001);

    this->RealSignalForFFT.CreateVoidField(this->NXfft, this->NYfft, this->NZfft); //image  - real part
    this->ImagSignalForFFT.CreateVoidField(this->NXfft, this->NYfft, this->NZfft); //image  - imaginary part
    this->RealFilterForFFT.CreateVoidField(this->NXfft, this->NYfft, this->NZfft); //filter - real part
    this->ImagFilterForFFT.CreateVoidField(this->NXfft, this->NYfft, this->NZfft); //filter - imaginary part

    this->ImageTemp.CreateVoidField(this->NXfft,this->NYfft,this->NZfft);

    this->AdjustKernelForFFT();


}

void FFTFilter::AdjustKernelForFFT(){

    int kernelX,kernelY,kernelZ;
    int centreX,centreY,centreZ;
    kernelX = loadedKernel.getDims(1);
    kernelY = loadedKernel.getDims(2);
    kernelZ = loadedKernel.getDims(3);
    centreX = (int)((kernelX + 1)/2);
    centreY = (int)((kernelY + 1)/2);
    centreZ = (int)((kernelZ + 1)/2);

    int x,y,z;

    for(x=0;x<centreX;x++) for(y=0;y<centreY;y++) for(z=0;z<centreZ;z++) this->RealFilterForFFT.set(loadedKernel.getValue(centreX + x,centreY + y,centreZ + z),x,y,z);
    for(x=centreX;x<kernelX;x++) for(y=0;y<centreY;y++) for(z=0;z<centreZ;z++) this->RealFilterForFFT.set(loadedKernel.getValue(x-centreX,centreY + y,centreZ + z),NXfft - x + centreX,y,z);

    for(x=0;x<centreX;x++) for(y=centreY;y<kernelY;y++) for(z=0;z<centreZ;z++) this->RealFilterForFFT.set(loadedKernel.getValue(centreX + x,y-centreY,centreZ + z),x,NYfft - y + centreY,z);
    for(x=centreX;x<kernelX;x++) for(y=centreY;y<kernelY;y++) for(z=0;z<centreZ;z++) this->RealFilterForFFT.set(loadedKernel.getValue(x-centreX,y-centreY,centreZ + z),NXfft - x + centreX,NYfft - y + centreY,z);

    for(x=0;x<centreX;x++) for(y=0;y<centreY;y++) for(z=centreZ;z<kernelZ;z++) this->RealFilterForFFT.set(loadedKernel.getValue(centreX + x,centreY + y,z-centreZ),x,y,NZfft - z + centreZ);
    for(x=centreX;x<kernelX;x++) for(y=0;y<centreY;y++) for(z=centreZ;z<kernelZ;z++) this->RealFilterForFFT.set(loadedKernel.getValue(x-centreX,centreY + y,z-centreZ),NXfft - x + centreX,y,NZfft - z + centreZ);

    for(x=0;x<centreX;x++) for(y=centreY;y<kernelY;y++) for(z=centreZ;z<kernelZ;z++) this->RealFilterForFFT.set(loadedKernel.getValue(centreX + x,y-centreY,z-centreZ),x,NYfft - y + centreY,NZfft - z + centreZ);
    for(x=centreX;x<kernelX;x++) for(y=centreY;y<kernelY;y++) for(z=centreZ;z<kernelZ;z++) this->RealFilterForFFT.set(loadedKernel.getValue(x-centreX,y-centreY,z-centreZ),NXfft - x + centreX,NYfft - y + centreY,NZfft - z + centreZ);

    this->DirectFFT(&this->RealFilterForFFT,&this->ImagFilterForFFT);

}

///Fast Fourier Transform of numerical recipies (slighly modified)
void FFTFilter::FFT1D(float *data, unsigned long nn, int isign){
    unsigned long n,mmax,m,j,istep,i;
    double wtemp,wr,wpr,wpi,wi,theta;
    float tempr,tempi;

    n=nn << 1;
    j=1;
    for (i=1;i<n;i+=2){
        if (j>i){
            tempr=data[j]; data[j]=data[i]; data[i]=tempr;
            tempr=data[j+1]; data[j+1]=data[i+1]; data[i+1]=tempr;
        }
        m=n >> 1;
        while ((m>=2) && (j>m)){
            j -= m;
            m >>= 1;
        }
        j += m;
    }
    mmax=2;
    while (n > mmax) {
        istep=mmax << 1;
        theta=isign*(6.28318530717959/mmax);
        wtemp=sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi=sin(theta);
        wr=1.0;
        wi=0.0;
        for (m=1;m<mmax;m+=2) {
            for (i=m;i<=n;i+=istep) {
                j=i+mmax;
                tempr=wr*data[j]-wi*data[j+1];
                tempi=wr*data[j+1]+wi*data[j];
                data[j]=data[i]-tempr;
                data[j+1]=data[i+1]-tempi;
                data[i] += tempr;
                data[i+1] += tempi;
            }
            wr=(wtemp=wr)*wpr-wi*wpi+wr;
            wi=wi*wpr+wtemp*wpi+wi;
        }
        mmax=istep;
    }
}



#ifdef COMPILE_WITH_OPENMP

///Fast Fourier Transform
void FFTFilter::DirectFFT(ScalarField * RealSignal,ScalarField * ImaginarySignal){
  int SizeX,SizeY,SizeZ;
  float SqrtSizeX,SqrtSizeY,SqrtSizeZ;
  int x,y,z;
  float * dataX;
  int MaxSizeXSizeYSizeZ;

  //1) extract the size of the images
  SizeX=RealSignal->NX;
  SizeY=RealSignal->NY;
  SizeZ=RealSignal->NZ;

  MaxSizeXSizeYSizeZ=SizeX;
  if (SizeY>MaxSizeXSizeYSizeZ) MaxSizeXSizeYSizeZ=SizeY;
  if (SizeZ>MaxSizeXSizeYSizeZ) MaxSizeXSizeYSizeZ=SizeZ;

  SqrtSizeX=static_cast<float>(sqrt(static_cast<double>(SizeX)));
  SqrtSizeY=static_cast<float>(sqrt(static_cast<double>(SizeY)));
  SqrtSizeZ=static_cast<float>(sqrt(static_cast<double>(SizeZ)));

  //BEGIN FORK FOR THREADS
  #pragma omp parallel default(shared) private(x,y,z,dataX)
  {
    dataX = new float [MaxSizeXSizeYSizeZ*2+1];

    //2) perform the fft along x axis
    #pragma omp for
    for (y = 0; y < SizeY; y++){
      for (z = 0; z < SizeZ; z++){
        for (x = 0; x < SizeX; x++){
          dataX[2*x+1]=RealSignal->get(x, y, z);
          dataX[2*x+2]=ImaginarySignal->get(x, y, z);
        }
        FFT1D(dataX, (unsigned long)SizeX, 1);
        for (x = 0; x < SizeX; x++){
          RealSignal->set(dataX[2*x+1]/SqrtSizeX, x, y, z);
          ImaginarySignal->set(dataX[2*x+2]/SqrtSizeX, x, y, z);
        }
      }
    }

    //3) perform the fft along y axis
    #pragma omp for
    for (x = 0; x < SizeX; x++){
      for (z = 0; z < SizeZ; z++) {
        for (y = 0; y < SizeY; y++){
          dataX[2*y+1]=RealSignal->get(x, y, z);
          dataX[2*y+2]=ImaginarySignal->get(x, y, z);
        }
        FFT1D(dataX, (unsigned long)SizeY, 1);
        for (y = 0; y < SizeY; y++){
          RealSignal->set(dataX[2*y+1]/SqrtSizeY,x, y, z);
          ImaginarySignal->set(dataX[2*y+2]/SqrtSizeY, x, y, z);
        }
      }
    }

    //4) perform the fft along z axis
    #pragma omp for
    for (y = 0; y < SizeY; y++){
      for (x = 0; x < SizeX; x++){
        for (z = 0; z < SizeZ; z++){
          dataX[2*z+1]=RealSignal->get(x, y, z);
          dataX[2*z+2]=ImaginarySignal->get(x, y, z);
        }
        FFT1D(dataX, (unsigned long)SizeZ, 1);
        for (z = 0; z < SizeZ; z++){
          RealSignal->set(dataX[2*z+1]/SqrtSizeZ,x, y, z);
          ImaginarySignal->set(dataX[2*z+2]/SqrtSizeZ, x, y, z);
        }
      }
    }
        std::cout << "Done 1D FFTs" << std::endl;
    delete dataX;
  //END FORK FOR THREADS
  }

}

#else

///Fast Fourier Transform
void FFTFilter::DirectFFT(ScalarField * RealSignal,ScalarField * ImaginarySignal){
    int SizeX,SizeY,SizeZ;
    float SqrtSizeX,SqrtSizeY,SqrtSizeZ;
    int x,y,z;
    float * dataX;
    float * dataY;
    float * dataZ;

    //1) extract the size of the images
    SizeX=RealSignal->NX;
    SizeY=RealSignal->NY;
    SizeZ=RealSignal->NZ;

    SqrtSizeX=static_cast<float>(sqrt(static_cast<double>(SizeX)));
    SqrtSizeY=static_cast<float>(sqrt(static_cast<double>(SizeY)));
    SqrtSizeZ=static_cast<float>(sqrt(static_cast<double>(SizeZ)));


    //2) perform the fft along x axis
    dataX = new float [SizeX*2+1];
    for (z = 0; z < SizeZ; z++) for (y = 0; y < SizeY; y++){
            for (x = 0; x < SizeX; x++){
                dataX[2*x+1]=RealSignal->get(x, y, z);
                dataX[2*x+2]=ImaginarySignal->get(x, y, z);
            }
            FFT1D(dataX, (unsigned long)SizeX, 1);
            for (x = 0; x < SizeX; x++){
                RealSignal->set(dataX[2*x+1]/SqrtSizeX, x, y, z);
                ImaginarySignal->set(dataX[2*x+2]/SqrtSizeX, x, y, z);
            }
        }
    delete dataX;

    //3) perform the fft along y axis
    dataY = new float [SizeY*2+1];
    for (z = 0; z < SizeZ; z++) for (x = 0; x < SizeX; x++){
            for (y = 0; y < SizeY; y++){
                dataY[2*y+1]=RealSignal->get(x, y, z);
                dataY[2*y+2]=ImaginarySignal->get(x, y, z);
            }
            FFT1D(dataY, (unsigned long)SizeY, 1);
            for (y = 0; y < SizeY; y++){
                RealSignal->set(dataY[2*y+1]/SqrtSizeY,x, y, z);
                ImaginarySignal->set(dataY[2*y+2]/SqrtSizeY, x, y, z);
            }
        }
    delete dataY;


    //4) perform the fft along z axis
    dataZ = new float [SizeZ*2+1];
    for (y = 0; y < SizeY; y++) for (x = 0; x < SizeX; x++){
            for (z = 0; z < SizeZ; z++){
                dataZ[2*z+1]=RealSignal->get(x, y, z);
                dataZ[2*z+2]=ImaginarySignal->get(x, y, z);
            }
            FFT1D(dataZ, (unsigned long)SizeZ, 1);
            for (z = 0; z < SizeZ; z++){
                RealSignal->set(dataZ[2*z+1]/SqrtSizeZ,x, y, z);
                ImaginarySignal->set(dataZ[2*z+2]/SqrtSizeZ, x, y, z);
            }
        }
    delete dataZ;
}

#endif



#ifdef COMPILE_WITH_OPENMP


///Inverse Fast Fourier Transform
void FFTFilter::InverseFFT(ScalarField * RealSignal,ScalarField * ImaginarySignal){
  int SizeX,SizeY,SizeZ;
  float SqrtSizeX,SqrtSizeY,SqrtSizeZ;
  int x,y,z;
  float * dataX;
  int MaxSizeXSizeYSizeZ;

  //1) extract the size of the images
  SizeX=RealSignal->NX;
  SizeY=RealSignal->NY;
  SizeZ=RealSignal->NZ;

  MaxSizeXSizeYSizeZ=SizeX;
  if (SizeY>MaxSizeXSizeYSizeZ) MaxSizeXSizeYSizeZ=SizeY;
  if (SizeZ>MaxSizeXSizeYSizeZ) MaxSizeXSizeYSizeZ=SizeZ;

  SqrtSizeX=static_cast<float>(sqrt(static_cast<double>(SizeX)));
  SqrtSizeY=static_cast<float>(sqrt(static_cast<double>(SizeY)));
  SqrtSizeZ=static_cast<float>(sqrt(static_cast<double>(SizeZ)));

  //BEGIN FORK FOR THREADS
  #pragma omp parallel default(shared) private(x,y,z,dataX)
  {
    dataX = new float [MaxSizeXSizeYSizeZ*2+1];

    //2) perform the ifft along z axis
    #pragma omp for
    for (y = 0; y < SizeY; y++){
      for (x = 0; x < SizeX; x++){
        for (z = 0; z < SizeZ; z++){
          dataX[2*z+1]=RealSignal->get(x, y, z);
          dataX[2*z+2]=ImaginarySignal->get(x, y, z);
        }
        FFT1D(dataX, (unsigned long)SizeZ, -1);
        for (z = 0; z < SizeZ; z++){
          RealSignal->set(dataX[2*z+1]/SqrtSizeZ, x, y, z);
          ImaginarySignal->set(dataX[2*z+2]/SqrtSizeZ,x, y, z);
        }
      }
    }

    //3) perform the ifft along y axis
    #pragma omp for
    for (x = 0; x < SizeX; x++){
      for (z = 0; z < SizeZ; z++){
        for (y = 0; y < SizeY; y++){
          dataX[2*y+1]=RealSignal->get(x, y, z);
          dataX[2*y+2]=ImaginarySignal->get(x, y, z);
        }
        FFT1D(dataX, (unsigned long)SizeY, -1);
        for (y = 0; y < SizeY; y++){
          RealSignal->set(dataX[2*y+1]/SqrtSizeY, x, y, z);
          ImaginarySignal->set(dataX[2*y+2]/SqrtSizeY, x, y, z);
        }
      }
    }

    //4) perform the ifft along x axis
    #pragma omp for
    for (y = 0; y < SizeY; y++){
      for (z = 0; z < SizeZ; z++){
        for (x = 0; x < SizeX; x++){
          dataX[2*x+1]=RealSignal->get(x, y, z);
          dataX[2*x+2]=ImaginarySignal->get(x, y, z);
        }
        FFT1D(dataX, (unsigned long)SizeX, -1);
        for (x = 0; x < SizeX; x++){
          RealSignal->set(dataX[2*x+1]/SqrtSizeX, x, y, z);
          ImaginarySignal->set(dataX[2*x+2]/SqrtSizeX,x, y, z);
        }
      }
    }

    delete dataX;
  //END FORK FOR THREADS
  }
}


#else

///Inverse Fast Fourier Transform
void FFTFilter::InverseFFT(ScalarField * RealSignal,ScalarField * ImaginarySignal){
    int SizeX,SizeY,SizeZ;
    float SqrtSizeX,SqrtSizeY,SqrtSizeZ;
    int x,y,z;
    float * dataX;
    float * dataY;
    float * dataZ;

    //1) extract the size of the images
    SizeX=RealSignal->NX;
    SizeY=RealSignal->NY;
    SizeZ=RealSignal->NZ;

    SqrtSizeX=static_cast<float>(sqrt(static_cast<double>(SizeX)));
    SqrtSizeY=static_cast<float>(sqrt(static_cast<double>(SizeY)));
    SqrtSizeZ=static_cast<float>(sqrt(static_cast<double>(SizeZ)));


    //2) perform the ifft along z axis
    dataZ = new float [SizeZ*2+1];
    for (y = 0; y < SizeY; y++) for (x = 0; x < SizeX; x++){
            for (z = 0; z < SizeZ; z++){
                dataZ[2*z+1]=RealSignal->get(x, y, z);
                dataZ[2*z+2]=ImaginarySignal->get(x, y, z);
            }
            FFT1D(dataZ, (unsigned long)SizeZ, -1);
            for (z = 0; z < SizeZ; z++){
                RealSignal->set(dataZ[2*z+1]/SqrtSizeZ, x, y, z);
                ImaginarySignal->set(dataZ[2*z+2]/SqrtSizeZ,x, y, z);
            }
        }
    delete dataZ;

    //3) perform the ifft along y axis
    dataY = new float [SizeY*2+1];
    for (z = 0; z < SizeZ; z++) for (x = 0; x < SizeX; x++){
            for (y = 0; y < SizeY; y++){
                dataY[2*y+1]=RealSignal->get(x, y, z);
                dataY[2*y+2]=ImaginarySignal->get(x, y, z);
            }
            FFT1D(dataY, (unsigned long)SizeY, -1);
            for (y = 0; y < SizeY; y++){
                RealSignal->set(dataY[2*y+1]/SqrtSizeY, x, y, z);
                ImaginarySignal->set(dataY[2*y+2]/SqrtSizeY, x, y, z);
            }
        }
    delete dataY;

    //4) perform the ifft along x axis
    dataX = new float [SizeX*2+1];
    for (z = 0; z < SizeZ; z++) for (y = 0; y < SizeY; y++){
            for (x = 0; x < SizeX; x++){
                dataX[2*x+1]=RealSignal->get(x, y, z);
                dataX[2*x+2]=ImaginarySignal->get(x, y, z);
            }
            FFT1D(dataX, (unsigned long)SizeX, -1);
            for (x = 0; x < SizeX; x++){
                RealSignal->set(dataX[2*x+1]/SqrtSizeX, x, y, z);
                ImaginarySignal->set(dataX[2*x+2]/SqrtSizeX,x, y, z);
            }
        }
    delete dataX;
}

#endif


///convolution of a 3D scalar field using the predifined kernel
void FFTFilter::ConvolveWithKernel(void){
    int x,y,z;
    float a,b,c,d;
    float CoefMult;

    //1) Copy the orginal image in the image that will be transformed
    for (z=0;z<this->NZfft;z++) for (y=0;y<this->NYfft;y++) for (x=0;x<this->NXfft;x++) this->RealSignalForFFT.set(0.,x,y,z);
    for (z=0;z<this->NZfft;z++) for (y=0;y<this->NYfft;y++) for (x=0;x<this->NXfft;x++) this->ImagSignalForFFT.set(0.,x,y,z);

    for (z = 0; z < image->NZ; z++) for (y = 0; y < image->NY; y++) for (x = 0; x < image->NX; x++) this->RealSignalForFFT.set(image->get(x,y,z),x,y,z);


    //2) Transform RealSignalForFFT and ImagSignalForFFT in Fourier spaces
    this->DirectFFT(&this->RealSignalForFFT,&this->ImagSignalForFFT);
    std::cout << "Done forward FFT" << std::endl;

    //3) filtering in Fourier spaces
    CoefMult=(float)(sqrt((double)this->RealSignalForFFT.NX)*sqrt((double)this->RealSignalForFFT.NY)*sqrt((double)this->RealSignalForFFT.NZ));

    for (z = 0; z < this->RealSignalForFFT.NZ; z++) for (y = 0; y < this->RealSignalForFFT.NY; y++) for (x = 0; x < this->RealSignalForFFT.NX; x++){
                a=this->RealSignalForFFT.get(x, y, z);
                b=this->ImagSignalForFFT.get(x, y, z);
                c=this->RealFilterForFFT.get(x, y, z)*CoefMult;
                d=this->ImagFilterForFFT.get(x, y, z)*CoefMult;

                this->RealSignalForFFT.set(a*c-b*d, x, y, z);
                this->ImagSignalForFFT.set(c*b+a*d,x, y, z);
            }

    std::cout<< "Done Fourier space multiplication" << std::endl;

    //4) IFFT
    this->InverseFFT(&this->RealSignalForFFT,&this->ImagSignalForFFT);

    std::cout << "Done inverse FFT" << std::endl;

    //5) Copy the image that has been convolved in the orginal image
    for (z = 0; z < image->NZ; z++) for (y = 0; y < image->NY; y++) for (x = 0; x < image->NX; x++){
                image->set(this->RealSignalForFFT.get(x,y,z),x,y,z);
            }

    std::cout << "Finished" << std::endl;
}



//-----------------------Class 6: SuperVoxels -----------------------------------------------------------

SuperVoxel::SuperVoxel(void){
    bConnectivity = 0;
    bIndices = 0;
}

SuperVoxel::~SuperVoxel(void){

    for(int i=0;i<nSuperVoxels;i++){
        if(bIndices){
            delete[] indices[i];
        }
    }
    if(bIndices){
        delete[] indices;
    }


    delete[] svSizes;
}

void SuperVoxel::readSegmentationImage(ScalarField inputImage) {

    int i,val;
    float minVal = inputImage.getMin();
    float maxVal = inputImage.getMax();

    bIndices = 1;

    inputImage = inputImage - inputImage.getMin();

    nSuperVoxels = (int)inputImage.getMax() + 1;

    int **indices = new int *[nSuperVoxels];

    int *sizeCount = new int[nSuperVoxels];
    int *svSizes = new int[nSuperVoxels];

    for(i=0;i<nSuperVoxels;i++){
        sizeCount[i] = 0;
        svSizes[i] = 0;
    }


    for(i=0;i<inputImage.getDim(1)*inputImage.getDim(2)*inputImage.getDim(3);i++){
        val = (int)inputImage.get(i);
        sizeCount[val] += 1;
        svSizes[val] +=1;
    }



   for(i=0;i<nSuperVoxels;i++){
       indices[i] = new int[sizeCount[i]];
       sizeCount[i] = 0;
   }

    for(i=0;i<inputImage.getDim(1)*inputImage.getDim(2)*inputImage.getDim(3);i++){
        val = (int)inputImage.get(i);
        indices[val][sizeCount[val]] = i;
        sizeCount[val] += 1;
    }

    this->indices = indices;
    this->svSizes = svSizes;

    delete(sizeCount);
}

int *SuperVoxel::getSuperVoxelIndices(int i){

    return this->indices[i];

}

int SuperVoxel::getNumSuperVoxels(void) {

    return nSuperVoxels;

}

int SuperVoxel::getSizeSuperVoxel(int i){
    return svSizes[i];
}

void SuperVoxel::getConnectivity(ScalarField inputImage, int **listOne, int **listTwo, int *nConn, int **sizeInterface){

    int i,j,x,y,z;
    int val;
    int vUp,vRight,vFront;
    bConnectivity = 1;
    std::pair<int,int> p,pt;



    for(z=1;z<inputImage.getDim(3)-1;z++) {
    for (y = 1; y < inputImage.getDim(2) - 1; y++) {
    for (x = 1; x < inputImage.getDim(1) - 1; x++) {
        val = (int)inputImage.get(x,y,z);
        vUp = (int)inputImage.get(x,y,z+1);
        vRight = (int)inputImage.get(x,y+1,z);
        vFront = (int)inputImage.get(x+1,y,z);
        if(val != vUp) {
            p.first = val;
            p.second = vUp;
            pt.first = vUp;
            pt.second = val;
            if (connectivity[p] == 0) {
                connectivity[p] = 1;
                connectivity[pt] = 1;
            }
            else
            {

            connectivity[p]++;
            connectivity[pt]++;
            }
        }

        if(val != vRight){
            p.first = val;
            p.second = vRight;
            pt.first = vRight;
            pt.second = val;
            if(connectivity[p] == 0)
            {
                connectivity[p] = 1;
                connectivity[pt] = 1;
            }
            else
            {
                connectivity[p]++;
                connectivity[pt]++;
            }
        }

        if(val != vFront){
            p.first = val;
            p.second = vFront;
            pt.first = vFront;
            pt.second = val;
            if(connectivity[p] == 0)
            {
                connectivity[p] = 1;
                connectivity[pt] = 1;
            }
            else
            {
                connectivity[p]++;
                connectivity[pt]++;
            }
        }

    }}}


    int iter = 0;

    iter = (int)connectivity.size() - 1;

    int *listOneTmp;
    int *listTwoTmp;
    int *listInterface;

    listOneTmp = new int[iter];
    listTwoTmp = new int[iter];
    listInterface = new int[iter];

    typedef std::map<std::pair<int,int>,int>::iterator it_type;

    int iter2 = 0;

    for(it_type iterator = connectivity.begin();iterator != connectivity.end();iterator++){
        listOneTmp[iter2] = iterator->first.first;
        listTwoTmp[iter2] = iterator->first.second;
        listInterface[iter2] = iterator->second;
        iter2 += 1;
    }


    *listOne = listOneTmp;
    *listTwo = listTwoTmp;
    *sizeInterface = listInterface;

    *nConn =iter2;

}



#ifndef ELEM_SWAP(a,b)
#define ELEM_SWAP(a,b) { register float t=(a);(a)=(b);(b)=t; }

float kth_smallest(float a[], uint16_t n, uint16_t k)
{
	uint64_t i,j,l,m ;
	float x ;
	l=0 ; m=n-1 ;
	while (l<m) {
		x=a[k] ;
		i=l ;
		j=m ;
		do {
			while (a[i]<x) i++ ;
			while (x<a[j]) j-- ;
			if (i<=j) {
				ELEM_SWAP(a[i],a[j]) ;
				i++ ; j-- ;
			}
		} while (i<=j) ;
		if (j<k) l=i ;
		if (k<i) m=j ;
	}
	return a[k] ;
}

#define wirth_median(a,n) kth_smallest(a,n,(((n)&1)?((n)/2):(((n)/2)-1)))

//===================== Method 2: =============================================
//This is the faster median determination method.
//Algorithm from Numerical recipes in C of 1992

float quick_select_median(float arr[], uint16_t n)
{
	uint16_t low, high ;
	uint16_t median;
	uint16_t middle, ll, hh;
	low = 0 ; high = n-1 ; median = (low + high) / 2;
	for (;;) {
		if (high <= low) /* One element only */
			return arr[median] ;
		if (high == low + 1) { /* Two elements only */
			if (arr[low] > arr[high])
			ELEM_SWAP(arr[low], arr[high]) ;
			return arr[median] ;
		}
		/* Find median of low, middle and high items; swap into position low */
		middle = (low + high) / 2;
		if (arr[middle] > arr[high])
		ELEM_SWAP(arr[middle], arr[high]) ;
		if (arr[low] > arr[high])
		ELEM_SWAP(arr[low], arr[high]) ;
		if (arr[middle] > arr[low])
		ELEM_SWAP(arr[middle], arr[low]) ;
		/* Swap low item (now in position middle) into position (low+1) */
		ELEM_SWAP(arr[middle], arr[low+1]) ;
		/* Nibble from each end towards middle, swapping items when stuck */
		ll = low + 1;
		hh = high;
		for (;;) {
			do ll++; while (arr[low] > arr[ll]) ;
			do hh--; while (arr[hh] > arr[low]) ;
			if (hh < ll)
				break;
			ELEM_SWAP(arr[ll], arr[hh]) ;
		}
		/* Swap middle item (in position low) back into correct position */
		ELEM_SWAP(arr[low], arr[hh]) ;
		/* Re-set active partition */
		if (hh <= median)
			low = ll;
		if (hh >= median)
			high = hh - 1;
	}
	return arr[median] ;
}
#endif

