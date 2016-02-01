 #include <RBMedImLib.h>
 
 #include <stdio.h>
 #include <float.h>
 #include "omp.h"
 #include <algorithm>
 #include "/local/newc4330/Documents/Libraries/fftw-3.3.4/api/fftw3.h"
 #include <map>

 
 int main(void){
 
     std::cout << "Hello World!\n";

     ScalarField Image1;
     ScalarField Image2;
     ScalarField OutputImage;

     Image1.readNifti("/data/Data/TestData/VesselSegmentation/SLIC_Supervoxels.nii");
     Image2.readNifti("/data/Data/TestData/VesselSegmentation/Difference_image.nii");



     SuperVoxel SV;
     int *listOne;
     int *listTwo;
     int *interfaceList;
     int nConn;

     SV.readSegmentationImage(Image1);
     SV.getConnectivity(Image1,&listOne,&listTwo,&nConn, &interfaceList);

     int i;
//     for(i=0;i<SV.getNumSuperVoxels();i++){
//         std::cout << SV.getSizeSuperVoxel(i)<< std::endl;
//     }


     std::cout << Image2.getMin() << std::endl;
     std::cout << Image2.getMax() << std::endl;

     float *test = new float[2];
     test[0] = 0;
     test[1] = 1;

     int N = SV.getNumSuperVoxels();
     int sizes = SV.getSizeSuperVoxel(N-1);
     int *indexes;

     std::cout << sizes << std::endl;
     std::cout << N << std::endl;

     Image2 = Image2 - Image2.getMin();
     int j;
     float median,median2;

     indexes = SV.getSuperVoxelIndices(N-1);

     for(i=0;i<SV.getNumSuperVoxels();i++){
         median = Image2.getRegionMedian(SV.getSuperVoxelIndices(i),SV.getSizeSuperVoxel(i));
         if(median > 0) {
             std::cout << "Found supervoxel: " << i << std::endl;

             std::cout << "Median Value: " << median << std::endl;
             int boldCount = 0;
             for (j = 0; j < nConn; j++) {
                 if (listOne[j] == i) {
                     median2 = Image2.getRegionMedian(SV.getSuperVoxelIndices(listTwo[j]), SV.getSizeSuperVoxel(listTwo[j]));
                     std::cout << "Found Neighbour: " << j << std::endl;
                     std::cout << "Median Value: " << median2 << std::endl;
                     if(median2 > 0){
                         boldCount += 1;
                     }
                 }
             }
             std::cout << "Number of bold neighbours: " << boldCount << std::endl;
             std::cout << "\n \n \n" << std::endl;
         }
     }

     return 0;
  }
