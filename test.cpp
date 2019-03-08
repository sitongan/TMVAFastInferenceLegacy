#include <iostream>
#include <vector>
#include "TMVA/RTensor.hxx"

using namespace std;
using TMVA::Experimental::RTensor;

int main(){
   //float data[] = {1, 2, 3, 4, 5, 6};
   vector<float> data {1, 2, 3, 4, 5, 6};
   RTensor<float> x(data.data(), {2, 3});
   std::cout << x << std::endl;

   return 0;
}
