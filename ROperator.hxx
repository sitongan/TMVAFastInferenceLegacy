#ifndef TMVA_SOFIE_ROPERATOR
#define TMVA_SOFIE_ROPERATOR

#include <vector>

#include "SOFIE.hxx"
#include "RGraph.hxx"



namespace TMVA{
namespace Experimental{
namespace SOFIE{

class RGraph;
class ROperator;

class ROperator{

public:
   virtual const std::vector<int_t> outputShape() = 0;
   virtual void forward_fallback() = 0;

};


}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_OPERATOR
