#ifndef TMVA_SOFIE_ROPERATOR
#define TMVA_SOFIE_ROPERATOR

#include <vector>

#include "SOFIE_common.hxx"

#include "onnx.pb.h"



namespace TMVA{
namespace Experimental{
namespace SOFIE{

class ROperator{

public:
   virtual const std::vector<std::vector<int_t>> shapeInference() = 0;
   virtual void Forward_reference() = 0;
   virtual ~ROperator(){}
};

class RGraph;

namespace INTERNAL{
extern ROperator* make_ROperator_Gemm(const onnx::NodeProto& nodeproto, RGraph& this_graph);
extern ROperator* make_ROperator_Transpose(const onnx::NodeProto& nodeproto, RGraph& this_graph);
extern ROperator* make_ROperator_Relu(const onnx::NodeProto& nodeproto, RGraph& this_graph);

using factoryMethodMap = std::unordered_map<std::string, ROperator* (*)(const onnx::NodeProto&, RGraph&)>;
const factoryMethodMap mapOptypeOperator = {
      {"Gemm", &make_ROperator_Gemm},
      {"Transpose", &make_ROperator_Transpose},
      {"Relu", &make_ROperator_Relu}
   };
}


}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_OPERATOR
