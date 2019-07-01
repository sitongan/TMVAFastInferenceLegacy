#ifndef TMVA_SOFIE
#define TMVA_SOFIE

#include <cstddef>
#include <vector>
#include <string>
#include <unordered_map>
#include <tuple>
#include <array>
#include <unordered_set>
#include <queue>
#include <deque>
#include <cstdint> //int64_t: used for ONNX IR
#include <cstddef> // std::size_t
#include <cassert>

#include "onnx.pb.h"

namespace TMVA{
namespace Experimental{
namespace SOFIE{   //Project code. System for Open, Fast Inference and Evaluation

   typedef std::int64_t int_t;   //int64 default int used in onnx.pb.h

   enum class ETensorType{
      UNDEFINED = 0, FLOAT = 1, UNINT8 = 2, INT8 = 3, UINT16 = 4, INT16 = 5, INT32 = 6, INT64 = 7, STRING = 8, BOOL = 9, //order sensitive
       FLOAT16 = 10, DOUBLE = 11, UINT32 = 12, UINT64 = 13, COMPLEX64 = 14, COMPLEX28 = 15, BFLOAT16 = 16
   };
   //Supported:FLOAT

   class ROperator;
   class RGraph;


   namespace INTERNAL{
   extern ROperator* make_ROperator_Gemm(const onnx::NodeProto& nodeproto, RGraph& this_graph);
   using factoryMethodMap = std::unordered_map<std::string, ROperator* (*)(const onnx::NodeProto&, RGraph&)>;
   extern factoryMethodMap mapOptypeOperator;
   }



   class RModel{
   public:
   private:
      int64_t fIRVersion;
      std::vector<std::tuple<std::string, int64_t>> fOpsetVersion;
      int64_t fModelVersion;
      std::map<std::string, std::string> fMetadata;
         //expected keys: "producer_name", "producer_version", "domain", "doc_string", "model_author", "model_license"
      //RGraph fGraph; //main graph to be executed for the model
   };

namespace INTERNAL{

void check_init_assert();

std::string print_nodelist(const std::unordered_set<int_t>& vec, const onnx::GraphProto& graph);

std::vector<int_t> topological_sort(const std::map<int_t, std::unordered_set<int_t>>& EdgesForward,
                                    const std::map<int_t, std::unordered_set<int_t>>& EdgesBackward);




}

}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE
