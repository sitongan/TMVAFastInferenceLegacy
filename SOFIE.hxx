#ifndef TMVA_SOFIE
#define TMVA_SOFIE

#include <cstddef>
#include <vector>
#include <string>
#include <map>
#include <tuple>
#include <array>
#include <set>
#include <queue>
#include <deque>
#include <cstdint> //int64_t: used for ONNX IR
#include <cstddef> // std::size_t

#include "onnx.pb.h"

namespace TMVA{
namespace Experimental{
namespace SOFIE{   //Project code. System for Open, Fast Inference and Evaluation

   typedef std::int64_t int_t;   //int64 default int used in ONNX IR

   enum class EAttributeType{
      UNDEFINED, FLOAT, INT, STRING, TENSOR, GRAPH, FLOATS, INTS, STRINGS, TENSORS, GRAPHS   //order sensitive
   };

   enum class ETensorType{
      UNDEFINED, FLOAT, UNINT8, INT8, UINT16, INT16, INT32, INT64, STRING, BOOL, FLOAT16, DOUBLE,  //order sensitive
      UINT32, UINT64, COMPLEX64, COMPLEX28, BFLOAT16
   };

   class RDataNode{

   protected:
      std::string fName;
      std::vector<int_t> fShape;
      ETensorType fType;
      bool fIsSegment;
      std::tuple<int_t, int_t> fSegmentIndex;
      bool fHasData;

   public:

      const onnx::TensorProto& fTensorProto; //TEMP, for data access
      RDataNode(const onnx::TensorProto& tensorproto);
   };


   template<typename T>
   class RDataNodeConcrete: public RDataNode{
   private:
      std::vector<T> fData;
   public:
      T* GetData() {return fData.data();}       //return std::std::vector.data()
   };











   class RAttribute{ // interface for attributes
   public:
      template<typename T>
      T& GetValue(){
            //return (RAttribute<T> *)(this)->fValue;
      }
      // in the operator functions:
      //      Gemm(const RAttribute& att1){
      //         att1.GetValue<int>
      //      }
   protected:
      std::string fName;
      std::size_t fSize;
      EAttributeType fType;

   };

   template<typename T>
   class RAttributeData: public RAttribute{
         T fValue;
   };
   //template specialisation for list of attributes?

   class ROpNode{
   public:


      //temp
      const onnx::NodeProto& fNodeProto; //for data access

      ROpNode(const onnx::NodeProto& nodeproto): fNodeProto(nodeproto) {}

   //private:
      std::vector<RDataNode*> fWeights;
      std::vector<RDataNode*> fInputs;
      std::vector<RDataNode*> fOutputs;
      std::vector<RAttribute*> fAttributes;
      std::string fOperator;
      //todo: map from string to function pointers for the operator
   };


   class RGraph{
   public:

   //private:
      std::string fName;
      std::map<int_t, std::vector<int_t>> fEdgesForward;
      std::map<int_t, std::vector<int_t>> fEdgesBackward;
      std::vector<RDataNode> fDataNodes;
      std::vector<ROpNode> fOpNodes;
   };


   class RModel{
   public:
   private:
      int64_t fIRVersion;
      std::vector<std::tuple<std::string, int64_t>> fOpsetVersion;
      int64_t fModelVersion;
      std::map<std::string, std::string> fMetadata;
         //expected keys: "producer_name", "producer_version", "domain", "doc_string", "model_author", "model_license"
      RGraph fGraph; //main graph to be executed for the model
   };

void check_init_assert();



std::string print_nodelist(const std::set<int_t>& vec, const onnx::GraphProto& graph);

std::vector<int_t> topological_sort(const std::map<int_t, std::set<int_t>>& EdgesForward,
                                    const std::map<int_t, std::set<int_t>>& EdgesBackward);

}
}
}

#endif //TMVA_SOFIE
