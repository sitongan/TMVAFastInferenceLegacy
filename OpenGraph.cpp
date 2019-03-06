#include <iostream>
#include <fstream>
#include <cstddef>
#include <vector>
#include <string>
#include <map>
#include <tuple>
#include <cstdint> //int64_t: used for ONNX IR
#include <cstddef> // std::size_t


#include "onnx.pb.h"


using std::vector;
using std::string;
using std::size_t;


using std::cout;
using std::endl;

namespace TMVA{
namespace Experimental{
namespace OpenGraph{

   enum class EAttributeType{
      UNDEFINED, FLOAT, INT, STRING, TENSOR, GRAPH, FLOATS, INTS, STRINGS, TENSORS, GRAPHS
   };

   enum class ETensorType{
      UNDEFINED, FLOAT, UNINT8, INT8, UINT16, INT16, INT32, INT64, STRING, BOOL, FLOAT16, DOUBLE,
      UINT32, UINT64, COMPLEX64, COMPLEX28, BFLOAT16
   };

   class RDataNode{
   public:

   private:
      string name;
      vector<int64_t> fDim;
      ETensorType fType;
      bool fHasData;
         //???todo
   };




   class RAttribute{ // interface for attributes
   public:
      template<typename T>
      T& GetValue(){
            //return (RAttribute<T> *)(this)->fValue;
      }
   protected:
      string fName;
      size_t fSize;
      EAttributeType fType;

//      Gemm(const RAttribute& att1){
//         att1.GetValue<int>
//      }

   };

   template<typename T>
   class RAttributeData: public RAttribute{
         T fValue;
   };
   //template specialisation

   class ROpNode{
   public:

   private:
      vector<RDataNode*> fWeights;
      vector<RDataNode*> fInputs;
      vector<RDataNode*> fOutputs;
      vector<RAttribute*> fAttributes;
      string fOperator;
      //todo: map from string to function pointers for the operator
   };


   class RGraph{
   public:

   private:
      string fName;
      std::map<size_t, vector<size_t>> fEdgesForward;
      std::map<size_t, vector<size_t>> fEdgesBackward;
      vector<RDataNode> fDataNodes;
      vector<ROpNode> fOpNodes;
   };


   class RModel{
   public:
   private:
      int64_t fIRVersion;
      vector<std::tuple<string, int64_t>> fOpsetVersion;
      int64_t fModelVersion;
      std::map<string, string> fMetadata;
         //expected keys: "producer_name", "producer_version", "domain", "doc_string", "model_author", "model_license"
      RGraph fGraph; //main graph to be executed for the model
   };


}
}
}


int main(){
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  onnx::ModelProto model;
  std::fstream input("LinearNN.onnx", std::ios::in | std::ios::binary);
  if (!model.ParseFromIstream(&input)){
    std::cerr << "Failed to parse onnx file." << endl;
    return -1;
  }
  cout << "IR version: " << model.ir_version() << endl;
  const onnx::OperatorSetIdProto& opset = model.opset_import(0);
  cout << "Opset version: " << opset.version() << endl;

  const onnx::GraphProto& graph = model.graph();
  cout << graph.name() << endl;









  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
