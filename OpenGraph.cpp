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
      undefined, float, int, string, tensor, graph, floats, ints, strings, tensors, graphs
   };

   enum class ETensorType{
      undefined, float, uint8, int8, uint16, int16, int32, int64, string, bool, float16, double,
      uint32, uint64, complex64, complex128, bfloat16
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



   class RAttributeBase{ // interface for attributes
   public:
      template<typename T>
      T& GetValue(....){
            return (RAttribute<T> *)(this)->data;
      }
   protected:
      string name;
      size_t size;
      EAttributeType fType;

//      Gemm(const RAttribute& att1){
//         att1.GetValue<int>
//      }

   };

   template<typename T>
   class RAttribute<T>: public RAttributeBase{
         T data;
   };
   //template specialisation

   class ROpNode{
   public:

   private:
      vector<RDataNode*> fWeights;
      vector<RDataNode*> fInputs;
      vector<RDataNode*> fOutputs;
      vector<RAttribute*> fAttributes;
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
