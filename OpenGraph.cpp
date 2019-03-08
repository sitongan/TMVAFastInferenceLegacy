#include <iostream>
#include <fstream>
#include <cstddef>
#include <vector>
#include <string>
#include <map>
#include <tuple>
#include <array>
#include <set>
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

//    protected:
      public:
      string name;
      vector<int64_t> fDim;
      ETensorType fType;
      bool fIsSegment;
      std::tuple<int64_t, int64_t> fSegmentIndex;
      bool fHasData;

      //temp
      const onnx::TensorProto& fTensorProto; //for data access

//    public:
      template<typename T>
      T* GetData(){} //return std::vector.data()
      RDataNode(const onnx::TensorProto& tensorproto)
      : fTensorProto(tensorproto), fType(static_cast<ETensorType>(tensorproto.data_type()))
      {
         if (tensorproto.has_name()){
            name = tensorproto.name();
         }else{
            name = "";
         }
         for (int i = 0; i < tensorproto.dims_size(); i++){
            fDim.push_back(tensorproto.dims(i));
         }
         fIsSegment = tensorproto.has_segment();
         if (fIsSegment){
            fSegmentIndex = std::make_tuple(tensorproto.segment().begin(),tensorproto.segment().end());
         }
      }

   };

   template<typename T>
   class RDataNodeData: public RDataNode{
      std::vector<T> fData;
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
      string fName;
      size_t fSize;
      EAttributeType fType;

   };

   template<typename T>
   class RAttributeData: public RAttribute{
         T fValue;
   };
   //template specialisation for list of attributes?

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

   //private:
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


string print(const vector<std::int64_t>& vec, const onnx::GraphProto& graph){
   string str {""};
   for (auto const& item : vec){
      str += std::to_string(item);
      if (item == -1){
         str += ("[I/O]");
      }else{
         str += ("[" + graph.node(item).op_type() + "]");
      }
      str += ",";
   }
   return str;
}


int main(){
   GOOGLE_PROTOBUF_VERIFY_VERSION;
   onnx::ModelProto model;
   std::fstream input("resnet18v1.onnx", std::ios::in | std::ios::binary);
   if (!model.ParseFromIstream(&input)){
      std::cerr << "Failed to parse onnx file." << endl;
      return -1;
   }
   //model I/O
   cout << "fIRVersion: " << model.ir_version() << endl;
   cout << "fModelVersion:" << model.model_version() << endl;
   cout << "opsetid_size: " << model.opset_import_size() << endl;
   const onnx::OperatorSetIdProto& opset = model.opset_import(0);
   cout << "Opset version: " << opset.version() << endl;

   cout << "size of int in sys:" << 8 * sizeof(int) << endl;
   cout << "size of int used by onnx:" << 8 * sizeof(model.ir_version()) << endl;

   const onnx::GraphProto& graph = model.graph();
   cout << graph.name() << endl;
   /*
   vector<const onnx::NodeProto> node;
   for (int i=0; i < graph.node_size(); i++){
     node.push_back(graph.node(i));
   }
   for (int i=0; i < graph.node_size(); i++){
     cout << node[i].op_type() << endl;
   }
   for (int i=0; i < graph.node_size(); i++){
     cout << node[i].op_type() << endl;
   }
   */
   std::map<string, std::int64_t> datanode_edge;
   //size_t will be the index of the other node (send/receive) of the datanode edge
   std::map<std::int64_t, vector<std::int64_t>> EdgesForward;
   std::map<std::int64_t, vector<std::int64_t>> EdgesBackward;

   std::set<string> initializer_names;

   for (int i=0; i < graph.initializer_size(); i++){
      initializer_names.insert(graph.initializer(i).name());
   }

   for (int i=0; i < graph.input_size(); i++){
      if (initializer_names.find(graph.input(i).name()) == initializer_names.end()){
         //input datanode is not a weight node (has no initializer)
         datanode_edge[graph.input(i).name()] = -1;
      }

   }
   for (int i=0; i < graph.output_size(); i++){
      cout << graph.input(i).name() << endl;
      datanode_edge[graph.output(i).name()] = -1;
   }

   cout << "datanode_edge after initialization" << endl;
   for (auto const& item: datanode_edge){
      cout << item.first << ":" << item.second << endl;
   }

   for (int i=0; i< graph.node_size(); i++){
      for (int j=0; j < graph.node(i).input_size(); j++){
         const string& datanode_name {graph.node(i).input(j)};
         if (initializer_names.find(datanode_name) == initializer_names.end()){
         //if input to this node is not an initializer
            if(datanode_edge.find(datanode_name) != datanode_edge.end()){
               EdgesForward[datanode_edge[datanode_name]].push_back(i);
               EdgesBackward[i].push_back(datanode_edge[datanode_name]);
            }else{
               datanode_edge[datanode_name] = i;
            }
         }
      }

      for (int j=0; j < graph.node(i).output_size(); j++){
         string datanode_name {graph.node(i).output(j)};
         if(datanode_edge.find(datanode_name) != datanode_edge.end()){
            EdgesBackward[datanode_edge[datanode_name]].push_back(i);
            EdgesForward[i].push_back(datanode_edge[datanode_name]);
         }else{
            datanode_edge[datanode_name] = i;
         }
      }
   }

   cout << "EdgesForward" << endl;
   for (auto const& item : EdgesForward){
      cout << item.first << ":" << print(item.second, graph) << endl;
   }
   cout << "EdgesBackward" << endl;
   for (auto const& item : EdgesBackward){
      cout << item.first << ":" << print(item.second, graph) << endl;
   }






   google::protobuf::ShutdownProtobufLibrary();
   return 0;
}
