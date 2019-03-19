#include <iostream>
#include <fstream>
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


using std::vector;
using std::string;
using std::size_t;


using std::cout;
using std::endl;

namespace TMVA{
namespace Experimental{
namespace SOFIE{   //Project code. System for Open, Fast Inference and Evaluation

   typedef std::int64_t int_t;

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
      vector<int_t> fDim;
      ETensorType fType;
      bool fIsSegment;
      std::tuple<int_t, int_t> fSegmentIndex;
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


      //temp
      const onnx::NodeProto& fNodeProto; //for data access

      ROpNode(const onnx::NodeProto& nodeproto): fNodeProto(nodeproto) {}

   //private:
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





string print_nodelist(const std::set<int_t>& vec, const onnx::GraphProto& graph){
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

vector<int_t> topological_sort(const std::map<int_t, std::set<int_t>>& EdgesForward,
                                    const std::map<int_t, std::set<int_t>>& EdgesBackward){
   //returns a list of nodes in topological order, ended with -1
   vector<int_t> sorted;
   std::map<int_t, int_t>in_degree;
   for (auto const& OpNodeID : EdgesBackward){
      in_degree[OpNodeID.first] = OpNodeID.second.size();
   }
   std::queue<int_t, std::deque<int_t>> queue(std::deque<int_t>(EdgesForward.at(-1).begin(), EdgesForward.at(-1).end())); //put inputs into queue
   while(!queue.empty()){
      int_t this_node_id = queue.front();
      sorted.push_back(this_node_id);
      queue.pop();
      if (this_node_id == -1) continue;
      for (auto const& child_ID : EdgesForward.at(this_node_id)){
         in_degree[child_ID] -= 1;
         if (in_degree[child_ID] == 0){
            queue.push(child_ID);
         }
      }
   }
   return sorted;
   //you should check sorted.size() == no. of nodes in graph to make sure non-DAG
}



int test(){
   GOOGLE_PROTOBUF_VERIFY_VERSION;
   //model I/O
   onnx::ModelProto model;
   std::fstream input("resnet18v1.onnx", std::ios::in | std::ios::binary);
   if (!model.ParseFromIstream(&input)){
      std::cerr << "Failed to parse onnx file." << endl;
      return -1;
   }
   cout << "fIRVersion: " << model.ir_version() << endl;
   cout << "fModelVersion:" << model.model_version() << endl;
   for (int i =0; i < model.opset_import_size(); i++){
      cout << "Opset version: " << model.opset_import(0).version() << endl;
   }

   cout << "size of int in sys:" << 8 * sizeof(int) << endl;
   cout << "size of int used by onnx:" << 8 * sizeof(model.ir_version()) << endl;


   vector<TMVA::Experimental::SOFIE::RDataNode> fDataNodes;
   vector<TMVA::Experimental::SOFIE::ROpNode> fOpNodes;
   const onnx::GraphProto& graph = model.graph();
   cout << "graph name: " << graph.name() << endl;
   /*
   for (int i=0; i < graph.node_size(); i++){
     cout << node[i].op_type() << endl;
   }
   */

   std::map<string, int_t> datanode_match;
   //size_t will be the index of the other node (send/receive) of the datanode edge
   std::map<int_t, std::set<int_t>> EdgesForward;
   std::map<int_t, std::set<int_t>> EdgesBackward;

   std::set<string> initializer_names;
   for (int i=0; i < graph.initializer_size(); i++){
      initializer_names.insert(graph.initializer(i).name());
   }

   for (int i=0; i < graph.input_size(); i++){
      if (initializer_names.find(graph.input(i).name()) == initializer_names.end()){
         //input datanode is not a weight node (has no initializer)
         datanode_match[graph.input(i).name()] = -1;
      }
   }
   for (int i=0; i < graph.output_size(); i++){
      cout << graph.input(i).name() << endl;
      datanode_match[graph.output(i).name()] = -1;
   }

   cout << "datanode_match after initialization" << endl;
   for (auto const& item: datanode_match){
      cout << item.first << ":" << item.second << endl;
   }

   for (int i=0; i< graph.node_size(); i++){
      for (int j=0; j < graph.node(i).input_size(); j++){
         const string& input_name {graph.node(i).input(j)};
         if (initializer_names.find(input_name) == initializer_names.end()){
         //if this input to this node is not an initializer

            if(datanode_match.find(input_name) != datanode_match.end()){
               EdgesForward[datanode_match[input_name]].insert(i);
               EdgesBackward[i].insert(datanode_match[input_name]);
            }else{
               datanode_match[input_name] = i;
            }
         }
      }
      for (int j=0; j < graph.node(i).output_size(); j++){
         string output_name {graph.node(i).output(j)};
         if(datanode_match.find(output_name) != datanode_match.end()){
            EdgesBackward[datanode_match[output_name]].insert(i);
            EdgesForward[i].insert(datanode_match[output_name]);
         }else{
            datanode_match[output_name] = i;
         }
      }
   }

   cout << "EdgesForward" << endl;
   for (auto const& item : EdgesForward){
      cout << item.first << ":" << print_nodelist(item.second, graph) << endl;
   }
   cout << "EdgesBackward" << endl;
   for (auto const& item : EdgesBackward){
      cout << item.first << ":" << print_nodelist(item.second, graph) << endl;
   }

   vector<int64_t> eval_order =topological_sort(EdgesForward, EdgesBackward);
   if (eval_order.size() != graph.node_size() +1){
      cout << "Error: Computational graph not a DAG!" << endl;
      return 1;
   }
   cout << "Topological sort: ";
   for (auto const& item : eval_order){
      cout << item << ", ";
   }
   cout << endl;










   google::protobuf::ShutdownProtobufLibrary();
   return 0;
}

}
}
}

int main(){
   return TMVA::Experimental::SOFIE::test();
}
