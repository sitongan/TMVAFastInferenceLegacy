#include <iostream>
#include <fstream>




#include "SOFIE.hxx"

using std::vector;
using std::string;
using std::size_t;


using std::cout;
using std::endl;

using namespace TMVA::Experimental::SOFIE;


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

// NOT ACTUALLY NEEDED to do topological sort, the nodes in their order are already topologically sorted
   vector<int64_t> eval_order = topological_sort(EdgesForward, EdgesBackward);
   if (eval_order.size() != graph.node_size() +1){
      cout << "Error: Computational graph not a DAG!" << endl;
      return 1;
   }

//   cout << "Topological sort: ";
//   for (auto const& item : eval_order){
//      cout << item << ", ";
//   }
//   cout << endl;


   cout << graph.initializer(0).name() << endl;
   cout << graph.initializer(0).float_data_size() << endl;






   google::protobuf::ShutdownProtobufLibrary();
   return 0;
}



int main(){
   return test();
}
