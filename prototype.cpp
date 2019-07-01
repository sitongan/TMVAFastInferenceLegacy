#include <iostream>
#include <fstream>
#include <cstdio>



#include "SOFIE.hxx"
#include "RGraph.hxx"
#include "RDataNode.hxx"

using std::vector;
using std::string;
using std::size_t;


using std::cout;
using std::endl;

using namespace TMVA::Experimental::SOFIE;
using namespace std;

void print_vector(vector<float>& t){
   for (auto const& i: t){
      cout << i << " ";
   }
   cout << "\n";
}

int test(){
   GOOGLE_PROTOBUF_VERIFY_VERSION;
   //model I/O
   onnx::ModelProto model;
   std::fstream input("LinearNN.onnx", std::ios::in | std::ios::binary);
//   std::fstream input("resnet18v1.onnx", std::ios::in | std::ios::binary);
   if (!model.ParseFromIstream(&input)){
      std::cerr << "Failed to parse onnx file." << endl;
      return -1;
   }


   RGraph graph_test(model.graph());


   RDataNode<float>* testtmp = static_cast<RDataNode<float>*>(graph_test.GetRDataNode("2"));
   cout << "shape: " << to_string(testtmp->GetShape()[0]) << "," << to_string(testtmp->GetShape()[1]) << endl;
   cout << "length: " << to_string(testtmp->GetLength()) << endl;
   //std::vector<float> t_a(const_cast<float*>(testtmp->GetData()), const_cast<float*>(testtmp->GetData()) + testtmp->GetLength());
   //cout << "content :";
   //print_vector(t_a);

   /*
   std::vector<float> input_a = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
   RDataNode<float>* dnode = new RDataNode<float>(input_a, {4,1,5,1}, "1");
   std::vector<float> t_a(const_cast<float*>(dnode->GetData()), const_cast<float*>(dnode->GetData()) + dnode->GetLength());
   print_vector(t_a);
   cout << endl;
   UTILITY::unidirectional_broadcast(*dnode, {2,4,2,5,2});
   std::vector<float> t_b(const_cast<float*>(dnode->GetData()), const_cast<float*>(dnode->GetData()) + dnode->GetLength());
   print_vector(t_b);
   cout << endl;
   */

   google::protobuf::ShutdownProtobufLibrary();

   exit(0);

   //model level metadata
   cout << "fIRVersion: " << model.ir_version() << endl;
   cout << "fModelVersion:" << model.model_version() << endl;
   for (int i =0; i < model.opset_import_size(); i++){
      cout << "Opset version: " << model.opset_import(0).version() << endl;
   }

   //extra check that onnx is using int_64
   cout << "size of int in sys:" << 8 * sizeof(int) << endl;
   cout << "size of int used by onnx:" << 8 * sizeof(model.ir_version()) << endl;


   //build computational graph representations
   const onnx::GraphProto& graph = model.graph();
   cout << "graph name: " << graph.name() << endl;
   std::map<string, int_t> datanode_match;
   //size_t will be the index of the other node (send/receive) of the datanode edge
   std::map<int_t, std::unordered_set<int_t>> EdgesForward;
   std::map<int_t, std::unordered_set<int_t>> EdgesBackward;

   std::unordered_set<string> initializer_names;
   for (int i=0; i < graph.initializer_size(); i++){
      initializer_names.insert(graph.initializer(i).name());
   }

   cout << "graph input names: [";
   for (int i=0; i < graph.input_size(); i++){
      if (initializer_names.find(graph.input(i).name()) == initializer_names.end()){
         //input datanode is not a weight node (has no initializer)
         datanode_match[graph.input(i).name()] = -1;
         cout << graph.input(i).name() << ",";
      }
   }
   cout << "]" << endl;

   cout << "graph output names: [";
   for (int i=0; i < graph.output_size(); i++){
      datanode_match[graph.output(i).name()] = -1;
      cout << graph.output(i).name() << ",";
   }
   cout << "]" << endl;


   cout << "check - datanode_match after initialization: [";
   for (auto const& item: datanode_match){
      cout << " " << item.first << ":" << item.second << ",";
   }
   cout << "]" << endl;

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

   cout << "EdgesForward: " << endl;
   for (auto const& item : EdgesForward){
      cout << item.first << ":" << INTERNAL::print_nodelist(item.second, graph) << endl;
   }
   cout << endl;

   cout << "EdgesBackward: " << endl;
   for (auto const& item : EdgesBackward){
      cout << item.first << ":" << INTERNAL::print_nodelist(item.second, graph) << endl;
   }
   cout << endl;

   // NOT ACTUALLY NEEDED to do topological sort, the nodes in their order are already topologically sorted
   vector<int64_t> eval_order = INTERNAL::topological_sort(EdgesForward, EdgesBackward);
   if (eval_order.size() != graph.node_size() +1){
      cout << "Error: Computational graph not a DAG!" << endl;
      return 1;
   }
/*

   //test RDataNode constructor for TensorProto
   RDataNode testnode (graph.initializer(0));
   auto ptr_data = static_cast<float64_t*>(testnode.GetData());

   cout.precision(17);
   cout << ptr_data[4999] << endl;
   //cout << ptr_data->at(4999) << endl;
   //cout << ptr_data->data()[4999] << endl;
   //cout << ptr_data->size() << endl;

   RDataNode testnode_2 (ptr_data, ETensorType::FLOAT, {50, 100});
   cout << "testnode 2" << endl;
   auto ptr_data_2 = static_cast<float64_t*>(testnode_2.GetData());
   cout << ptr_data_2[4999] << endl;
*/

   RDataNode<float> testnode (graph.initializer(0)); //this line will be jitted
   auto ptr_data = testnode.GetData();
   cout.precision(17);
   cout << ptr_data[4999] << endl;


   cout << "INPUT DIMENSION: ";
   for (int i = 0; i < graph.input_size(); i++){
      if (initializer_names.find(graph.input(i).name()) != initializer_names.end()) continue;
      cout << graph.input(i).name() << ":"  << graph.input(i).type().tensor_type().shape().dim_size();
      cout << " [";
      for (int j = 0; j <graph.input(i).type().tensor_type().shape().dim_size(); j++){
         if (graph.input(i).type().tensor_type().shape().dim(j).has_dim_value()){
            cout << graph.input(i).type().tensor_type().shape().dim(j).dim_value() << ",";
         }else if (graph.input(i).type().tensor_type().shape().dim(j).has_dim_param()){
            cout << graph.input(i).type().tensor_type().shape().dim(j).dim_param() << ",";
         }
      }
      cout << "]" << endl;
   }

   cout << "OUTPUT DIMENSION: ";
   for (int i = 0; i < graph.output_size(); i++){
      cout << graph.output(i).name() << ":"  << graph.output(i).type().tensor_type().shape().dim_size();
      cout << " [";
      for (int j = 0; j <graph.output(i).type().tensor_type().shape().dim_size(); j++){
         if (graph.output(i).type().tensor_type().shape().dim(j).has_dim_value()){
            cout << graph.output(i).type().tensor_type().shape().dim(j).dim_value() << ",";
         }else if (graph.output(i).type().tensor_type().shape().dim(j).has_dim_param()){
            cout << graph.output(i).type().tensor_type().shape().dim(j).dim_param() << ",";
         }
      }
      cout << "]" << endl;
   }

   cout << "node idx: input to node : output from node" << endl;
   for (int i = 0; i < graph.node_size(); i++){
      cout << i << ":";
      for (int j = 0; j < graph.node(i).input_size(); j++){
          if (initializer_names.find(graph.node(i).input(j)) != initializer_names.end()) continue;
          cout << " " << graph.node(i).input(j) << ",";
      }
      cout << ":";
      for (int j = 0; j < graph.node(i).output_size(); j++){
          cout << " " << graph.node(i).output(j) << ",";
      }
      cout << endl;
   }







   google::protobuf::ShutdownProtobufLibrary();
   return 0;
}



int main(){
   return test();
}
