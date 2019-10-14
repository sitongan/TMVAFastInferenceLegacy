#include <iostream>

#include <cstdio>



#include "RModel.hxx"
#include "SOFIE_utilities.hxx"

#include "TMVA/RTensor.hxx"
#include "TMVA/DNN/Architectures/Cpu/CpuBuffer.h"

using std::vector;
using std::string;
using std::size_t;


using std::cout;
using std::endl;

using namespace TMVA::Experimental::SOFIE;
using namespace std;

template<typename T>
void print_vector(vector<T>& t){
   for (auto const& i: t){
      cout << i << " ";
   }
   cout << "\n";
}
/*
#include "ROperator_Transpose.hxx"
#include <vector>
int test_2(){
   vector<int> a;
   for (int i = 0; i < 1000; i++){
      a.push_back(i);
   }
   print_vector(a);
   vector<int> b(1000, 0);
   OPERATION::Transpose_reference(a.data(), {4, 2, 5, 5, 5}, b.data(), {5, 4, 5, 2, 5}, {2, 0, 4, 1, 3});
   print_vector(b);
   return 0;
}
*/
/*
#include "ROperator_Gemm.hxx"
int test_3(){
   vector<float> a (50, 1.0);
   vector<float> b (50, 2.0);
   vector<float> c (25, 0.0);
   vector<float> y (25, 0.0);
   OPERATION::Gemm_reference(a.data(),b.data(), c.data(), y.data(),5, 10, 5);
   print_vector(y);
}
*/

int test(){

   using namespace TMVA::Experimental;

   TMVA::Experimental::RTensor<float, TMVA::DNN::TCpuBuffer<float>> test_RT({2,3,5});

   RModel model("LinearNN.onnx");
   RGraph* graph_test = model.GetMutableGraph();


   RDataNode<RTensor<float,TMVA::DNN::TCpuBuffer<float>>>* testtmp;

   RDataNode<RTensor<float,TMVA::DNN::TCpuBuffer<float>>>* testinput = static_cast<RDataNode<RTensor<float,TMVA::DNN::TCpuBuffer<float>>>*>(graph_test->GetRDataNode("input.1"));
   vector<float> test_input(6400, 1.0);
   testinput->Update(std::move(test_input), {64, 100});
   cout << "shape: " << to_string(testinput->GetShape()[0]) << "," << to_string(testinput->GetShape()[1]) << endl;
   cout << "length: " << to_string(testinput->GetLength()) << endl;
   graph_test->Forward();

   testtmp = static_cast<RDataNode<RTensor<float,TMVA::DNN::TCpuBuffer<float>>>*>(graph_test->GetRDataNode("0.weight"));
   cout << "shape: " << to_string(testtmp->GetShape()[0]) << "," << to_string(testtmp->GetShape()[1]) << endl;
   cout << "length: " << to_string(testtmp->GetLength()) << endl;

   testtmp = static_cast<RDataNode<RTensor<float,TMVA::DNN::TCpuBuffer<float>>>*>(graph_test->GetRDataNode("0.bias"));
   cout << "shape: " << to_string(testtmp->GetShape()[0]) << "," << to_string(testtmp->GetShape()[1]) << endl;
   cout << "length: " << to_string(testtmp->GetLength()) << endl;



   testtmp = static_cast<RDataNode<RTensor<float,TMVA::DNN::TCpuBuffer<float>>>*>(graph_test->GetRDataNode("7"));
   cout << "shape: " << to_string(testtmp->GetShape()[0]) << "," << to_string(testtmp->GetShape()[1]) << endl;
   cout << "length: " << to_string(testtmp->GetLength()) << endl;

   vector<float> output (testtmp->GetData(), testtmp->GetData() + 10);

   print_vector(output);
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


   exit(0);

/*
   //model level metadata



   //extra check that onnx is using int_64
   cout << "size of int in sys:" << 8 * sizeof(int) << endl;
   cout << "size of int used by onnx:" << 8 * sizeof(model.ir_version()) << endl;



   // NOT ACTUALLY NEEDED to do topological sort, the nodes in their order are already topologically sorted
   vector<int64_t> eval_order = INTERNAL::topological_sort(EdgesForward, EdgesBackward);
   if (eval_order.size() != graph.node_size() +1){
      cout << "Error: Computational graph not a DAG!" << endl;
      return 1;
   }
   */
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
/*
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

*/





   google::protobuf::ShutdownProtobufLibrary();
   return 0;
}



int main(){
   return test();
}
