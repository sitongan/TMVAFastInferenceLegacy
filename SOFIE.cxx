#include "SOFIE.hxx"

namespace SOFIE = TMVA::Experimental::SOFIE;
using namespace SOFIE;

//constructor for graph.initializer (input with values)
SOFIE::RDataNode::RDataNode(const onnx::TensorProto& tensorproto): fTensorProto(tensorproto)
{
   if (tensorproto.has_name()){
      fName = tensorproto.name();
   }else{
      fName = "";
   }
   switch(tensorproto.data_type()) {
      case onnx::TensorProto::FLOAT : {
         fType = ETensorType::FLOAT;
         if (tensorproto.has_raw_data()){
            const float64_t* raw_data = reinterpret_cast<const float64_t*>(tensorproto.raw_data().c_str());
            fData.ptr_vector = new std::vector<float64_t>(raw_data, raw_data+5000);
            fHasData = true;
         }else if (tensorproto.float_data_size() > 0){
            const google::protobuf::RepeatedField<float64_t>& float_data = tensorproto.float_data();
            fData.ptr_vector = new std::vector<float64_t>(float_data.begin(), float_data.end());
            fHasData = true;
         }else{
            throw std::runtime_error("Tensor " + fName + " is not valid.");
         }
         break;
         }
      default: throw std::runtime_error("Data type in tensor " + fName + " not supported!");
   }
   for (int i = 0; i < tensorproto.dims_size(); i++){
      fShape.push_back(tensorproto.dims(i));
   }
   fIsSegment = tensorproto.has_segment();
   if (fIsSegment){
      fSegmentIndex = std::make_tuple(tensorproto.segment().begin(),tensorproto.segment().end());
   }


}


SOFIE::RDataNode::~RDataNode(){
   if (fHasData){
      switch(fType){
         case ETensorType::FLOAT: {
            delete static_cast<std::vector<float64_t>*>(fData.ptr_vector);
         }
      }
   }
}



void SOFIE::check_init_assert()
{
   static_assert(8 * sizeof(float) == 32, "TMVA-SOFIE is not supported on machines with non-32 bit float");
}



std::string SOFIE::print_nodelist(const std::unordered_set<int_t>& vec, const onnx::GraphProto& graph)
{
   std::string str {""};
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

std::vector<int_t> SOFIE::topological_sort(const std::map<int_t, std::unordered_set<int_t>>& EdgesForward,
                                    const std::map<int_t, std::unordered_set<int_t>>& EdgesBackward){
   //returns a list of nodes in topological order, ended with -1
   std::vector<int_t> sorted;
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
