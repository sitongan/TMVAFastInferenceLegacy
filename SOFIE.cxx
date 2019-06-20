#include "SOFIE.hxx"

namespace SOFIE = TMVA::Experimental::SOFIE;
using namespace SOFIE;


 template class SOFIE::RDataNode<float64_t>;   //explicit template initialization


/*
//constructor for graph.initializer (input with immutable values)
SOFIE::RDataNode::RDataNode(const onnx::TensorProto& tensorproto)
{
   if (tensorproto.has_name()){
      fName = tensorproto.name();
   }else{
      fName = "";
   }
   fLength = 1;
   for (int i = 0; i < tensorproto.dims_size(); i++){
      fShape.push_back(tensorproto.dims(i));
      fLength *= tensorproto.dims(i);
   }
   switch(tensorproto.data_type()) {
      case onnx::TensorProto::FLOAT : {
         fType = ETensorType::FLOAT;
         if (tensorproto.has_raw_data()){
            //const float64_t* raw_data = reinterpret_cast<const float64_t*>(tensorproto.raw_data().c_str());
            //fDataVector = new std::vector<float64_t>(raw_data, raw_data+fLength);
            //fHasData = true;
            fImmutableData = reinterpret_cast<const float64_t*>(tensorproto.raw_data().c_str());
            fHasImmutableData = true;
         }else if (tensorproto.float_data_size() > 0){
            //const google::protobuf::RepeatedField<float64_t>& float_data = tensorproto.float_data();
            //fDataVector = new std::vector<float64_t>(tensorproto.float_data().begin(), tensorproto.float_data().end());
            //fHasData = true;
            fImmutableData = tensorproto.float_data().data();
            fHasImmutableData = true;
         }else{
            throw std::runtime_error("Tensor " + fName + " is not valid.");
         }
         break;
         }
      default: throw std::runtime_error("Data type in tensor " + fName + " not supported!");
   }

   fIsSegment = tensorproto.has_segment();
   if (fIsSegment){
      fSegmentIndex = std::make_tuple(tensorproto.segment().begin(),tensorproto.segment().end());
   }
}

SOFIE::RDataNode::RDataNode(const onnx::ValueInfoProto& valueinfoproto){
   if (valueinfoproto.has_name()){
      fName = valueinfoproto.name();
   }else{
      fName = "";
   }
   //fLength = 1;
   for (int i = 0; i < valueinfoproto.type().tensor_type().shape().dim_size(); i++){
      //fShape.push_back(valueinfoproto.type().tensor_type().shape().dim(i));
      //fLength *= tensorproto.dims(i);
   }



   //int_t a = valueinfoproto.type().elem_type();
}

//copy constructor
SOFIE::RDataNode::RDataNode(const void* data, const ETensorType& type, const std::vector<int_t>& shape, const std::string& name)
   : fType(type), fName(name) {

   fShape = shape;
   fLength = 1;
   for (int i = 0; i < fShape.size(); i++){
      fLength *= fShape[i];
   }

   switch(fType) {
      case ETensorType::FLOAT : {
         const float64_t* raw_data = static_cast<const float64_t*>(data);
         fDataVector = new std::vector<float64_t>(raw_data, raw_data+fLength);
         break;
         }
      default: throw std::runtime_error("Data type in tensor " + fName + " not supported!");
   }

   fHasData = true;

   }


const void* SOFIE::RDataNode::RDataNode::GetData(){
   if (fHasData){
      switch(fType){
         case ETensorType::FLOAT: {
            return static_cast<std::vector<float64_t>*>(fDataVector)->data();
            break;
         }
      }
   }
   if (fHasImmutableData){
      switch(fType){
         case ETensorType::FLOAT: {
            return static_cast<const float64_t*>(fImmutableData);
            break;
         }
      }
   }
   throw std::runtime_error("No data in RDataNode " + fName);

}
*/


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
