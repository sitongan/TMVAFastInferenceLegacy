#include "SOFIE.hxx"

using namespace TMVA::Experimental::SOFIE;

TMVA::Experimental::SOFIE::RDataNode::RDataNode(const onnx::TensorProto& tensorproto): fTensorProto(tensorproto)
{
   fType = static_cast<ETensorType>(tensorproto.data_type());
   if (tensorproto.has_name()){
      fName = tensorproto.name();
   }else{
      fName = "";
   }
   for (int i = 0; i < tensorproto.dims_size(); i++){
      fShape.push_back(tensorproto.dims(i));
   }
   fIsSegment = tensorproto.has_segment();
   if (fIsSegment){
      fSegmentIndex = std::make_tuple(tensorproto.segment().begin(),tensorproto.segment().end());
   }


}






void TMVA::Experimental::SOFIE::check_init_assert()
{
   static_assert(8 * sizeof(float) == 32, "TMVA-SOFIE is not supported on machines with non-32 bit float");
}



std::string TMVA::Experimental::SOFIE::print_nodelist(const std::set<int_t>& vec, const onnx::GraphProto& graph)
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

std::vector<int_t> TMVA::Experimental::SOFIE::topological_sort(const std::map<int_t, std::set<int_t>>& EdgesForward,
                                    const std::map<int_t, std::set<int_t>>& EdgesBackward){
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
