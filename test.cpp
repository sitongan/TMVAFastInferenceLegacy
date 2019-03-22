#include <iostream>
#include <vector>
#include "TMVA/RTensor.hxx"
#include <string>

using namespace std;
using TMVA::Experimental::RTensor;



class DataNode{
public:

struct DataType{
   void* ptr_data;
   operator int*() const{
      return static_cast<int*>(ptr_data);
   }
   operator float*() const{
      return static_cast<float*>(ptr_data);
   }
} fData;

DataType GetData(){
   return fData;
}
};





class op;
class op_add;
class op_mul;

class database{
public:
   virtual ~database() {}
   string name;
   virtual void EvalBy(const op*) const =0;///
   //virtual void EvalBy(const op_add*) const =0;///
   //virtual void EvalBy(const op_mul*) const =0;///
   database(string n): name(n) {}
};

class dataint : public database{
public:
    void EvalBy(const op* b) const override;///
    //void EvalBy(const op_add* b) const override;///
    //void EvalBy(const op_mul* b) const override;///
    dataint(string n): database(n) {}
};

class datafloat : public database{
public:
    void EvalBy(const op* b) const override;///
    ///void EvalBy(const op_add* b) const override;///
    ///void EvalBy(const op_mul* b) const override;///
    datafloat(string n): database(n) {}
};

/*
template<typename T>
class datatemplate : public database{
    void EvalBy(const op* b) const override;///
    ///void EvalBy(const op_add* b) const override;///
    ///void EvalBy(const op_mul* b) const override;///
};
*/

class op{
public:
   virtual ~op() {}
   virtual void getDataPtr(const database*) const = 0;
   virtual void getDataPtr(std::vector<database*> input_list) = 0;
   virtual void getDataPtr(const dataint*) const {
      cout << "this op does not support int" << endl;
   }
   virtual void getDataPtr(const datafloat*) const {
      cout << "this op does not support float" << endl;
   }
};

class op_add : public op{
public:
    void getDataPtr(const database* a) const override{
      a->EvalBy(this);
   }
   void getDataPtr( std::vector<database*> input_list)  override{
      for (auto& input : input_list){
         input->EvalBy(this);
      }

   }


    void getDataPtr(const dataint* a) const override{
      cout << "Eval add on int " << a->name << endl;
   }
    void getDataPtr(const datafloat* a) const override{
      cout << "Eval add on float " << a->name << endl;
   }
   /*template<typename T>
    void getDataPtr(const datatemplate<T>* a) const {
      cout << "Eval add on template" << endl;
   }*/
};

class op_mul : public op{
public:
    void getDataPtr(const database* a) const override{
      a->EvalBy(this);
   }
   void getDataPtr( std::vector<database*> input_list)  override{
      for (auto& input : input_list){
         input->EvalBy(this);
      }
   }

    void getDataPtr(const dataint* a) const override{
      cout << "Eval mul on int" << endl;
   }
   /* void getDataPtr(const datafloat* a) const override{
      cout << "Eval mul on float" << endl;
   }*/

   /*template<typename T>
   void getDataPtr(const datatemplate<T>* a) const {
      cout << "Eval mul on template" << endl;
   }*/
};

/*
void dataint::EvalBy(const op_add* b) const {
   b->getDataPtr(this);
}
void dataint::EvalBy(const op_mul* b) const {
   b->getDataPtr(this);
}*/

void dataint::EvalBy(const op* b) const {
   b->getDataPtr(this);
   //cout << "I was here" << endl;
}
/*
void datafloat::EvalBy(const op_add* b) const {
   b->getDataPtr(this);
}
void datafloat::EvalBy(const op_mul* b) const {
   b->getDataPtr(this);
}*/
void datafloat::EvalBy(const op* b) const {
   b->getDataPtr(this);
}
/*
template<typename T>
void datatemplate<T>::EvalBy(const op_add* b) const {
   b->getDataPtr(this);
}
template<typename T>
void datatemplate<T>::EvalBy(const op_mul* b) const {
   b->getDataPtr(this);
}
template<typename T>
void datatemplate<T>::EvalBy(const op* b) const {
   b->getDataPtr(this);
}*/
void* getdata(std::vector<float> input){
   return static_cast<void*>(input.data());
}

int main(){
   //float data[] = {1, 2, 3, 4, 5, 6};
   vector<float> data {1, 2, 3, 4, 5, 6};
   RTensor<float> x(data.data(), {2, 3});
   std::cout << x << std::endl;

   dataint d_int("int1");
   datafloat d_float("float1");
   database* ptr_d_int = &d_int;
   database* ptr_d_float = &d_float;
   op_add o_add;
   op_mul o_mul;
   op* ptr_o_add = &o_add;
   op* ptr_o_mul = &o_mul;

   ptr_o_add->getDataPtr(ptr_d_int);



   ptr_o_add->getDataPtr(ptr_d_float);
   ptr_o_mul->getDataPtr(ptr_d_int);
   ptr_o_mul->getDataPtr(ptr_d_float);

   //datatemplate<int> d_template;
   //database* ptr_d_template = &d_template;
   //ptr_o_add->getDataPtr(ptr_d_template);
   //ptr_o_mul->getDataPtr(ptr_d_template);

   cout << "Test input list" << endl;
   std::vector<database*> input_list;
   input_list.push_back(ptr_d_int);
   input_list.push_back(ptr_d_float);
   ptr_o_add->getDataPtr(input_list);


   void* ptr_void;
   ptr_void = new std::vector<int>;
   static_cast<vector<int>*>(ptr_void)->push_back(1);
   cout << static_cast<vector<int>*>(ptr_void)->front() << endl;



   std::vector<float> v {100, 2, 3};
   DataNode dn {v.data()};
   float* ptr_float = dn.GetData();
   cout << *ptr_float << endl;
   return 0;


}
