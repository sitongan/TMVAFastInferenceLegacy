#include <iostream>
#include <vector>
#include "TMVA/RTensor.hxx"
#include <string>

using namespace std;
using TMVA::Experimental::RTensor;


class op;
class op_add;
class op_mul;

class database{
public:
   virtual void EvalUpon(const op*) const =0;///
   //virtual void EvalUpon(const op_add*) const =0;///
   //virtual void EvalUpon(const op_mul*) const =0;///
};

class dataint : public database{
public:
    void EvalUpon(const op* b) const override;///
    //void EvalUpon(const op_add* b) const override;///
    //void EvalUpon(const op_mul* b) const override;///
};

class datafloat : public database{
public:
    void EvalUpon(const op* b) const override;///
    ///void EvalUpon(const op_add* b) const override;///
    ///void EvalUpon(const op_mul* b) const override;///
};

template<typename T>
class datatemplate : public database{
    void EvalUpon(const op* b) const override;///
    ///void EvalUpon(const op_add* b) const override;///
    ///void EvalUpon(const op_mul* b) const override;///
};

class op{
public:
   virtual void eval(const database*) const = 0;
   virtual void eval(const dataint*) const {
      cout << "this op does not support int" << endl;
   }
   virtual void eval(const datafloat*) const {
      cout << "this op does not support float" << endl;
   }
};

class op_add : public op{
public:
    void eval(const database* a) const override{
      a->EvalUpon(this);
   }
    void eval(const dataint* a) const override{
      cout << "Eval add on int" << endl;
   }
    void eval(const datafloat* a) const override{
      cout << "Eval add on float" << endl;
   }
   template<typename T>
    void eval(const datatemplate<T>* a) const {
      cout << "Eval add on template" << endl;
   }
};

class op_mul : public op{
public:
    void eval(const database* a) const override{
      a->EvalUpon(this);
   }
    void eval(const dataint* a) const override{
      cout << "Eval mul on int" << endl;
   }
   /* void eval(const datafloat* a) const override{
      cout << "Eval mul on float" << endl;
   }*/
   template<typename T>
   void eval(const datatemplate<T>* a) const {
      cout << "Eval mul on template" << endl;
   }
};

/*
void dataint::EvalUpon(const op_add* b) const {
   b->eval(this);
}
void dataint::EvalUpon(const op_mul* b) const {
   b->eval(this);
}*/

void dataint::EvalUpon(const op* b) const {
   b->eval(this);
   //cout << "I was here" << endl;
}
/*
void datafloat::EvalUpon(const op_add* b) const {
   b->eval(this);
}
void datafloat::EvalUpon(const op_mul* b) const {
   b->eval(this);
}*/
void datafloat::EvalUpon(const op* b) const {
   b->eval(this);
}
/*
template<typename T>
void datatemplate<T>::EvalUpon(const op_add* b) const {
   b->eval(this);
}
template<typename T>
void datatemplate<T>::EvalUpon(const op_mul* b) const {
   b->eval(this);
}*/
template<typename T>
void datatemplate<T>::EvalUpon(const op* b) const {
   b->eval(this);
}

int main(){
   //float data[] = {1, 2, 3, 4, 5, 6};
   vector<float> data {1, 2, 3, 4, 5, 6};
   RTensor<float> x(data.data(), {2, 3});
   std::cout << x << std::endl;

   dataint d_int;
   datafloat d_float;
   database* ptr_d_int = &d_int;
   database* ptr_d_float = &d_float;
   op_add o_add;
   op_mul o_mul;
   op* ptr_o_add = &o_add;
   op* ptr_o_mul = &o_mul;

   ptr_o_add->eval(ptr_d_int);
   ptr_o_add->eval(ptr_d_float);
   ptr_o_mul->eval(ptr_d_int);
   ptr_o_mul->eval(ptr_d_float);

   datatemplate<int> d_template;
   database* ptr_d_template = &d_template;
   //ptr_o_add->eval(ptr_d_template);
   //ptr_o_mul->eval(ptr_d_template);


   return 0;
}
