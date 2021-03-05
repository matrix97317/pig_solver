//
// Created by jinyuanfeng on 2021/3/3.
//

#ifndef PIG_SOLVER_PIG_SOLVER_H
#define PIG_SOLVER_PIG_SOLVER_H

#include <NumCpp.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>

using namespace std;

template<typename T>
class OP;//前置声明

template<typename T>
class Tensor{
public:
    OP<T>*  parent_op;
    nc::NdArray<T> data;
    nc::NdArray<T> grad;
public:
    Tensor(nc::NdArray<T>a, OP<T>*op);
    Tensor(const Tensor<T> &obj);
    ~Tensor();
    void bp(nc::NdArray<T> grad);
    void zero_grad();
};

template <typename T>
class OP{
public:
    OP()=default;
    virtual ~OP()=default;
    vector<Tensor<T>*> parent_tensor;
    virtual Tensor<T>* forward();
    virtual void show_name();
    virtual vector<Tensor<T>*> get_parent_tensor();
    virtual vector<nc::NdArray<T> > backward(nc::NdArray<T> grad);
};

template <typename T>
class Add:public OP<T>{
private:
    string name;
public:
    Add();
    ~Add()=default;
    void show_name();
    vector<Tensor<T>*> parent_tensor;
    Tensor<T>* forward(Tensor<T> *a, Tensor<T> *b);
    vector<Tensor<T>*> get_parent_tensor();
    vector<nc::NdArray<T> > backward(nc::NdArray<T> grad);
};

template <typename T>
class MulMat:public OP<T>{
private:
    string name;
public:
    MulMat();
    ~MulMat()=default;
    void show_name();
    vector<Tensor<T>*> parent_tensor;
    Tensor<T>* forward(Tensor<T> *a, Tensor<T> *b);
    vector<Tensor<T>*> get_parent_tensor();
    vector<nc::NdArray<T> > backward(nc::NdArray<T> grad);
};

template <typename T>
class AddConst:public OP<T>{
private:
    string name;
public:
    AddConst();
    ~AddConst()=default;
    void show_name();
    vector<Tensor<T>*> parent_tensor;
    Tensor<T>* forward(Tensor<T> *a, T b);
    vector<Tensor<T>*> get_parent_tensor();
    vector<nc::NdArray<T> > backward(nc::NdArray<T> grad);
};

template <typename T>
class Sub:public OP<T>{
private:
    string name;
public:
    Sub();
    ~Sub()=default;
    void show_name();
    vector<Tensor<T>*> parent_tensor;
    Tensor<T>* forward(Tensor<T> *a, Tensor<T> *b);
    vector<Tensor<T>*> get_parent_tensor();
    vector<nc::NdArray<T> > backward(nc::NdArray<T> grad);
};

template <typename T>
class SubConst:public OP<T>{
private:
    string name;
public:
    SubConst();
    ~SubConst()=default;
    void show_name();
    vector<Tensor<T>*> parent_tensor;
    Tensor<T>* forward( T a, Tensor<T> * b);
    vector<Tensor<T>*> get_parent_tensor();
    vector<nc::NdArray<T> > backward(nc::NdArray<T> grad);
};

template <typename T>
class MulConst:public OP<T>{
private:
    string name;
public:
    MulConst();
    ~MulConst()=default;
    void show_name();
    vector<Tensor<T>*> parent_tensor;
    vector<T> const_v;
    Tensor<T>* forward(Tensor<T> *a, T b);
    vector<Tensor<T>*> get_parent_tensor();
    vector<nc::NdArray<T> > backward(nc::NdArray<T> grad);
};

template <typename T>
class Div:public OP<T>{
private:
    string name;
public:
    Div();
    ~Div()=default;
    void show_name();
    vector<Tensor<T>*> parent_tensor;
    Tensor<T>* forward(Tensor<T> *a, Tensor<T> *b);
    vector<Tensor<T>*> get_parent_tensor();
    vector<nc::NdArray<T> > backward(nc::NdArray<T> grad);
};

template <typename T>
class DivConst:public OP<T>{
private:
    string name;
public:
    DivConst();
    ~DivConst()=default;
    void show_name();
    vector<Tensor<T>*> parent_tensor;
    vector<T> const_v;
    Tensor<T>* forward(T a,Tensor<T> *b);
    vector<Tensor<T>*> get_parent_tensor();
    vector<nc::NdArray<T> > backward(nc::NdArray<T> grad);
};

template <typename T>
class Sum:public OP<T>{
private:
    string name;
public:
    Sum();
    ~Sum()=default;
    void show_name();
    vector<Tensor<T>*> parent_tensor;
    Tensor<T>* forward(Tensor<T> *a);
    vector<Tensor<T>*> get_parent_tensor();
    vector<nc::NdArray<T> > backward(nc::NdArray<T> grad);
};

template <typename T>
class Exp:public OP<T>{
private:
    string name;
public:
    Exp();
    ~Exp()=default;
    void show_name();
    vector<Tensor<T>*> parent_tensor;
    Tensor<T>* forward(Tensor<T> *a);
    vector<Tensor<T>*> get_parent_tensor();
    vector<nc::NdArray<T> > backward(nc::NdArray<T> grad);
};

template <typename T>
class Log:public OP<T>{
private:
    string name;
public:
    Log();
    ~Log()=default;
    void show_name();
    vector<Tensor<T>*> parent_tensor;
    Tensor<T>* forward(Tensor<T> *a);
    vector<Tensor<T>*> get_parent_tensor();
    vector<nc::NdArray<T> > backward(nc::NdArray<T> grad);
};

#endif //PIG_SOLVER_PIG_SOLVER_H
