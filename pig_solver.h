//
// Created by jinyuanfeng on 2021/3/3.
//

#ifndef PIG_SOLVER_PIG_SOLVER_H
#define PIG_SOLVER_PIG_SOLVER_H
//#define EIGEN_USE_THREADS
//#define EIGEN_USE_BLAS
#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>
#include <typeinfo>
#include <chrono>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor-blas/xblas.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <unordered_map>

using namespace std;

template <class T, std::size_t I>
struct new_initializer_list
{
    using type = std::initializer_list<typename new_initializer_list<T, I - 1>::type>;
};

template <class T>
struct new_initializer_list<T, 0>
{
    using type = T;
};

template <class T, std::size_t I>
using new_initializer_list_t = typename new_initializer_list<T, I>::type;

template <size_t D>
using xtsize = std::array<size_t, D>;

template <typename T>
class OP;

template <typename T>
class Add;

template <typename T>
class MulMat;

namespace PS{
    int node_count;
    // data storage
    template <typename T>
    unordered_map<string,xt::xarray<T>> data_storage;
    template <typename T>
    using data_itr_type = typename unordered_map<string,xt::xarray<T>>::iterator;

    template <typename T>
    void save_ctx_data(string key, xt::xarray<T> value){
        data_itr_type<T> it = data_storage<T>.find(key);
        if(it == data_storage<T>.end()){
            data_storage<T>.insert(pair<string,xt::xarray<T>>(key,value));
        }
    }

    template <typename T>
    auto get_ctx_data(string key){
        data_itr_type<T> it = data_storage<T>.find(key);
        if(it != data_storage<T>.end()){
            return it->second;
        }
        return xt::xarray<T>();
    }

    // grad storage
    template <typename T>
    unordered_map<string,xt::xarray<T>> grad_storage;
    template <typename T>
    using grad_itr_type = typename unordered_map<string,xt::xarray<T>>::iterator;

    template <typename T>
    void save_ctx_grad(string key, xt::xarray<T> value){
        grad_itr_type<T> it = grad_storage<T>.find(key);
        if(it == grad_storage<T>.end()){
            grad_storage<T>.insert(pair<string,xt::xarray<T>>(key,value));
        }
    }

    template <typename T>
    auto get_ctx_grad(string key){
        grad_itr_type<T> it = grad_storage<T>.find(key);
        if(it != grad_storage<T>.end()){
            return it->second;
        }
        return xt::xarray<T>();
    }

    template <typename T>
    void update_ctx_grad(string key, xt::xarray<T> new_grad){
        grad_itr_type<T> it = grad_storage<T>.find(key);
        if(it != grad_storage<T>.end()){
            it->second += new_grad ;
        }
    }


    template <typename T>
    unordered_map<string,OP<T>*> op_hash;
    template <typename T>
    using op_itr_type = typename unordered_map<string,OP<T>*>::iterator;

    template <typename T>
    void save_ctx_op(string key, OP<T>* value){
        if(op_hash<T>.find(key)== op_hash<T>.end()){
            op_hash<T>.insert(pair<string,OP<T>*>(key,value));
        }
    }

    template <typename T>
    auto get_ctx_op(string key){
        op_itr_type<T> it = op_hash<T>.find(key);
        if(it != op_hash<T>.end()){
            return it->second;
        }
        OP<T>* ret = nullptr;
        return ret;
    }

    template <typename T>
    void bp(string id, xt::xarray<T> parent_grad=xt::zeros<T>({1}),int flag=0){
        if(!flag){
            parent_grad = get_ctx_grad<T>(id)+1;
            update_ctx_grad<T>(id,parent_grad);
        }
        auto parent_op = get_ctx_op<T>(id);
        if (parent_op!= nullptr){
            vector<xt::xarray<T>> next_grad = parent_op->backward(parent_grad);
            vector<string> pt_name = parent_op->get_parent();
            for(int i=0; i<next_grad.size();i++){
                update_ctx_grad<T>(pt_name[i],next_grad[i]);
                bp(pt_name[i],next_grad[i],1);
            }
        }else{
            return;
        }
    }
    template <typename T>
    void clear_env(){
        data_storage<T>.clear();
        grad_storage<T>.clear();
        op_itr_type<T> it = op_hash<T>.begin();
        while(it!= op_hash<T>.end()){
            delete it->second;
            it++;
        }
        op_hash<T>.clear();
    }
    template <typename T>
    class PADTensor{
    private:
        string id;
        xt::xarray<T> inner_data;
        OP<T> *parent_op;
    public:
        PADTensor( OP<T> * op){
            parent_op = op;
            PS::node_count++;
            id = "tensor"+to_string(PS::node_count);
            save_ctx_op<T>(id,op);
        }
        PADTensor(new_initializer_list_t<T ,1> t, OP<T> * op){
            parent_op = op;
            PS::node_count++;
            id = "tensor"+to_string(PS::node_count);
            save_ctx_data<T>(id,xt::xarray<T>(t));
            save_ctx_grad<T>(id,xt::xarray<T>(t)*0);
            save_ctx_op<T>(id,op);
        }
        PADTensor(new_initializer_list_t<T ,2> t, OP<T> * op){
            parent_op = op;
            PS::node_count++;
            id = "tensor"+to_string(PS::node_count);
            save_ctx_data<T>(id,xt::xarray<T>(t));
            save_ctx_grad<T>(id,xt::xarray<T>(t)*0);
            save_ctx_op<T>(id,op);
        }
        PADTensor(new_initializer_list_t<T ,3> t, OP<T> * op){
            parent_op = op;
            PS::node_count++;
            id = "tensor"+to_string(PS::node_count);
            save_ctx_data<T>(id,xt::xarray<T>(t));
            save_ctx_grad<T>(id,xt::xarray<T>(t)*0);
            save_ctx_op<T>(id,op);
        }
        PADTensor(new_initializer_list_t<T ,4> t, OP<T> * op){
            parent_op = op;
            PS::node_count++;
            id = "tensor"+to_string(PS::node_count);
            save_ctx_data<T>(id,xt::xarray<T>(t));
            save_ctx_grad<T>(id,xt::xarray<T>(t)*0);
            save_ctx_op<T>(id,op);
        }
        PADTensor(new_initializer_list_t<T ,5> t, OP<T> * op){
            parent_op = op;
            PS::node_count++;
            id = "tensor"+to_string(PS::node_count);
            save_ctx_data<T>(id,xt::xarray<T>(t));
            save_ctx_grad<T>(id,xt::xarray<T>(t)*0);
            save_ctx_op<T>(id,op);
        }
        ~PADTensor(){
            //cout<<"release: "<<id<<endl;
           /* if (parent_op!= nullptr){
                delete parent_op;
            }*/
        }
        auto operator+(PADTensor<T> b){
            Add<T>* op = new  Add<T>();
            return op->forward((*this),b);
        }
        auto mulmat(PADTensor<T> b){
            MulMat<T>* op = new  MulMat<T>();
            return op->forward((*this),b);
        }
        auto mulmat_xt(PADTensor<T> b){
            return xt::linalg::dot(this->data(),b.data());
        }
        auto data();
        auto grad();
        string get_name();
        void wrap_data(xt::xarray<T> xt_data);
        void zero_grad();
    };

    template <typename T,int dims>
    auto ones(std::array<std::size_t,dims> t){
        if(t.size()==0){
            throw "param can't as empty";
        }
        xtsize<dims> dim_list = t;
        PS::PADTensor<T> new_tensor = PS::PADTensor<T>(nullptr);
        new_tensor.wrap_data(xt::ones<T>(dim_list));
        return new_tensor;
    }
    template <typename T,int dims>
    auto zeros(std::array<std::size_t,dims> t){
        if(t.size()==0){
            throw "param can't as empty";
        }
        xtsize<dims> dim_list = t;
        PS::PADTensor<T> new_tensor = PS::PADTensor<T>(nullptr);
        new_tensor.wrap_data(xt::zeros<T>(dim_list));
        return new_tensor;
    }
    template <typename T,int dims>
    auto randn(std::array<std::size_t,dims> t){
        if(t.size()==0){
            throw "param can't as empty";
        }
        xtsize<dims> dim_list = t;
        PS::PADTensor<T> new_tensor = PS::PADTensor<T>(nullptr);
        new_tensor.wrap_data(xt::random::randn<T>(dim_list));
        return new_tensor;
    }

};

template <typename T>
class OP{
public:
    virtual vector<string> get_parent();
    virtual void forward();
    virtual vector<xt::xarray<T>> backward(xt::xarray<T> grad);
};

template <typename T>
class Add:public OP<T>{
private:
    vector<string> parent_name;
public:
    vector<string> get_parent();
    PS::PADTensor<T> forward(PS::PADTensor<T> a, PS::PADTensor<T> b);
    vector<xt::xarray<T>> backward(xt::xarray<T> grad);
};

template <typename T>
class MulMat:public OP<T>{
private:
    vector<string> parent_name;
    vector<xt::xarray<T>> parent_data;
public:
    vector<string> get_parent();
    PS::PADTensor<T> forward(PS::PADTensor<T> a, PS::PADTensor<T> b);
    vector<xt::xarray<T>> backward(xt::xarray<T> grad);
};
#endif //PIG_SOLVER_PIG_SOLVER_H
