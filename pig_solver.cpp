//
// Created by jinyuanfeng on 2021/3/3.
//
#include "pig_solver.h"

//Implemention Of PADTensor
template <typename T>
auto PS::PADTensor<T>::data() {
    return get_ctx_data<T>(id);
}
template <typename T>
auto PS::PADTensor<T>::grad() {
    return get_ctx_grad<T>(id);
}
template <typename T>
string PS::PADTensor<T>::get_name() {
    return id;
}
template <typename T>
void PS::PADTensor<T>::wrap_data(xt::xarray<T> xt_data){
    //auto s = chrono::system_clock::now();
    save_ctx_data<T>(id,xt_data);
    save_ctx_grad<T>(id,xt_data*0);
    //auto e = chrono::system_clock::now();
    //cout<<"run time warp data:"<<chrono::duration<double>{e-s}.count()*1000<<endl;
}
template <typename T>
void PS::PADTensor<T>::zero_grad() {
    save_ctx_grad<T>(id,xt::zeros<T>(grad().shape()));
}

template <typename T>
vector<string> OP<T>::get_parent(){
    return vector<string>();
};
template <typename T>
void OP<T>::forward(){}

template <typename T>
vector<xt::xarray<T>> OP<T>::backward(xt::xarray<T> grad){
    return vector<xt::xarray<T>>();
}

template <typename T>
vector<string> Add<T>::get_parent(){
    return parent_name;
}
template <typename T>
PS::PADTensor<T> Add<T>::forward(PS::PADTensor<T> a, PS::PADTensor<T> b){
    // save parent tensor
    parent_name = {a.get_name(),b.get_name()};
    // forward
    PS::PADTensor<T> out(this);
    out.wrap_data(a.data()+b.data());
    return out;
}
template <typename T>
vector<xt::xarray<T>> Add<T>::backward(xt::xarray<T> grad){
    vector<xt::xarray<T>> out;
    out.push_back(grad);
    out.push_back(grad);
    return out;
}

template <typename T>
vector<string> MulMat<T>::get_parent(){
    return parent_name;
}
template <typename T>
PS::PADTensor<T> MulMat<T>::forward(PS::PADTensor<T> a, PS::PADTensor<T> b){
    // save parent tensor
    parent_name = {a.get_name(),b.get_name()};
    // forward
    PS::PADTensor<T> out(this);
    out.wrap_data(xt::linalg::dot(a.data(),b.data()));
    return out;
}
template <typename T>
vector<xt::xarray<T>> MulMat<T>::backward(xt::xarray<T> grad){
    vector<xt::xarray<T>> out;
    //AxB=C
    //delta A = grad x B^T
    //delta B = A^T x grad
    xt::xarray<T> B = PS::get_ctx_data<T>(parent_name[1]);
    xt::xarray<T> A = PS::get_ctx_data<T>(parent_name[0]);
    out.push_back(xt::linalg::dot(grad,xt::transpose(B, {1,0})));
    out.push_back(xt::linalg::dot(xt::transpose(A, {1,0}),grad));
    return out;
}


int main() {
          PS::PADTensor<float> u = PS::ones<float,2>({3000,1200});//PS::PADTensor<float>({{1,2},{1,2},{1,2}}, nullptr);
          PS::PADTensor<float> v = PS::ones<float,2>({1200,3000});//PS::PADTensor<float>({{2,2},{2,2}}, nullptr);
          cout<<u.data()<<endl;
          cout<<v.data()<<endl;
          //MulMat<float>* op = new MulMat<float>();
          auto s = chrono::system_clock::now();
          auto res = u.mulmat_xt(v);
          auto e = chrono::system_clock::now();
          cout<<"run time:"<<chrono::duration<double>{e-s}.count()*1000<<endl;
          PS::PADTensor<float> a = u.mulmat(v);
          PS::bp<float>(a.get_name());
          cout<<"bp grad:"<<endl;
          cout<<u.grad()<<endl;
          cout<<v.grad()<<endl;
          PS::clear_env<float>();
    return 0;
}

