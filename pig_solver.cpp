//
// Created by jinyuanfeng on 2021/3/3.
//
#include "pig_solver.h"
template <typename T>
Tensor<T>:: Tensor(nc::NdArray<T> a, OP<T> *op){
    data = a;
    parent_op = op;
    grad = nc::NdArray<T>(data.shape())=0;
}

template <typename T>
Tensor<T>:: Tensor(const Tensor<T> &obj){
    data = obj.data;
    parent_op = obj.parent_op;
}
template <typename T>
Tensor<T>::~Tensor() {
    if (parent_op != nullptr){
        delete parent_op;
    }
}
template <typename T>
void Tensor<T>::zero_grad(){
    grad = nc::NdArray<T>(data.shape())=0;
}
template <typename T>
void Tensor<T>::bp(nc::NdArray<T> from_grad){

    if(from_grad.isempty()){
       from_grad =  nc::NdArray<T>(data.shape())=1;
    }
    if(parent_op!= nullptr){
        vector<nc::NdArray<T>> next_grad = parent_op->backward(from_grad);
        vector<Tensor<T>*> pt = parent_op->get_parent_tensor();
        for(int i=0; i<next_grad.size();i++){
            pt[i]->grad = pt[i]->grad + next_grad[i];
            pt[i]->bp(next_grad[i]);
        }
    }else{
        return;
    }
}
//Implemention of OP
template <typename T>
Tensor<T>* OP<T>::forward(){
    return nullptr;
}

template <typename T>
void OP<T>::show_name() {
    cout<<"OP"<<endl;
}

template <typename T>
vector<Tensor<T>*> OP<T>::get_parent_tensor(){
    return parent_tensor;
}

template <typename T>
vector<nc::NdArray<T> > OP<T>::backward(nc::NdArray<T> grad){
    vector<nc::NdArray<T> > ret;
    ret.push_back(grad);
    return ret;
}

//Implemention of Add
template <typename T>
Add<T>::Add(){
    name="Add";
}

template <typename T>
Tensor<T>* Add<T>::forward(Tensor<T> *a, Tensor<T> *b) {
    parent_tensor.push_back(a);
    parent_tensor.push_back(b);
    auto ret = new  Tensor<T>(a->data+b->data,this);
    return ret;
}

template <typename T>
void Add<T>::show_name() {
    cout<<name<<endl;
}

template <typename T>
vector<Tensor<T>*> Add<T>::get_parent_tensor() {
   return parent_tensor;
}

template <typename T>
vector<nc::NdArray<T>> Add<T>::backward(nc::NdArray<T> grad){
    vector<nc::NdArray<T>> ret;
    ret.push_back(grad);
    ret.push_back(grad);
    return ret;
}

//Implemention of MulMat
template <typename T>
MulMat<T>::MulMat(){
    name="MulMat";
}

template <typename T>
Tensor<T>* MulMat<T>::forward(Tensor<T> *a, Tensor<T> *b) {
    parent_tensor.push_back(a);
    parent_tensor.push_back(b);
    auto ret = new  Tensor<T>( nc::dot<T>(a->data,b->data),this);
    return ret;
}

template <typename T>
void MulMat<T>::show_name() {
    cout<<name<<endl;
}

template <typename T>
vector<Tensor<T>*> MulMat<T>::get_parent_tensor() {
    return parent_tensor;
}

template <typename T>
vector<nc::NdArray<T>> MulMat<T>::backward(nc::NdArray<T> grad){
    vector<nc::NdArray<T>> ret;
    nc::NdArray<T> A = parent_tensor[0]->data;
    nc::NdArray<T> B = parent_tensor[1]->data;
    //AxB=C
    //delta A = grad x B^T
    //delta B = A^T x grad
    ret.push_back(nc::dot(grad,B.transpose()));
    ret.push_back(nc::dot(A.transpose(),grad));
    return ret;
}

template <typename T>
AddConst<T>::AddConst(){
    name="AddConst";
}

template <typename T>
Tensor<T>* AddConst<T>::forward(Tensor<T> *a, T b) {
    parent_tensor.push_back(a);
    auto ret = new  Tensor<T>( a->data+b,this);
    return ret;
}

template <typename T>
void AddConst<T>::show_name() {
    cout<<name<<endl;
}

template <typename T>
vector<Tensor<T>*> AddConst<T>::get_parent_tensor() {
    return parent_tensor;
}

template <typename T>
vector<nc::NdArray<T>> AddConst<T>::backward(nc::NdArray<T> grad){
    vector<nc::NdArray<T>> ret;
    ret.push_back(grad);
    return ret;
}

template <typename T>
Sub<T>::Sub(){
    name="Sub";
}

template <typename T>
Tensor<T>* Sub<T>::forward(Tensor<T> *a, Tensor<T> *b) {
    parent_tensor.push_back(a);
    parent_tensor.push_back(b);
    auto ret = new  Tensor<T>(a->data-b->data,this);
    return ret;
}

template <typename T>
void Sub<T>::show_name() {
    cout<<name<<endl;
}

template <typename T>
vector<Tensor<T>*> Sub<T>::get_parent_tensor() {
    return parent_tensor;
}

template <typename T>
vector<nc::NdArray<T>> Sub<T>::backward(nc::NdArray<T> grad){
    vector<nc::NdArray<T>> ret;
    ret.push_back(grad);
    ret.push_back(-1*grad);
    return ret;
}

template <typename T>
SubConst<T>::SubConst(){
    name="SubConst";
}

template <typename T>
Tensor<T>* SubConst<T>::forward(T a, Tensor<T>* b) {
    parent_tensor.push_back(b);
    auto ret = new  Tensor<T>( a-b->data,this);
    return ret;
}

template <typename T>
void SubConst<T>::show_name() {
    cout<<name<<endl;
}

template <typename T>
vector<Tensor<T>*> SubConst<T>::get_parent_tensor() {
    return parent_tensor;
}

template <typename T>
vector<nc::NdArray<T>> SubConst<T>::backward(nc::NdArray<T> grad){
    vector<nc::NdArray<T>> ret;
    ret.push_back(-1*grad);
    return ret;
}

template <typename T>
MulConst<T>::MulConst(){
    name="MulConst";
}

template <typename T>
Tensor<T>* MulConst<T>::forward(Tensor<T> *a, T b) {
    parent_tensor.push_back(a);
    const_v.push_back(b);
    auto ret = new  Tensor<T>(a->data*b,this);
    return ret;
}

template <typename T>
void MulConst<T>::show_name() {
    cout<<name<<endl;
}

template <typename T>
vector<Tensor<T>*> MulConst<T>::get_parent_tensor() {
    return parent_tensor;
}

template <typename T>
vector<nc::NdArray<T>> MulConst<T>::backward(nc::NdArray<T> grad){
    vector<nc::NdArray<T>> ret;
    ret.push_back(const_v[0]*grad);
    return ret;
}

template <typename T>
DivConst<T>::DivConst(){
    name="DivConst";
}

template <typename T>
Tensor<T>* DivConst<T>::forward(T a, Tensor<T> * b) {
    const_v.push_back(a);
    parent_tensor.push_back(b);
    auto ret = new  Tensor<T>(a / b->data,this);
    return ret;
}

template <typename T>
void DivConst<T>::show_name() {
    cout<<name<<endl;
}

template <typename T>
vector<Tensor<T>*> DivConst<T>::get_parent_tensor() {
    return parent_tensor;
}

template <typename T>
vector<nc::NdArray<T>> DivConst<T>::backward(nc::NdArray<T> grad){
    vector<nc::NdArray<T>> ret;
    T t0 = const_v[0];
    nc::NdArray<T> t1 = parent_tensor[0]->data;
    ret.push_back(grad/t1);
    ret.push_back((-1*grad*t0)/(t1*t1));
    return ret;
}

template <typename T>
Div<T>::Div(){
    name="Div";
}

template <typename T>
Tensor<T>* Div<T>::forward(Tensor<T> *a, Tensor<T> *b) {
    parent_tensor.push_back(a);
    parent_tensor.push_back(b);
    auto ret = new  Tensor<T>(a->data / b->data,this);
    return ret;
}

template <typename T>
void Div<T>::show_name() {
    cout<<name<<endl;
}

template <typename T>
vector<Tensor<T>*> Div<T>::get_parent_tensor() {
    return parent_tensor;
}

template <typename T>
vector<nc::NdArray<T>> Div<T>::backward(nc::NdArray<T> grad){
    vector<nc::NdArray<T>> ret;
    nc::NdArray<T> t0 = parent_tensor[0]->data;
    nc::NdArray<T> t1 = parent_tensor[1]->data;
    ret.push_back(grad/t1);
    ret.push_back((-1*grad*t0)/(t1*t1));
    return ret;
}

template <typename T>
Sum<T>::Sum(){
    name="Sum";
}

template <typename T>
Tensor<T>* Sum<T>::forward(Tensor<T> *a) {
    parent_tensor.push_back(a);
    auto ret = new  Tensor<T>(nc::sum(a->data),this);
    return ret;
}

template <typename T>
void Sum<T>::show_name() {
    cout<<name<<endl;
}

template <typename T>
vector<Tensor<T>*> Sum<T>::get_parent_tensor() {
    return parent_tensor;
}

template <typename T>
vector<nc::NdArray<T>> Sum<T>::backward(nc::NdArray<T> grad){
    vector<nc::NdArray<T>> ret;
    nc::NdArray<T> t0 = parent_tensor[0]->data;
    ret.push_back(*grad.data()*nc::ones<T>(t0.shape()));
    return ret;
}

template <typename T>
Exp<T>::Exp(){
    name="Exp";
}

template <typename T>
Tensor<T>* Exp<T>::forward(Tensor<T> *a) {
    parent_tensor.push_back(a);
    auto ret = new  Tensor<T>(nc::exp(a->data),this);
    return ret;
}

template <typename T>
void Exp<T>::show_name() {
    cout<<name<<endl;
}

template <typename T>
vector<Tensor<T>*> Exp<T>::get_parent_tensor() {
    return parent_tensor;
}

template <typename T>
vector<nc::NdArray<T>> Exp<T>::backward(nc::NdArray<T> grad){
    vector<nc::NdArray<T>> ret;
    nc::NdArray<T> t0 = parent_tensor[0]->data;
    ret.push_back(grad * nc::exp(t0));
    return ret;
}

template <typename T>
Log<T>::Log(){
    name="Log";
}

template <typename T>
Tensor<T>* Log<T>::forward(Tensor<T> *a) {
    parent_tensor.push_back(a);
    auto ret = new  Tensor<T>(nc::log(a->data),this);
    return ret;
}

template <typename T>
void Log<T>::show_name() {
    cout<<name<<endl;
}

template <typename T>
vector<Tensor<T>*> Log<T>::get_parent_tensor() {
    return parent_tensor;
}

template <typename T>
vector<nc::NdArray<T>> Log<T>::backward(nc::NdArray<T> grad){
    vector<nc::NdArray<T>> ret;
    nc::NdArray<T> t0 = parent_tensor[0]->data;
    ret.push_back(grad / t0);
    return ret;
}

int main()
{
  
    nc::NdArray<float> a = {{1,2},{2,3}};
    Sum<float>* op_sum = new Sum<float>();
    Tensor<float>*  ta = new  Tensor<float>(a, nullptr);
    Tensor<float>*  c = op_sum->forward(ta);
    cout<<c->data<<endl;
    nc::NdArray<float> z;
    c->bp(z);
    cout<<ta->grad<<endl;
    return 0;
}

