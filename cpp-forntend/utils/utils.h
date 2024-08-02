#pragma once
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <torch/torch.h>
#include <ATen/ATen.h>

#define BLOCK 3

bool compare(std::string a, std::string b){ return a < b;}

template <typename Map>
void IndexMap(Map& m){
  std::cout << "{" ;
  for (auto& item : m){
    std::cout << item.first << ":" << item.second << " ";
  }
  std::cout << "}\n";
}

template <typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n){
  auto first = v.begin() + m;
  auto last = v.begin() + n;
  std::vector<T> vec(first, last);
  return vec;
}


// class Range_iterator{
//   public:
//     Range_iterator(std::size_t c) : count(c) {}
//     ~Range_iterator() = default;

//     Range_iterator &operator++() {++count; return *this;}
//     bool operator!=(const Range_iterator &ri) {return count != ri.count;}
//     std::size_t &operator*() {return count;}
//   private:
//     std::size_t count;
// }

// class Range{
//   public:
//     Range(std::size_t b, std::size_t e) : Begin(b), End(e) {if(b>e) Begin = End;}
//     Range(std::size_t e) : Range(0, e) {}
//     ~Range = default;

//     Range_iterator begin() {return Range_iterator(Begin);}
//     Range_iterator end() {return Range_iterator(End);}
//   private:
//     std::size_t Begin;
//     std::sizt_t End;
// }

template <typename Map>
void buildDataset(const std::vector<std::string> words, const Map& stoi, torch::Tensor & X, torch::Tensor & Y, int col)
{
  int xraw = words.size();
  std::vector<int> tensorx;
  std::vector<int> tensory;
  auto opts = torch::TensorOptions().dtype(torch::kInt32);
  std::vector<int> context = {0};
  for (std::string w: words){
    context.resize(3, 0);
    for (char v : w+'.'){
      int ix = stoi.find(v)->second;
      tensorx.insert(tensorx.end(), context.begin(), context.end());
      tensory.push_back(ix);
      context.erase(context.begin());
      context.push_back(ix);
    }
  }

  X = torch::from_blob(tensorx.data(), {xraw*col}, opts).reshape({xraw, col}).clone();
  Y = torch::from_blob(tensory.data(), {xraw}, opts).clone();
  std::cout << "Torch Size is: " << X.sizes() << " Torch Size is: " << Y.sizes() << std::endl;
}

void cmp(std::string s, torch::Tensor dt, torch::Tensor t){
  auto ex = dt.eq(t.grad()).all().item();
  auto app = torch::allclose(dt, t.grad());
  float maxdiff = (dt - t.grad()).abs().max().item<float>();
  // bool app = appclose == 1 ? true : false;
  // std::cout << appclose << sizeof(appclose) << std::endl;
  std::cout << std::left << std::setw(15) << s << " | exact :" << std::left << std::setw(5) <<  ex << " | approximate: " << std::left << std::setw(5) << app << " | maxdiff :" << maxdiff << std::endl;
}
