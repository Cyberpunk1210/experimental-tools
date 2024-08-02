#pragma once
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <memory>

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
  auto first = v.cbegin() + m;
  auto last = v.cbegin() + n;
  std::vector<T> vec(first, last);
  return vec;
}

template <typename Map>
void buildDataset(const std::vector<std::string> words, const Map& stoi, torch::Tensor & X, torch::Tensor & Y)
{
  std::vector<std::vector<int32_t>> tensorx;
  std::vector<int32_t> tensory;
  // auto options = torch::TensorOptions().dtype(torch::kLong);
  for (std::string w: words){
    std::vector<int32_t> context = {0};
    context.resize(BLOCK, 0);
    for (char& v : w+"."){
      int32_t ix = stoi.find(v)->second;
      tensorx.push_back(context);
      tensory.push_back(ix);
      context.erase(context.begin());
      context.push_back(ix);
    }
  }

  X = torch::from_blob(tensorx.data(), {static_cast<int32_t>(tensorx.size()), static_cast<int32_t>(BLOCK)}, torch::TensorOptions().dtype(torch::kInt32)).clone();
  Y = torch::from_blob(tensory.data(), {static_cast<int32_t>(tensory.size())}, torch::TensorOptions().dtype(torch::kInt32)).clone();
  std::cout << "Torch Size is: " << X.sizes() << " Torch Size is: " << Y.sizes() << std::endl;
}
