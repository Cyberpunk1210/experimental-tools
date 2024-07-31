#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <map>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/extension.h>

// int batch_size = 32;
// typedef at::BFloat16 bf16;

// std::vector<at::Tensor> gpttoy_forward(
//     torch::Tensor input,
//     torch::Tensor weight1,
//     torch::Tensor bias1,
//     torch::Tensor weight2,
//     torch::Tensor bias2,
//     torch::Tensor bngain,
//     torch::Tensor bnbias,
//     torch::Tensor Xtr,
//     torch::Tensor Ytr) {
//   auto ix = torch::randint(0, Xtr.sizes()[0], (batch_size));
//   auto Xb = Xtr.slice({ix});
//   auto Yb = Ytr.slice({ix});
//   auto emb = input.slice({Xb});
//   auto embcat = emb.view({emb.sizes()[0], -1});
//   auto hprebn = embcat.mul(weight1) + bias1;
//   auto bnmean = hprebn.mean(0, true);
//   auto bnvar = hprebn.var(0, true).unbiased(true);
//   auto bnvar_inv = (bnvar + 0.00001).pow(-0.5);
//   auto bnraw = (hprebn - bnmean) * bnvar_inv;
//   auto hpreact = bngain * bnraw + bnbias;
//   auto h = torch::tanh(hpreact);
//   auto logits = h.mul(weight2) + bias2;
//   auto loss = torch::nn::functional::cross_entropy(logits, Yb);
//   return {
//     input,
//     weight1,
//     bias1,
//     weight2,
//     bias2,
//     bngain,
//     bnbias};
// }


// std::vector<torch::Tensor> gpttoy_backward(
//     torch::Tensor input,
//     torch::Tensor emb,
//     torch::Tensor embcat,
//     torch::Tensor hprebn,
//     torch::Tensor bnmean,
//     torch::Tensor bnvar,
//     torch::Tensor bnvar_inv,
//     torch::Tensor bnraw,
//     torch::Tensor hpreact,
//     torch::Tensor h,
//     torch::Tensor logits,
//     torch::Tensor Xb,
//     torch::Tensor Yb,
//     torch::Tensor weight1,
//     torch::Tensor bias1,
//     torch::Tensor weight2,
//     torch::Tensor bias2,
//     torch::Tensor bngain,
//     torch::Tensor bnbias) {
//   auto dlogits = torch::nn::functional::softmax(logits, 1);
//   dlogits = dlogits.slice({{std::vector<int> batch_size}, Yb}) - 1;
//   dlogits = dlogits / batch_size;
//   auto dh = dlogits.mul(weight1.transpose({0, 1}));
//   auto dweight2 = h.transpose({0, 1}).mul(dlogits);
//   auto dbias2 = dlogits.sum(0);

//   auto dhpreact = (1.0 - h.pow(2)) * dh;

//   auto dbngain = (bnraw * dhpreact).sum(0, true);
//   auto dbnbias = dhpreact.sum(0, true);
//   auto dhprebn = bngain*bnvar_inv/batch_size * (batch_size*dhpreact - dhpreact.sum(0) - batch_size/(batch_size-1)*bnraw*(dhpreact*bnraw).sum(0));

//   auto dembcat = dhprebn.mul(weight1.transpose({0, 1}));
//   auto dweight1 = embcat.transpose({0, 1}).mul(dhprebn);
//   auto dbias1 = dhprebn.sum(0);
//   auto demb = dembcat.view({emb.sizes()});
//   auto dinput = torch::zeros_like(input);
//   for (int k=0; k<Xb.size()[0]; k++){
//     for (int j=0; j<Xb.size()[1]; k++){
//         auto ix = Xb.select({k, j});
//         auto dC.index({ix}) = demb.slice({k, j});
//     }
//   }
//   return {dinput, dweight1, dbias1, dweight2, dbias2, dbngain, dbnbias};
// }


// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
//     m.def("forward", &gpttoy_forward, "gpttoy forward");
//     m.def("backward", &gpttoy_backward, "gpttoy backward");
// }


#define BLOCKSIZE 3
#define BATCHSIZE 64

bool compare(std::string a, std::string b){ return a < b;}

template <typename Map>
void IndexMap(Map& m){
  std::cout << "{" ;
  for (auto& item : m){
    std::cout << item.first << ":" << item.second << " ";
  }
  std::cout << "}\n";
}

at::Tensor build_dataset(std::vector<std::string> words, std::map stoi){
  std::vector<int> X;
  std::vector<int> Y;
  for (std::string w : words){
    std::vector context;
    context.resize(BATCHSIZE);
    w += ".";
    for (char ch : w){
      std::vector<int> ix = stoi.find(ch);
      X.push_back(context);
      Y.push_back(ix);
    }
  }
  at::TensorOptions opts = at::TensorOptions().dtype(at::kInt);
  c10::IntArrayRef xsize = {1, 3};
  c10::IntArrayRef ysize = {1};
  at::Tensor TensorX = at::from_blob(X.data(), xsize, opts).clone();
  at::Tensor TensorY = at::from_blob(Y.data(), ysize, opts).clone();
  return TensorX, TensorY;
}

int main(){
  std::string filename = "names.txt";
  std::fstream myfile;

  myfile.open("../" + filename);
  std::vector<std::string> words;

  if (myfile.is_open()){
    std::string str;

    while (std::getline(myfile, str)){
      words.push_back(str);
    }

    myfile.close();
  }

  std::vector<int> maxLen;
  std::vector<int>::iterator result;
  for (int i=0; i<words.size(); i++){
    int lens = words[i].size();
    maxLen.push_back(lens);
  }
  result = std::max_element(maxLen.begin(), maxLen.end()); // printf the max length

  std::cout << "words lens: " << words.size() << std::endl;
  std::cout << "The max length is: " << *result << std::endl;
  std::cout << "The first eight elements are: ";
  for (int i=0; i<8; i++){
    std::cout << words[i] << " " << std::endl;
  }

  std::vector<std::string> all_string;
  std::vector<char> sorted_string;
  std::string combing_string = std::accumulate(words.begin(), words.end(), std::string(""));
  std::sort(combing_string.begin(), combing_string.end(), [](char x, char y) {return x < y; });
  combing_string.erase(std::unique(combing_string.begin(), combing_string.end()), combing_string.end());
  combing_string += ".";
  
  for (int i=0; i<combing_string.size(); i++)
    sorted_string.push_back(combing_string[i]);

  std::map<int, char> itos;
  for (int i=0; i<sorted_string.size(); i++)
    itos[i+1] = sorted_string[i];
  std::cout << "itos: ";
  IndexMap(itos);
  std::cout << sorted_string.size() << std::endl;
  return 0;
}
