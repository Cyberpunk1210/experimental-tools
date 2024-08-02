#include "utils.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <map>
#include <random>
#include <cmath>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/extension.h>

#define BATCH 32


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

  for (int i=0; i<combing_string.size(); i++)
    sorted_string.push_back(combing_string[i]);

  std::map<int, char> itos;
  std::map<char, int> stoi;
  for (int i=0; i<sorted_string.size(); i++)
  {
    itos[i+1] = sorted_string[i];
    stoi[sorted_string[i]] = i+1;
  }
  stoi.insert(std::map<char, int>::value_type('.', 0));
  itos.insert(std::map<int, char>::value_type(0, '.'));

  std::cout << "itos: ";
  IndexMap(itos);
  std::cout << "stoi: " ;
  IndexMap(stoi);
  std::cout << sorted_string.size() + 1 << std::endl;

  torch::manual_seed(2147483647);
  std::random_device rd;
  std::mt19937 rng(rd());
  std::shuffle(words.begin(), words.end(), rng);
  int first_rate = words.size() * 0.8;
  int second_rate = words.size() * 0.9;
  // std::cout << first_rate << " " << second_rate << std::endl; 25626 28829
  // std::cout << words.size() << std::endl; 32033
  // std::cout << three vector size; 25626 3203 3204
  torch::Tensor Xtr, Ytr, Xdev, Ydev, Xte, Yte;
  std::vector<std::string> trwords, devwords, tewords;
  trwords = slice(words, 0, first_rate);
  devwords = slice(words, first_rate, second_rate);
  tewords = slice(words, second_rate, words.size());

  buildDataset(trwords, stoi, Xtr, Ytr, BLOCK);
  buildDataset(devwords, stoi, Xdev, Ydev, BLOCK);
  buildDataset(tewords, stoi, Xte, Yte, BLOCK);

  std::cout << "vocab_size: " << itos.size() << std::endl;
  std::cout << Xtr.sizes() << std::endl;
  int vocab_size = itos.size();

  int n_embd=10, n_hidden=64, emblock=BLOCK*10;
  std::vector<at::Tensor> parameters;
  auto C = torch::randn({vocab_size, n_embd});
  parameters.push_back(C);

  // layer 1
  auto weight1 = torch::randn({emblock, n_hidden});
  weight1 = weight1 * (5/3)/(std::pow(emblock, 0.5));
  parameters.push_back(weight1);
  auto bias1 = torch::randn({n_hidden});
  bias1 = bias1 * 0.1;
  parameters.push_back(bias1);

  // layer 2
  auto weight2 = torch::randn({n_hidden, vocab_size});
  weight2 = weight2 * 0.1;
  parameters.push_back(weight2);
  auto bias2 = torch::randn({vocab_size});
  bias2 = bias2 * 0.1;
  parameters.push_back(bias2);

  auto bngain = torch::randn({1, n_hidden});
  bngain = bngain * 0.1 + 1.0;
  parameters.push_back(bngain);
  auto bnbias = torch::randn({1, n_hidden});
  bnbias = bnbias * 0.1;
  parameters.push_back(bnbias);

  for (auto &x : parameters)
    x.requires_grad();

  auto ix = torch::randint(0, Xtr.sizes()[0], {BATCH});

  auto Xb = Xtr.index({ix});
  auto Yb = Ytr.index({ix});

  auto emb = C.index({Xb});
  auto embcat = emb.view({emb.sizes()[0], -1});

  /* Linear layer 1 */
  auto hprebn = embcat.mm(weight1) + bias1;
  // std::cout << embcat.sizes() << " " << weight1.sizes() << " " << bias1.sizes() << std::endl;
  std::cout << hprebn.sizes() << std::endl;

  /* BatchNorm layer */
  int n = 32;
  auto bnmeani = hprebn.sum(0, true);
  bnmeani *= 1.0 / n;
  auto bndiff = hprebn - bnmeani;
  auto bndiff2 = torch::pow(bndiff, 2);
  auto bnvar = bndiff2.sum(0, true);
  bnvar *= 1.0 / (n - 1);
  auto bnvar_inv = (bnvar+1e-5f).pow(-0.5);
  auto bnraw = bndiff * bnvar_inv;
  auto hpreact = bngain * bnraw + bnbias;

  /* Non-linearity */
  auto h = torch::tanh(hpreact);

  /* Linear layer 2 */
  auto logits = h.mm(weight2) + bias2;

  /* cross entropy loss (same as F.cross_entropy(logits, Yb) */
  std::tuple<torch::Tensor, torch::Tensor> logit_maxes_tuple = torch::max(logits, 1, true);

  auto logit_maxes = std::get<0>(logit_maxes_tuple);
  auto norm_logits = logits - logit_maxes;
  auto counts = norm_logits.exp();
  auto counts_sum = counts.sum(1, true);
  auto counts_sum_inv = counts_sum.pow(-1);
  auto probs = counts * counts_sum_inv;
  auto logprobs = probs.log();
  int nprobs[n];
  #pragma unroll
  for (int i=0; i<n; i++)
    nprobs[i] = i+1;
  auto rangen = torch::from_blob(nprobs, {n}, torch::kInt32);
  // auto loss = -logprobs.index({rangen});

  std::cout << rangen.sizes();
  // for (auto &p : parameters)
  //   p.grad = NULL;



  return 0;
}
