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
#include <chrono>

#define BATCH 32

namespace F = torch::nn::functional;

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
  torch::manual_seed(42);
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
  int vocab_size = itos.size();

  int n_embd=10, n_hidden=64, emblock=BLOCK*10;
  std::vector<at::Tensor> parameters;
  auto C = torch::randn({vocab_size, n_embd}, torch::requires_grad());
  parameters.push_back(C);

  // layer 1
  auto weight1 = torch::randn({emblock, n_hidden}, torch::requires_grad());
  weight1 = weight1 * (5/3)/(std::pow(emblock, 0.5));
  parameters.push_back(weight1);
  auto bias1 = torch::randn({n_hidden}, torch::requires_grad());
  bias1 = bias1 * 0.1;
  parameters.push_back(bias1);

  // layer 2
  auto weight2 = torch::randn({n_hidden, vocab_size}, torch::requires_grad());
  weight2 = weight2 * 0.1;
  parameters.push_back(weight2);
  auto bias2 = torch::randn({vocab_size}, torch::requires_grad());
  bias2 = bias2 * 0.1;
  parameters.push_back(bias2);

  auto bngain = torch::randn({1, n_hidden}, torch::requires_grad());
  bngain = bngain * 0.1 + 1.0;
  parameters.push_back(bngain);
  auto bnbias = torch::randn({1, n_hidden}, torch::requires_grad());
  bnbias = bnbias * 0.1;
  parameters.push_back(bnbias);

  auto ix = torch::randint(0, Xtr.sizes()[0], {BATCH});

  auto Xb = Xtr.index({ix});
  auto Yb = Ytr.index({ix});

  /* forward pass */
  auto emb = C.index({Xb});
  auto embcat = emb.reshape({emb.sizes()[0], -1});

  /* Linear layer 1 */
  auto hprebn = embcat.mm(weight1) + bias1;
  // std::cout << embcat.sizes() << " " << weight1.sizes() << " " << bias1.sizes() << std::endl;

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
    nprobs[i] = i;
  auto rangen = torch::from_blob(nprobs, {n}, torch::kInt32);
  auto loss = -logprobs.index({rangen, Yb}).mean();

  torch::Tensor t[19] = {logprobs, probs, counts, counts_sum, counts_sum_inv,
                         norm_logits, logit_maxes, logits, h, hpreact, bnraw,
                         bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
                         embcat, emb};

  for (auto &p : parameters)
    p.retain_grad();
  // p.requires_grad_(true);

  for (auto &v : t)
    v.retain_grad();

  loss.backward();
  std::cout << loss << std::endl;

  /* Backward pass*/
  auto dlogprobs = torch::zeros({logprobs.sizes()});
  dlogprobs.index_put_({rangen, Yb}, -1.0/n);
  auto dprobs = (1.0 / probs) * dlogprobs;
  auto dcounts_sum_inv = (counts * dprobs).sum(1, true);
  auto dcounts = counts_sum_inv * dprobs;
  auto dcounts_sum = -counts_sum.pow(-2) * dcounts_sum_inv;
  dcounts += torch::ones({counts.sizes()}) * dcounts_sum;
  auto dnorm_logits = counts * dcounts;
  auto dlogits = dnorm_logits.clone();
  auto dlogit_maxes = (-dnorm_logits).sum(1, true);
  std::tuple<torch::Tensor, torch::Tensor> maxtuple = torch::max(logits, 1);
  torch::Tensor max_indices = std::get<1>(maxtuple);
  dlogits += F::one_hot(max_indices, logits.sizes()[1]) * dlogit_maxes;
  auto dh = dlogits.mm(weight2.t());
  auto dweight2 = h.t().mm(dlogits);
  auto dbias2 = dlogits.sum(0);
  auto dhpreact = (1.0 - h.pow(2)) * dh;
  auto dbngain = (dhpreact * bnraw).sum(0, true);
  auto dbnraw = (bngain * dhpreact);
  auto dbnbias = dhpreact.sum(0);
  auto dbndiff = bnvar_inv * dbnraw;
  auto dbnvar_inv = (bndiff * dbnraw).sum(0, true);
  auto dbnvar = -0.5 * (bnvar + 1e-5f).pow(-1.5) * dbnvar_inv;
  auto dbndiff2 = (1.0 / (n-1))* torch::ones({bndiff2.sizes()}) * dbnvar;
  dbndiff += 2 * bndiff * dbndiff2;
  auto dbnmeani = (-dbndiff).sum(0);
  auto dhprebn = dbndiff.clone();
  dhprebn += (1.0/n) * torch::ones({hprebn.sizes()}) * dbnmeani;
  auto dembcat = dhprebn.mm(weight1.t());
  auto dweight1 = embcat.t().mm(dhprebn);
  auto dbias1 = dhprebn.sum(0);
  auto demb = dembcat.reshape({emb.sizes()});
  auto dC = torch::zeros({C.sizes()});
  for (int k=0; k<Xb.sizes()[0]; k++){
    for (int j=0; j<Xb.sizes()[1]; j++){
      auto ix = Xb.index({k, j});
      dC[ix] += demb.index({k, j});
    }
  }

  cmp("logprobs", dlogprobs, logprobs);
  cmp("probs", dprobs, probs);
  cmp("counts_sum_inv", dcounts_sum_inv, counts_sum_inv);
  cmp("counts_sum", dcounts_sum, counts_sum);
  cmp("counts", dcounts, counts);
  cmp("norm_logits", dnorm_logits, norm_logits);
  cmp("logit_maxes", dlogit_maxes, logit_maxes);
  cmp("logits", dlogits, logits);
  cmp("h", dh, h);
  cmp("weight2", dweight2, weight2);
  cmp("bias2", dbias2, bias2);
  cmp("hpreact", dhpreact, hpreact);
  cmp("bngain", dbngain, bngain);
  cmp("bnraw", dbnraw, bnraw);
  cmp("bnbias", dbnbias, bnbias);
  cmp("bnvar_inv", dbnvar_inv, bnvar_inv);
  cmp("bnvar", dbnvar, bnvar);
  cmp("bndiff2", dbndiff2, bndiff2);
  cmp("bndiff", dbndiff, bndiff);
  cmp("bnmeani", dbnmeani, bnmeani);
  cmp("hprebn", dhprebn, hprebn);
  cmp("embcat", dembcat, embcat);
  cmp("weight1", dweight1, weight1);
  cmp("bias1", dbias1, bias1);
  cmp("emb", demb, emb);
  cmp("C", dC, C);

  /* unnecessary */
  // auto loss_fast = torch::nn::functional::cross_entropy(logits, Yb);
  // std::cout << loss_fast.item() << "diff: " << (loss_fast - loss).item() << std::endl;

  // dlogits = torch::nn::functional::softmax(logits, 1);
  // dlogits = (dlogits.index({rangen, Yb}) - 1) / n;
  // cmp("logits", dlogits, logits);
  // std::cout << "Original func: " << torch::nn::functional::softmax(logits, 1)[0] << std::endl;
  // std::cout << "Manually func: " << dlogits[0] * n << std::endl;
  // std::cout << dlogits[0].sum() << std::endl;

  /* TODO forward pass & backward pass & inference*/
  n_embd = 10;
  n_hidden = 200;
  C = torch::randn({vocab_size, n_embd});
  /* Layer 1 */
  auto W1 = torch::randn({n_embd * BLOCK, n_hidden});
  W1 = W1 * (5/3)/(std::pow(n_embd * BLOCK, 0.5));
  auto b1 = torch::randn({n_hidden});
  b1 *= 0.1;
  /* Layer 2 */
  auto W2 = torch::randn({n_hidden, vocab_size});
  W2 *= 0.1;
  auto b2 = torch::randn({vocab_size});
  b2 *= 0.1;
  /* BatchNorm parameters */
  bngain = torch::randn({1, n_hidden});
  bngain = bngain * 0.1 + 1.0;
  bnbias = torch::randn({1, n_hidden});
  bnbias *= 0.1;

  std::vector<torch::Tensor> paras = {C, W1, b1, W2, b2, bngain, bnbias};

  for (auto &p : paras)
    p.requires_grad_(true);

  int max_steps = 200000;
  int batch_size;
  batch_size = n = 12;
  std::vector<float> lossi;

  /* no_grad */
  {
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  torch::NoGradGuard nogard;
  torch::manual_seed(2147483647);

  for (int i=0; i<max_steps; i++){
    auto ix = torch::randint(0, Xtr.sizes()[0], {batch_size});
    auto Xb = Xtr.index({ix});
    auto Yb = Ytr.index({ix});

    auto emb = C.index({Xb});
    auto embcat = emb.view({emb.sizes()[0], -1});
    auto hprebn = embcat.mm(W1) + b1;

    auto bnmean = hprebn.mean(0, true);
    auto bnvar = hprebn.var(0, true, true);
    auto bnvar_inv = (bnvar + 1e-5f).pow(-0.5);
    auto bnraw = (hprebn - bnmean) * bnvar_inv;
    auto hpreact = bngain * bnraw + bnbias;
    /* Non-linearity */
    auto h = torch::tanh(hpreact);
    auto logits = h.mm(W2) + b2;
    auto loss = F::cross_entropy(logits, Yb.toType(torch::kLong), F::CrossEntropyFuncOptions().ignore_index(-100).reduction(torch::kMean));

    for (auto &p : parameters)
      p.grad().zero_();


    /* manual backprop */
    auto dlogits = F::softmax(logits, 1);
    auto rangesecn = torch::from_blob(nprobs, {n}, torch::kInt32);
    dlogits.index({rangesecn, Yb}) -= 1;
    dlogits /= n;

    auto dh = dlogits.mm(W2.t());
    auto dW2 = h.t().mm(dlogits);
    auto db2 = dlogits.sum(0);

    auto dhpreact = (1.0 - h.pow(2)) * dh;

    auto dbngain = (bnraw * dhpreact).sum(0, true);
    auto dbnbias = dhpreact.sum(0, true);
    auto dhprebn = bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0));

    auto dembcat = dhprebn.mm(W1.t());
    auto dW1 = embcat.t().mm(dhprebn);
    auto db1 = dhprebn.sum(0);

    auto demb = dembcat.view({emb.sizes()});
    auto dC = torch::zeros({C.sizes()});
    // for (int k=0; k<Xb.sizes()[0]; k++){
    //   for (int j=0; j<Xb.sizes()[1]; j++){
    //     auto ix = Xb.index({k, j});
    //     dC.index({ix}) += demb.index({k, j});
    //   }
    // }
    dC.index({Xb}).add_(demb);
    std::vector<torch::Tensor> grads = {dC, dW1, db1, dW2, db2, dbngain, dbnbias};
    float lr = i < 100000 ? 0.1 : 0.01;
    for (int i=0; i<paras.size(); ++i)
      paras[i].data() += -lr * grads[i];

    if (i % 10000 == 0)
      std::cout << std::left << std::setw(6) << i << "/" << max_steps << " : " << loss.item() << std::endl;

    auto loss_val = torch::log10(loss);
    lossi.push_back(loss_val.item<float>());
  }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_times = end - start;
  std::cout << "Done, Elapsed times: " << elapsed_times.count() << 's' << "\n";
  }

  return 0;
}
