#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <set>
#include <sys/types.h>
#include <sys/wait.h>
#include <math.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>

#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>

namespace ip = boost::interprocess;
//Define an STL compatible allocator of ints that allocates from the managed_shared_memory.
//This allocator will allow placing containers in the segment
typedef ip::allocator<int, ip::managed_shared_memory::segment_manager>  ShmemAllocatorInt;

//Alias a vector that uses the previous STL-like allocator so that allocates
//its values from the segment
typedef ip::vector<int, ShmemAllocatorInt> MyVectorInt;

typedef ip::allocator<float, ip::managed_shared_memory::segment_manager>  ShmemAllocatorFloat;
typedef ip::vector<float, ShmemAllocatorFloat> MyVectorFloat;

using namespace std;
using namespace cnn;
const int TOKENSOS = 0;
const int TOKENEOS = 1;
const int TAGSOS = 0;
const int TAGEOS = 1;

namespace po = boost::program_options;

unsigned LAYERS = 1;
unsigned INPUT_DIM = 256; // originally 128
unsigned XCRIBE_DIM = 256; // originally 128
unsigned TAG_RNN_HIDDEN_DIM = 128; // originally 32
unsigned TAG_DIM = 128; // originally 32
unsigned TAG_HIDDEN_DIM = 256; // originally 128
bool use_pretrained_embeding = false;
bool ner_tagging = false;
string pretrained_embeding = "";
int bias_mode = 0;
bool greedy_decoding = false;

int SampleFromDist(vector<float> dist){
  unsigned w = 0;
  float p = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  for (; w < dist.size(); ++w) {
    p -= dist[w];
    if (p < 0.0) { break; }
  }
  // this should not really happen?
  if (w == dist.size()) w = dist.size() - 1;
  return w;
}

bool extract_tag(int tag, cnn::Dict& td, pair<string, string>& res){
  if (tag == td.Convert("O") || tag == TAGSOS || tag == TAGEOS) return true;
  vector<string> fields;
  boost::algorithm::split( fields, td.Convert(tag), boost::algorithm::is_any_of( "-" ) );
  assert (fields.size() == 2);
  res.first = fields[0];
  res.second = fields[1];
  return false;
}

// returns embeddings of labels
struct SymbolEmbedding {
  unsigned dim;
  bool zero_out;
  SymbolEmbedding(Model& m, unsigned n, unsigned dim_, bool zero_out_ = false) {
    dim = dim_;
    zero_out = zero_out_;
    p_labels = m.add_lookup_parameters(n, {dim});
  }
  void load_embedding(cnn::Dict& d, string pretrain_path){
    ifstream fin(pretrain_path);  
       string s;
       while( getline(fin,s) )
       {   
        vector <string> fields;
        boost::algorithm::trim(s);
        boost::algorithm::split( fields, s, boost::algorithm::is_any_of( " " ) );
        string word = fields[0];
        vector<float> p_embeding;
        for (unsigned int ind = 1; ind < fields.size(); ++ind){
          p_embeding.push_back(std::stod(fields[ind]));
        }
        if (d.Contains(word)){
          // cout << "init" << endl;
          p_labels->Initialize(d.Convert(word), p_embeding);
        }
      }
  }
  void new_graph(ComputationGraph& g) { cg = &g; }
  Expression embed(unsigned label_id) {
    if (zero_out){
      vector<float> data(dim, 0.0f);
      return input(*cg, {dim}, data);
    }else{
      return lookup(*cg, p_labels, label_id);
    }
  }
  ComputationGraph* cg;
  LookupParameters* p_labels;
};

template <class Builder>
struct BiTrans {
  Builder l2rbuilder;
  Builder r2lbuilder;
  Parameters* p_f2c;
  Parameters* p_r2c;
  Parameters* p_cb;


  explicit BiTrans(Model& model) :
      l2rbuilder(LAYERS, INPUT_DIM, XCRIBE_DIM, &model),
      r2lbuilder(LAYERS, INPUT_DIM, XCRIBE_DIM, &model) {
    p_f2c = model.add_parameters({XCRIBE_DIM, XCRIBE_DIM});
    p_r2c = model.add_parameters({XCRIBE_DIM, XCRIBE_DIM});
    p_cb = model.add_parameters({XCRIBE_DIM});
  }

  vector<Expression> transcribe(ComputationGraph& cg, const vector<Expression>& x, Expression start, Expression end, Expression& start_c, Expression& end_c, bool use_dropout, float dropout_rate = 0) {
    l2rbuilder.new_graph(cg);
    if(use_dropout){
      l2rbuilder.set_dropout(dropout_rate);
    }else{
      l2rbuilder.disable_dropout();
    }
    l2rbuilder.start_new_sequence();
    r2lbuilder.new_graph(cg);
    if(use_dropout){
      r2lbuilder.set_dropout(dropout_rate);
    }else{
      r2lbuilder.disable_dropout();
    }
    r2lbuilder.start_new_sequence();
    Expression f2c = parameter(cg, p_f2c);
    Expression r2c = parameter(cg, p_r2c);
    Expression cb = parameter(cg, p_cb);

    const int len = x.size();
    vector<Expression> fwd(len), rev(len), res(len);

    Expression l2r_start = l2rbuilder.add_input(start);
    for (int i = 0; i < len; ++i)
      fwd[i] = l2rbuilder.add_input(x[i]);
    Expression l2r_end = l2rbuilder.add_input(end);

    Expression r2l_start = r2lbuilder.add_input(end);
    for (int i = len - 1; i >= 0; --i)
      rev[i] = r2lbuilder.add_input(x[i]);
    Expression r2l_end = r2lbuilder.add_input(start);

    for (int i = 0; i < len; ++i){
      if(bias_mode == 0 || bias_mode == 3){
        res[i] = affine_transform({cb, f2c, fwd[i], r2c, rev[i]});
      }else if (bias_mode == 1){
        // biased to the left, so no r to l information
        res[i] = affine_transform({cb, f2c, fwd[i]});
      }else if (bias_mode == 2){
        res[i] = affine_transform({cb, r2c, rev[i]});
      }
    }

    if (bias_mode == 0 || bias_mode == 3){
      start_c = affine_transform({cb, f2c, l2r_start, r2c, r2l_end});
      end_c = affine_transform({cb, f2c, l2r_end, r2c, r2l_start});
    }else if(bias_mode == 1){
      start_c = affine_transform({cb, f2c, l2r_start});
      end_c = affine_transform({cb, f2c, l2r_end});
    }else if(bias_mode == 2){
      start_c = affine_transform({cb, r2c, r2l_end});
      end_c = affine_transform({cb, r2c, r2l_start});
    }

    return res;
  }
};

template <class Builder>
struct ModelOne {
  SymbolEmbedding* xe;
  SymbolEmbedding* ye;
  BiTrans<Builder> bt;
  Builder tagrnn;

  Parameters* p_thbias;
  Parameters* p_cth;
  Parameters* p_h_rnn_lastth;
  Parameters* p_tbias;
  Parameters* p_th2t;

  cnn::Dict d;
  cnn::Dict td;
  explicit ModelOne(Model& model, cnn::Dict& d_, cnn::Dict& td_) :
      bt(model),
      tagrnn(LAYERS, TAG_DIM, TAG_RNN_HIDDEN_DIM, &model){
    d = d_;
    td = td_;
    xe = new SymbolEmbedding(model, d.size(), INPUT_DIM);
    if (use_pretrained_embeding) {
       xe->load_embedding(d, pretrained_embeding);
    }
    if (bias_mode == 0 || bias_mode == 1 || bias_mode == 2){
      ye = new SymbolEmbedding(model, td.size(), TAG_DIM);
    }else if (bias_mode == 3){
      // zero out the y embeddings, this gives no information to the next step, the model can only rely on the x
      ye = new SymbolEmbedding(model, td.size(), TAG_DIM, true);
    }

    p_cth = model.add_parameters({TAG_HIDDEN_DIM, XCRIBE_DIM});
    p_h_rnn_lastth = model.add_parameters({TAG_HIDDEN_DIM, TAG_RNN_HIDDEN_DIM});
    p_thbias = model.add_parameters({TAG_HIDDEN_DIM});
    p_tbias = model.add_parameters({td.size()});
    p_th2t = model.add_parameters({td.size(), TAG_HIDDEN_DIM});
  }

  vector<Expression> ConstructInput(const vector<int>& x,
                                  ComputationGraph& cg){
  xe->new_graph(cg);
  vector<Expression> xins(x.size());
  for (int i = 0; i < x.size(); ++i)
    xins[i] = xe->embed(x[i]);
    return xins;
  }

  // return Expression of total loss
  Expression ComputeLoss(vector<Expression>& xins,
                            const vector<int>& y,
                            vector<int>& y_decode,
                            ComputationGraph& cg,
                            bool use_dropout,
                            float dropout_rate = 0,
                            bool training_mode = true,
                            float self_normalize_alpha_v = 0,
                            float tau_v = 0) {
    int len = xins.size();
    y_decode.clear();

    Expression i_thbias = parameter(cg, p_thbias);
    Expression i_cth = parameter(cg, p_cth);
    Expression i_h_rnn_lastth = parameter(cg, p_h_rnn_lastth);
    Expression i_tbias = parameter(cg, p_tbias);
    Expression i_th2t = parameter(cg, p_th2t);

    vector<Expression> errs;
    Expression start_c, end_c;
    vector<Expression> c = bt.transcribe(cg, xins, xe->embed(TOKENSOS), xe->embed(TOKENEOS), start_c, end_c, use_dropout, dropout_rate);
    
    // Construct the forward rnn over y tags
    tagrnn.new_graph(cg);  // reset RNN builder for new graph
    if(use_dropout){
      tagrnn.set_dropout(dropout_rate);
    }else{
      tagrnn.disable_dropout();
    }
    tagrnn.start_new_sequence();
    ye->new_graph(cg);

    tagrnn.add_input(ye->embed(TAGSOS));

    // float total_nz = 0;
    for(unsigned int t = 0; t < len+1; ++t){
      // first compute the y tag at this step
      // the factor -- last time h from forward tag rnn
      Expression h_rnn_last = tagrnn.back();
      // the factor -- c from the bi-rnn at this time step
      Expression i_th = tanh(affine_transform({i_thbias, i_cth, t < len ? c[t] : end_c, i_h_rnn_lastth, h_rnn_last}));
      Expression i_t = affine_transform({i_tbias, i_th2t, i_th});

      Expression i_err;
      if(training_mode && self_normalize_alpha_v > 0){
        Expression self_normalize_alpha = input(cg, self_normalize_alpha_v);
        Expression nz = square(log(sum_cols(exp(transpose(i_t)))));
        i_err = pickneglogsoftmax(i_t, t < len ? y[t] : TAGEOS) + self_normalize_alpha * nz;
        // float nz_f = as_scalar(cg.get_value(nz.i));
        // total_nz += nz_f;
        // cerr << nz_f << endl;
      }else{
        i_err = pickneglogsoftmax(i_t, t < len ? y[t] : TAGEOS);
      } 
      Expression t_err;

      if (tau_v == 0){
        t_err = i_err;
      }else{
        Expression tau = input(cg, tau_v);
        Expression e_p = softmax(i_t);
        Expression e_logp = log(e_p);
        Expression ent = transpose(e_p) * e_logp;
        t_err = i_err - tau * ent;
      }
      errs.push_back(t_err);

      
      // In this block, we decode the y tag at this step and put the corresponding y embedding on to tag rnn
      if(t < len){
        set<int> skip_ind;
        vector<float> dist = as_vector(cg.get_value(i_t.i));
        double best = -9e99;
        int besti = -1;
        bool resample = false;
        do{
          // either insert -1 or the tag last time
          skip_ind.insert(besti);
          best = -9e99;
          besti = -1;
          resample = false;
          for (int i = 0; i < dist.size(); ++i) {
            if(skip_ind.find(i) != skip_ind.end()){
              // skip this index
              continue;
            }
            if (dist[i] > best) { best = dist[i]; besti = i; }
          }
          if(ner_tagging){
            pair<string, string> res;
            extract_tag(besti, td, res);
            if(t == 0){
              if(res.first.compare("I") == 0 || res.first.compare("E") == 0){
                resample = true;
                continue;
              }
            }else if(t > 0){
              if(res.first.compare("I") == 0){
                int previous_tag = y_decode.back();
                pair<string, string> prev;
                extract_tag(previous_tag, td, prev);
                if(previous_tag == besti || (prev.second.compare(res.second) == 0 && prev.first.compare("B") == 0) ){
                  resample = false;
                }else{
                  resample = true;
                  continue;
                }
              }else if(res.first.compare("E") == 0){
                int previous_tag =  y_decode.back();
                pair<string, string> prev;
                extract_tag(previous_tag, td, prev);
                if(prev.second.compare(res.second) == 0 && (prev.first.compare("I") == 0 || prev.first.compare("B") == 0)){
                  resample = false;
                }else{
                  resample = true;
                  continue;
                }
              }
            }
          }
        } while(resample || (int) besti == TAGSOS || (int) besti == TAGEOS);
        // assert(besti != TAGSOS);
        y_decode.push_back(besti);

        // add to the tag rnn
        if(training_mode){
          // in the training mode, push the gold tag
          tagrnn.add_input(ye->embed(y[t]));
        }else{
          // in the decoding mode, we can only push the predicted tag
          tagrnn.add_input(ye->embed(besti));
        }

      }
    }
    // cerr << "here:\t " << total_nz << endl;
    return sum(errs);
  }

  float ComputeZ(vector<Expression>& xins,
                 const vector<int>& y,
                 ComputationGraph& cg) {
    int len = xins.size();

    Expression i_thbias = parameter(cg, p_thbias);
    Expression i_cth = parameter(cg, p_cth);
    Expression i_h_rnn_lastth = parameter(cg, p_h_rnn_lastth);
    Expression i_tbias = parameter(cg, p_tbias);
    Expression i_th2t = parameter(cg, p_th2t);

    vector<Expression> errs;
    Expression start_c, end_c;
    vector<Expression> c = bt.transcribe(cg, xins, xe->embed(TOKENSOS), xe->embed(TOKENEOS), start_c, end_c, false, 0);
    
    // Construct the forward rnn over y tags
    tagrnn.new_graph(cg);  // reset RNN builder for new graph

    tagrnn.disable_dropout();
    
    tagrnn.start_new_sequence();
    ye->new_graph(cg);

    tagrnn.add_input(ye->embed(TAGSOS));

    for(unsigned int t = 0; t < len+1; ++t){
      // first compute the y tag at this step
      // the factor -- last time h from forward tag rnn
      Expression h_rnn_last = tagrnn.back();
      // the factor -- c from the bi-rnn at this time step
      Expression i_th = tanh(affine_transform({i_thbias, i_cth, t < len ? c[t] : end_c, i_h_rnn_lastth, h_rnn_last}));
      Expression i_t = affine_transform({i_tbias, i_th2t, i_th});

      Expression i_err;
      i_err = square(log(sum_cols(exp(transpose(i_t)))));
      errs.push_back(i_err);
      // In this block, we decode the y tag at this step and put the corresponding y embedding on to tag rnn
      if(t < len){
          tagrnn.add_input(ye->embed(y[t]));
      }
    }
    Expression total_z = sum(errs);
    return sqrt(as_scalar(cg.get_value(total_z.i)));
  }

  void Sample(const vector<int> x,
              vector<int>& y_decode){
    y_decode.clear();
    ComputationGraph cg;
    vector<Expression> xins = ConstructInput(x, cg);
    int len = xins.size();

    Expression i_thbias = parameter(cg, p_thbias);
    Expression i_cth = parameter(cg, p_cth);
    Expression i_h_rnn_lastth = parameter(cg, p_h_rnn_lastth);
    Expression i_tbias = parameter(cg, p_tbias);
    Expression i_th2t = parameter(cg, p_th2t);

    Expression annel = input(cg, 0.8f);
    Expression start_c, end_c;
    vector<Expression> c = bt.transcribe(cg, xins, xe->embed(TOKENSOS), xe->embed(TOKENEOS), start_c, end_c, false, 0.0f);
    
    // Construct the forward rnn over y tags
    tagrnn.new_graph(cg);  // reset RNN builder for new graph
    tagrnn.disable_dropout();

    tagrnn.start_new_sequence();
    ye->new_graph(cg);
    tagrnn.add_input(ye->embed(TAGSOS));

    for(unsigned int t = 0; t < len; ++t){
      // first compute the y tag at this step
      // the factor -- last time h from forward tag rnn
      Expression h_rnn_last = tagrnn.back();
      // the factor -- c from the bi-rnn at this time step
      Expression i_th = tanh(affine_transform({i_thbias, i_cth, t < len ? c[t] : end_c, i_h_rnn_lastth, h_rnn_last}));
      Expression i_t = affine_transform({i_tbias, i_th2t, i_th}) * annel;

      {
        Expression res = softmax(i_t);
        vector<float> dist = as_vector(cg.get_value(res.i));
        unsigned sample_i = 0;
        do {
          sample_i = SampleFromDist(dist);
        } while((int) sample_i == TOKENSOS || (int) sample_i == TOKENEOS);
        y_decode.push_back((int)sample_i);
        // add to the tag rnn
        tagrnn.add_input(ye->embed(sample_i));
      }
    }
    return;

  }
  void SampleParallel(const vector<int> x,
                      vector<vector<int>>& y_samples,
                      vector<float>& y_scores,
                      int sample_num,
                      string model_file_prefix,
                      float annel_value = 1.0f,
                      bool use_dropout = false,
                      float dropout_rate = 0){
    ComputationGraph cg;
    vector<Expression> xins = ConstructInput(x, cg);
    int len = xins.size();

    Expression i_thbias = parameter(cg, p_thbias);
    Expression i_cth = parameter(cg, p_cth);
    Expression i_h_rnn_lastth = parameter(cg, p_h_rnn_lastth);
    Expression i_tbias = parameter(cg, p_tbias);
    Expression i_th2t = parameter(cg, p_th2t);

    // Expression alpha = input(cg, annel_value);
    Expression annel = input(cg, annel_value);
    Expression start_c, end_c;
    vector<Expression> c = bt.transcribe(cg, xins, xe->embed(TOKENSOS), xe->embed(TOKENEOS), start_c, end_c, use_dropout, dropout_rate);
    string sufix = to_string(getpid()) + to_string(rand());
    string mem_name_int = "MySharedMemoryInt_" + sufix;
    string vector_name_int = "MyVectorInt_" + sufix;
    string mem_name_float = "MySharedMemoryFloat_" + sufix;
    string vector_name_float = "MyVectorFloat_" + sufix;

    assert (cnn::ps->is_shared());

    ip::message_queue::remove(mem_name_int.c_str());
    ip::message_queue::remove(mem_name_float.c_str());

    //Create a new segment with given name and size
    ip::managed_shared_memory segment_int(ip::create_only, mem_name_int.c_str(), 2 * sizeof(int) * len * sample_num);
    ip::managed_shared_memory segment_float(ip::create_only, mem_name_float.c_str(), 2 * sizeof(float) * sample_num);
    //Initialize shared memory STL-compatible allocator
    const ShmemAllocatorInt alloc_inst_int (segment_int.get_segment_manager());
    const ShmemAllocatorFloat alloc_inst_float (segment_float.get_segment_manager());

    //Construct a vector named vector_name_int in shared memory with argument alloc_inst_int
    MyVectorInt *myvector_int = segment_int.construct<MyVectorInt>(vector_name_int.c_str())(alloc_inst_int);
    MyVectorFloat *myvector_float = segment_float.construct<MyVectorFloat>(vector_name_float.c_str())(alloc_inst_float);

    for(int i = 0; i < sample_num * len; ++i){
       myvector_int->push_back(0);
    }
    for(int i = 0; i < sample_num; ++i){
       myvector_float->push_back(9e99);
    }

    vector<pid_t> cp_ids; 
    // Fork to many processes
    pid_t pid;
    unsigned cid;
    for (cid = 0; cid < sample_num; ++cid) {
      pid = fork();
      if (pid == -1) {
        cerr << "Fork failed. Exiting ..." << std::endl;
        abort();
      }
      else if (pid == 0) {
        srand(time(NULL) ^ (getpid()<<16) + cid * 2);

        ip::managed_shared_memory segment_in_child_int(ip::open_only, mem_name_int.c_str());
        ip::managed_shared_memory segment_in_child_float(ip::open_only, mem_name_float.c_str());

        //Find the vector using the c-string name
        MyVectorInt *myvector_int_in_child = segment_in_child_int.find<MyVectorInt>(vector_name_int.c_str()).first;
        MyVectorFloat *myvector_float_in_child = segment_in_child_float.find<MyVectorFloat>(vector_name_float.c_str()).first;

        float sen_log_prob = 0;

        //child
        tagrnn.new_graph(cg);  // reset RNN builder for new graph
        if(use_dropout){
          tagrnn.set_dropout(dropout_rate);
        }else{
          tagrnn.disable_dropout();
        }
        tagrnn.start_new_sequence();
        ye->new_graph(cg);
        tagrnn.add_input(ye->embed(TAGSOS));

        for(unsigned int t = 0; t < len+1; ++t){
          // first compute the y tag at this step
          // the factor -- last time h from forward tag rnn
          Expression h_rnn_last = tagrnn.back();
          // the factor -- c from the bi-rnn at this time step
          Expression i_th = tanh(affine_transform({i_thbias, i_cth, t < len ? c[t] : end_c, i_h_rnn_lastth, h_rnn_last}));
          Expression i_t = affine_transform({i_tbias, i_th2t, i_th}) * annel;
          Expression i_t_annel = affine_transform({i_tbias, i_th2t, i_th}) * annel;
          unsigned sample_i = 0;
          Expression res = softmax(i_t);
          vector<float> dist = as_vector(cg.get_value(res.i));

          Expression res_annel = softmax(i_t_annel);
          vector<float> dist_annel = as_vector(cg.get_value(res_annel.i));

          bool resample;
          if(t < len){
            do {
              resample = false;
              sample_i = SampleFromDist(dist_annel);
              if(ner_tagging){
                pair<string, string> res;
                extract_tag(sample_i, td, res);
                if(t == 0){
                  if(res.first.compare("I") == 0 || res.first.compare("E") == 0){
                    resample = true;
                    continue;
                  }
                }else if(t > 0){
                  if(res.first.compare("I") == 0){
                    int previous_tag = (*myvector_int_in_child)[cid * len + t-1];
                    pair<string, string> prev;
                    extract_tag(previous_tag, td, prev);
                    if(previous_tag == sample_i || (prev.second.compare(res.second) == 0 && prev.first.compare("B") == 0) ){
                      resample = false;
                    }else{
                      resample = true;
                      continue;
                    }
                  }else if(res.first.compare("E") == 0){
                    int previous_tag = (*myvector_int_in_child)[cid * len + t-1];
                    pair<string, string> prev;
                    extract_tag(previous_tag, td, prev);
                    if(prev.second.compare(res.second) == 0 && (prev.first.compare("I") == 0 || prev.first.compare("B") == 0)){
                      resample = false;
                    }else{
                      resample = true;
                      continue;
                    }
                  }
                }
              }
            } while(resample || (int) sample_i == TAGSOS || (int) sample_i == TAGEOS);
            (*myvector_int_in_child)[cid * len + t] = (int)sample_i;
            // add to the tag rnn
            tagrnn.add_input(ye->embed(sample_i));
          }
          sen_log_prob = sen_log_prob + (t < len ? log(dist[sample_i]) : log(dist[TAGEOS]));
        }

        float v_e = -sen_log_prob;
        (*myvector_float_in_child)[cid] = v_e;
        exit(0);
      }else if (pid > 0){
          cp_ids.push_back(pid);
      }
    }
    assert(cp_ids.size() == sample_num);
    for (auto cp_id : cp_ids){
      int returnStatus;    
      waitpid(cp_id, &returnStatus, 0);  // Parent process waits here for child to terminate.
    }

   // for (int i = 0; i < sample_num * len; ++i){
   //  cerr << "parent " << "i = " << (*myvector_int)[i]  << endl;
   // }
   for (unsigned int i = 0; i < sample_num; ++i){
    for (unsigned int j = 0; j < len; ++j){
      y_samples[i].push_back((*myvector_int)[i * len + j]);
    }
   }
   for (unsigned int i = 0; i < sample_num; ++i){
    y_scores.push_back((*myvector_float)[i]);
   }
   ip::message_queue::remove(mem_name_int.c_str());
   ip::message_queue::remove(mem_name_float.c_str());
   return;
  }

  void rerank_parallel(const vector<int> &x,
                       vector<int> &y_pred,
                       unsigned int sent_id,
                       const vector<vector<int>>& reranking_set,
                       unsigned int sample_num,
                       string model_file_prefix){
    ComputationGraph cg;
    vector<Expression> xins = ConstructInput(x, cg);
    int len = xins.size();

    Expression i_thbias = parameter(cg, p_thbias);
    Expression i_cth = parameter(cg, p_cth);
    Expression i_h_rnn_lastth = parameter(cg, p_h_rnn_lastth);
    Expression i_tbias = parameter(cg, p_tbias);
    Expression i_th2t = parameter(cg, p_th2t);

    Expression start_c, end_c;
    vector<Expression> c = bt.transcribe(cg, xins, xe->embed(TOKENSOS), xe->embed(TOKENEOS), start_c, end_c, false, 0);
    string sufix = to_string(getpid()) + "_d_" + to_string(rand());
    string mem_name_float = "MySharedMemoryFloat_" + sufix;
    string vector_name_float = "MyVectorFloat_" + sufix;

    assert (cnn::ps->is_shared());

    ip::message_queue::remove(mem_name_float.c_str());

    //Create a new segment with given name and size
    ip::managed_shared_memory segment_float(ip::create_only, mem_name_float.c_str(), 2 * sizeof(float) * sample_num);
    //Initialize shared memory STL-compatible allocator
    const ShmemAllocatorFloat alloc_inst_float (segment_float.get_segment_manager());

    MyVectorFloat *myvector_float = segment_float.construct<MyVectorFloat>(vector_name_float.c_str())(alloc_inst_float);

    for(int i = 0; i < sample_num; ++i){
       myvector_float->push_back(9e99);
    }

    vector<pid_t> cp_ids; 
    // Fork to many processes
    pid_t pid;
    unsigned cid;
    for (cid = 0; cid < sample_num; ++cid) {
      pid = fork();
      if (pid == -1) {
        cerr << "Fork failed. Exiting ..." << std::endl;
        abort();
      }
      else if (pid == 0) {
        ip::managed_shared_memory segment_in_child_float(ip::open_only, mem_name_float.c_str());

        //Find the vector using the c-string name
        MyVectorFloat *myvector_float_in_child = segment_in_child_float.find<MyVectorFloat>(vector_name_float.c_str()).first;

        float sen_log_prob = 0;
        tagrnn.new_graph(cg);
        //child
        tagrnn.disable_dropout();
        
        tagrnn.start_new_sequence();
        ye->new_graph(cg);
        tagrnn.add_input(ye->embed(TAGSOS));

        for(unsigned int t = 0; t < len+1; ++t){
          // first compute the y tag at this step
          // the factor -- last time h from forward tag rnn
          Expression h_rnn_last = tagrnn.back();
          // the factor -- c from the bi-rnn at this time step
          Expression i_th = tanh(affine_transform({i_thbias, i_cth, t < len ? c[t] : end_c, i_h_rnn_lastth, h_rnn_last}));
          Expression i_t = affine_transform({i_tbias, i_th2t, i_th});
          
          Expression res = softmax(i_t);
          vector<float> dist = as_vector(cg.get_value(res.i));
          if(t < len){
            tagrnn.add_input(ye->embed(reranking_set[sent_id * sample_num + cid][t]));
          }
          sen_log_prob = sen_log_prob + (t < len ? log(dist[reranking_set[sent_id * sample_num + cid][t]]) : log(dist[TAGEOS]));
        }
        float v_e = -sen_log_prob;
        (*myvector_float_in_child)[cid] = v_e;
        exit(0);
      }else if (pid > 0){
          cp_ids.push_back(pid);
      }
    }
    assert(cp_ids.size() == sample_num);
    for (auto cp_id : cp_ids){
      int returnStatus;    
      waitpid(cp_id, &returnStatus, 0);  // Parent process waits here for child to terminate.
    }
    float best_score = 9e99;
    unsigned best_i = 0;
   for (unsigned int i = 0; i < sample_num; ++i){
    if((*myvector_float)[i] < best_score){
      best_score = (*myvector_float)[i];
      best_i = i;
    }
   }
   y_pred = reranking_set[sent_id * sample_num + best_i];
   ip::message_queue::remove(mem_name_float.c_str());
  }
};

pair<vector<int>,vector<int>> ParseTrainingInstance(const std::string& line, cnn::Dict& d, cnn::Dict& td) {
  bool test_only = false;
  std::istringstream in(line);
  std::string word;
  std::string sep = "|||";
  vector<int> x;
  vector<int> y;
  while(1) {
    in >> word;
    if (!test_only){
      if (word == sep) break;
    }else{
      if (word == sep) break;
      if (!in) break;
    }
    x.push_back(d.Convert(word));
  }
  if(!test_only){
    while(1) {
      in >> word;
      if (!in) break;
      int tag = td.Convert(word);
      y.push_back(tag);
    }
  }
  return make_pair(x, y);
}
double evaluate_POS(vector<vector<int>>& y_preds,
                vector<vector<int>>& y_golds,
                cnn::Dict& d,
                cnn::Dict& td){
  assert(y_preds.size() == y_golds.size());
  int correct = 0;
  int total = 0;
  for (unsigned int i = 0; i < y_preds.size(); i++){
    assert(y_preds[i].size() == y_golds[i].size());
    total += y_preds[i].size();
    for (unsigned int j = 0; j < y_preds[i].size(); j++){
      correct += ((y_preds[i][j] == y_golds[i][j]) ? 1 : 0);
    }
  }
  double f = (double)(correct) / (double)(total);
  cerr << "acc: " << f << endl;
  return f;
}



void extract_ners(vector<int> tags, cnn::Dict& td, std::set<pair<pair<int, int>, string>>& ners){
  int i = 0;

  // cerr << endl;
  // for (auto y_tag :tags){
  //   cerr << td.Convert(y_tag) << " ";
  // }
  // cerr << endl;

  while(i < tags.size()){
    pair<string, string> res;
    if (extract_tag(tags[i], td, res)){
      i++;
      continue;
    }
    if(res.first.compare("S") == 0){
      ners.insert(make_pair(make_pair(i, i+1), res.second));
      i++;
      continue;
    }
    if(res.first.compare("B") == 0 || res.first.compare("I") == 0 || res.first.compare("E") == 0){
      int j;
      for(j = i+1; j < tags.size()+1; j++){
        if (j == tags.size()){
          break;
        }
        pair<string, string> res_j;
        if(extract_tag(tags[j], td, res_j) || res_j.first.compare("B") == 0 || res_j.first.compare("S") == 0){
          break;
        }
      }
      ners.insert(make_pair(make_pair(i,j), res.second));
      i = j;
    }
  }
}

double evaluate_NER(vector<vector<int>>& y_preds,
                vector<vector<int>>& y_golds,
                cnn::Dict& d,
                cnn::Dict& td){
  assert(y_preds.size() == y_golds.size());
  int p_total = 0;
  int p_correct = 0;
  int r_total = 0;
  int r_correct = 0;
  for (unsigned int i = 0; i < y_golds.size(); i++){
    //cerr << "eval " << i << "/" << y_golds.size() << endl;
    std::set<pair<pair<int, int>, string>> gold_ners;
    extract_ners(y_golds[i], td, gold_ners);
    std::set<pair<pair<int, int>, string>> pred_ners;
    extract_ners(y_preds[i], td, pred_ners);

    // cerr << endl;
    // for (auto y_tag : y_golds[i]){
    //   cerr << td.Convert(y_tag) << " ";
    // }
    // cerr << endl;
    // for (auto y_ner : gold_ners){
    //   cerr << y_ner.first.first << " " << y_ner.first.second << " " << y_ner.second << endl;
    // }

    // cerr << endl;
    // for (auto y_tag : y_preds[i]){
    //   cerr << td.Convert(y_tag) << " ";
    // }
    // cerr << endl;
    // for (auto y_ner : pred_ners){
    //   cerr << y_ner.first.first << " " << y_ner.first.second << " " << y_ner.second << endl;
    // }
    // cerr << endl;

    for (auto pred : pred_ners){
      if(gold_ners.find(pred) != gold_ners.end()){
        p_correct++;
      }
      p_total++;
    }
    for (auto gold : gold_ners){
      if(pred_ners.find(gold) != pred_ners.end()){
        r_correct++;
      }
      r_total++;
    }
  }
  double p = (double)(p_correct) / (double)(p_total);
  double r = (double)(r_correct) / (double)(r_total);
  double f = 2.0 * ((p * r) / (p + r));
  cerr << "p: " << p << "\tr: " << r << "\tf: " << f << endl;
  return f;
}


double predict_and_evaluate(ModelOne<LSTMBuilder>& modelone,
                            const vector<pair<vector<int>,vector<int>>>&input_set,
                            int sample_num,
                            string model_file_prefix,
                            float annel_value,
                            string set_name = "DEV"
                            ){
  // vector<vector<int>> y_preds;
  // vector<vector<int>> y_golds;
  // for (auto& sent : input_set) {
  //   vector<float> y_scores;
  //   vector<vector<int>> y_samples(sample_num);
  //   modelone.SampleParallel(sent.first, y_samples, y_scores, sample_num, model_file_prefix, annel_value, false, 0);
  //   assert(y_samples.size() == y_scores.size());
  //   float min_v = 9e99;
  //   unsigned int min_i = 0;
  //   for(unsigned int i = 0; i < y_scores.size(); ++i){
  //     if(y_scores[i] < min_v){
  //       min_v = y_scores[i];
  //       min_i = i;
  //     }
  //   }
  //   y_golds.push_back(sent.second);
  //   y_preds.push_back(y_samples[min_i]);
  // }
  // double f = evaluate(y_preds, y_golds, modelone.d, modelone.td);
  // cerr << set_name << endl;
  // return f;
  vector<vector<int>> y_preds;
  vector<vector<int>> y_golds;
  for (auto& sent : input_set) {
    ComputationGraph cg;
    vector<int> y_pred;
    vector<Expression> xins = modelone.ConstructInput(sent.first, cg);
    modelone.ComputeLoss(xins, sent.second, y_pred, cg, false, 0, false, 0);
    y_golds.push_back(sent.second);
    y_preds.push_back(y_pred);
  }
  double f = ner_tagging ? evaluate_NER(y_preds, y_golds, modelone.d, modelone.td) : evaluate_POS(y_preds, y_golds, modelone.d, modelone.td);
  cerr << set_name << endl;
  return f;
}

double evaluate(vector<vector<int>>& y_preds,
                vector<vector<int>>& y_golds,
                cnn::Dict& d,
                cnn::Dict& td){
  double f = ner_tagging ? evaluate_NER(y_preds, y_golds, d, td) : evaluate_POS(y_preds, y_golds, d, td);
  return f;
}


void test_only(ModelOne<LSTMBuilder>& modelone,
          vector<pair<vector<int>,vector<int>>>&test_set)
{
  for (auto& sent : test_set) {
    ComputationGraph cg;
    vector<int> y_pred;
    vector<Expression> xins = modelone.ConstructInput(sent.first, cg);
    modelone.ComputeLoss(xins, sent.second, y_pred, cg, false, 0, false, 0);
    unsigned int i;
    for(i = 0; i < y_pred.size()-1; ++i){
      auto pred = modelone.td.Convert(y_pred[i]);
      cout << pred << " ";
    }
    auto pred = modelone.td.Convert(y_pred[i]);
    cout << pred;
    cout << endl;
  }
}

void read_reranking_file(string file_path,
                         cnn::Dict& d, 
                         cnn::Dict& td,
                         vector<vector<int>>& reranking_set)
{
  reranking_set.clear();
  string line;
  cerr << "Reading reranking list from " << file_path << "...\n";
  {
    ifstream in(file_path);
    assert(in);
    while(getline(in, line)) {
      vector <string> fields;
      boost::algorithm::trim(line);
      boost::algorithm::split(fields, line, boost::algorithm::is_any_of( " " ));
      vector<int> tags;
      for(string m : fields){
        tags.push_back(td.Convert(m));
      }
      reranking_set.push_back(tags);
    }
  }
}

void test_and_evaluate(ModelOne<LSTMBuilder>& modelone,
                       vector<pair<vector<int>,vector<int>>>&test_set,
                       vector<vector<int>>& reranking_set,
                       unsigned int sample_num,
                       cnn::Dict& d,
                       cnn::Dict& td,
                       string model_file_prefix){

  if(greedy_decoding){
      vector<vector<int>> y_preds;
      vector<vector<int>> y_golds;
      for (auto& sent : test_set) {
        ComputationGraph cg;
        vector<int> y_pred;
        vector<Expression> xins = modelone.ConstructInput(sent.first, cg);
        modelone.ComputeLoss(xins, sent.second, y_pred, cg, false, 0, false, 0);
        y_golds.push_back(sent.second);
        y_preds.push_back(y_pred);
        string output_s = "";
        for(int j = 0; j < y_pred.size(); j++){
          output_s += (td.Convert(y_pred[j]) + " ");
        }
        boost::algorithm::trim(output_s);
        cout << output_s << endl;
      }
      evaluate(y_preds, y_golds, d, td);
  }else{
    cerr << "start testing" << endl;
    assert(test_set.size() * sample_num == reranking_set.size());
    vector<vector<int>> y_preds;
    vector<vector<int>> y_golds;
    for(int sent_id = 0; sent_id < test_set.size(); sent_id++){
      cerr << "test sentence " << sent_id << endl;
      auto sent = test_set[sent_id];
      auto x = sent.first;
      vector<int> y_pred;
      modelone.rerank_parallel(x, y_pred, sent_id, reranking_set, sample_num, model_file_prefix);
      y_preds.push_back(y_pred);
      string output_s = "";
      for(int j = 0; j < y_pred.size(); j++){
        output_s += (td.Convert(y_pred[j]) + " ");
      }
      boost::algorithm::trim(output_s);
      cout << output_s << endl;
      y_golds.push_back(sent.second);
    }
    evaluate(y_preds, y_golds, d, td);
  }
}

void sample_only(ModelOne<LSTMBuilder>& modelone,
          vector<pair<vector<int>,vector<int>>>&test_set,
          unsigned int sample_num){
  for (auto& sent : test_set) {
    for (unsigned int si = 0; si < sample_num; ++si){
      vector<int> y_pred;
      modelone.Sample(sent.first, y_pred);
      unsigned int i;
      for(i = 0; i < y_pred.size()-1; ++i){
        auto pred = modelone.td.Convert(y_pred[i]);
        cout << pred << " ";
      }
      auto pred = modelone.td.Convert(y_pred[i]);
      cout << pred;
      cout << endl;
    }
  }
}


void read_file(string file_path,
                     cnn::Dict& d,
                     cnn::Dict& td,
                     vector<pair<vector<int>,vector<int>>>&read_set)
{
  read_set.clear();
  string line;
  cerr << "Reading data from " << file_path << "...\n";
  {
    ifstream in(file_path);
    assert(in);
    while(getline(in, line)) {
      read_set.push_back(ParseTrainingInstance(line, d, td));
    }
  }
  cerr << "Reading data from " << file_path << " finished \n";
}

void save_models(string model_file_prefix,
                    cnn::Dict& d,
                    cnn::Dict& td,
                    Model& model){
  cerr << "saving models..." << endl;

  const string f_name = model_file_prefix + ".params";
  ofstream out(f_name);
  boost::archive::text_oarchive oa(out);
  oa << model;
  out.close();

  const string f_d_name = model_file_prefix + ".dict";
  ofstream out_d(f_d_name);
  boost::archive::text_oarchive oa_d(out_d);
  oa_d << d;
  out_d.close();

  const string f_td_name = model_file_prefix + ".tdict";
  ofstream out_td(f_td_name);
  boost::archive::text_oarchive oa_td(out_td);
  oa_td << td;
  out_td.close();
  cerr << "saving models finished" << endl;
}

void load_models(string model_file_prefix,
                 Model& model){
  cerr << "loading models..." << endl;

  string fname = model_file_prefix + ".params";
  ifstream in(fname);
  boost::archive::text_iarchive ia(in);
  ia >> model;
  in.close();

  cerr << "loading models finished" << endl;
}

void load_dicts(string model_file_prefix,
                 cnn::Dict& d,
                 cnn::Dict& td)
{
  cerr << "loading dicts..." << endl;
  string f_d_name = model_file_prefix + ".dict";
  ifstream in_d(f_d_name);
  boost::archive::text_iarchive ia_d(in_d);
  ia_d >> d;
  in_d.close();

  string f_td_name = model_file_prefix + ".tdict";
  ifstream in_td(f_td_name);
  boost::archive::text_iarchive ia_td(in_td);
  ia_td >> td;
  in_td.close();
  cerr << "loading dicts finished" << endl;
}




int main(int argc, char** argv) {
  cnn::Initialize(argc, argv, 0, true);
  unsigned dev_every_i_reports;
  float annel_value;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help", "produce help message")
      ("train", po::bool_switch()->default_value(false), "the training mode")
      ("load_original_model", po::bool_switch()->default_value(false), "continuing the training by loading the model, only valid during training")
      ("test", po::bool_switch()->default_value(false), "the test mode")
      ("sample", po::bool_switch()->default_value(false), "the sample mode -- for each sentence, generate sample_num samples")
      ("sample_num", po::value<unsigned int>()->default_value(100), "the number of samples for each sentence")
      ("dev_every_i_reports", po::value<unsigned>(&dev_every_i_reports)->default_value(1000))
      ("evaluate_test", po::bool_switch()->default_value(false), "evaluate test set every training iteration")
      ("train_file", po::value<string>(), "path of the train file")
      ("dev_file", po::value<string>(), "path of the dev file")
      ("test_file", po::value<string>(), "path of the test file")
      ("model_file_prefix", po::value<string>(), "prefix path of the model files (and dictionaries)")
      ("upe", po::value<string>(), "use pre-trained word embeding")
      ("eta0", po::value<float>()->default_value(0.1f), "initial learning rate")
      ("eta_decay_onset_epoch", po::value<unsigned>()->default_value(8), "start decaying eta every epoch after this epoch (try 8)")
      ("eta_decay_rate", po::value<float>()->default_value(0.5f), "how much to decay eta by (recommended 0.5)")
      ("dropout_rate", po::value<float>(), "dropout rate, also indicts using dropout during training")
      ("annel_value", po::value<float>(&annel_value)->default_value(0.8f), "annel value when sampling")
      ("pm_layers", po::value<unsigned>(&LAYERS)->default_value(1), "Layer number of the lstm" )
      ("pm_input_dim", po::value<unsigned>(&INPUT_DIM)->default_value(256), "input dim")
      ("pm_xcribe_dim", po::value<unsigned>(&XCRIBE_DIM)->default_value(256), "xcribe dim")
      ("pm_rnn_hidden_dim", po::value<unsigned>(&TAG_RNN_HIDDEN_DIM)->default_value(128), "tag rnn hidden dim")
      ("pm_tag_dim", po::value<unsigned>(&TAG_DIM)->default_value(128), "tag dim")
      ("pm_tag_hidden_dim", po::value<unsigned>(&TAG_HIDDEN_DIM)->default_value(256), "tag hidden dim")
      ("ner", po::bool_switch(&ner_tagging)->default_value(false), "the ner mode")
      ("bias_mode", po::value<int>(&bias_mode)->default_value(0), "bias the model in some interesting ways, 0 means no bias, 1 means bias to the left, 2 means bias to the right, 3 means only use observation to predict")
      ("self_normalize_alpha", po::value<float>()->default_value(0), "self normalize alpha")
      ("compute_normalize_z", po::bool_switch(&ner_tagging)->default_value(false), "compute normalize z")
      ("greedy_decoding", po::bool_switch(&greedy_decoding)->default_value(false), "decoding using the greedy mode")
      ("reranking_file", po::value<string>(), "specify the reranking file")
      ("max_ent_reg", po::value<float>()->default_value(0), "the tau")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
      cerr << desc << "\n";
      return 1;
  }
  if(vm["train"].as<bool>()){
    if (vm.count("upe")){
      cerr << "using pre-trained embeding from " << vm["upe"].as<string>() << endl;
      use_pretrained_embeding = true;
      pretrained_embeding = vm["upe"].as<string>();
    }else{
      use_pretrained_embeding = false;
      cerr << "not using pre-trained embeding" << endl;
    }

    bool use_dropout = false;
    float dropout_rate = 0;
    if (vm.count("dropout_rate")){
      dropout_rate = vm["dropout_rate"].as<float>();
      use_dropout = (dropout_rate > 0);
      if(use_dropout){
        cerr << "using dropout training, dropout rate: " << dropout_rate << endl;
      }
    }else{
      use_dropout = false;
    }

    // create two dictionaries
    cnn::Dict d;
    cnn::Dict td;
    int tkSOS = d.Convert("<s>");
    int tkEOS = d.Convert("</s>");
    assert(tkSOS == 0 && tkEOS == 1);
    int tgSOS = td.Convert("<s>");
    int tgEOS = td.Convert("</s>");
    assert(tgSOS == 0 && tgEOS == 1);
    vector<pair<vector<int>,vector<int>>> training, dev, test;
    read_file(vm["train_file"].as<string>(), d, td, training);

    d.Freeze();  // no new word types allowed
    td.Freeze(); // no new tag types allowed
    d.SetUnk("<UNK>"); // set UNK to allow the unseen character in the dev and test set

    read_file(vm["dev_file"].as<string>(), d, td, dev);
    if (vm["evaluate_test"].as<bool>()){
      read_file(vm["test_file"].as<string>(), d, td, test);
    }
    
    float eta_decay_rate = vm["eta_decay_rate"].as<float>();
    unsigned eta_decay_onset_epoch = vm["eta_decay_onset_epoch"].as<unsigned>();

    cerr << "eta_decay_rate: " << eta_decay_rate << endl;
    cerr << "eta_decay_onset_epoch: " << eta_decay_onset_epoch << endl;
    cerr << "self_normalize_alpha: " << vm["self_normalize_alpha"].as<float>() << endl; 

    Model model;
    // auto sgd = new SimpleSGDTrainer(&model);
    // auto sgd = new AdamTrainer(&model, 1e-6, 0.0005, 0.01, 0.9999, 1e-8);
    Trainer* sgd = new SimpleSGDTrainer(&model);
    sgd->eta0 = sgd->eta = vm["eta0"].as<float>();

    ModelOne<LSTMBuilder> modelone(model, d, td);

    if(vm["load_original_model"].as<bool>()){
      load_models(vm["model_file_prefix"].as<string>(), model);
    }

    double f_best = 0;
    unsigned report_every_i = 100;

    unsigned si = training.size();
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
    bool first = true;
    int report = 0;
    unsigned lines = 0;
    int completed_epoch = -1;
    while(1) {
      Timer iteration("completed in");
      double loss = 0;
      unsigned ttags = 0;
      double correct = 0;
      for (unsigned i = 0; i < report_every_i; ++i) {
        if (si == training.size()) {
          si = 0;
          if (first) { first = false; } else { sgd->update_epoch(); }
          completed_epoch++;
          if (eta_decay_onset_epoch && completed_epoch >= (int)eta_decay_onset_epoch)
            sgd->eta *= eta_decay_rate;
          cerr << "**SHUFFLE\n" << endl;
          shuffle(order.begin(), order.end(), *rndeng);
        }

        // build graph for this instance
        ComputationGraph cg;
        auto& sent = training[order[si]];
        ++si;
        vector<Expression> xins = modelone.ConstructInput(sent.first, cg);
        vector<int> y_pred;

        modelone.ComputeLoss(xins, sent.second, y_pred, cg, use_dropout, dropout_rate, true, vm["self_normalize_alpha"].as<float>(), vm["max_ent_reg"].as<float>());
        ttags += sent.second.size();
        loss += as_scalar(cg.forward());
        cg.backward();
        sgd->update(1.0);
        ++lines;
      }
      sgd->status();
      cerr << " E = " << (loss / ttags) << " ppl=" << exp(loss / ttags) << " (acc=" << (correct / ttags) << ") " << endl;
      report++;
      if (report % dev_every_i_reports == 0) {
        double f = predict_and_evaluate(modelone, dev, vm["sample_num"].as<unsigned int>(), vm["model_file_prefix"].as<string>(), annel_value);
        // if(true){
        //   float total_len = 0;
        //   float total_z = 0;
        //   for(auto sent : dev){
        //     ComputationGraph cg;
        //     vector<Expression> xins = modelone.ConstructInput(sent.first, cg);
        //     total_z += modelone.ComputeZ(xins, sent.second, cg);
        //     total_len += (sent.first.size() + 1);
        //   }
        //   cerr << "normalize_z: " <<  (total_z/total_len) << endl;
        // }
        if (f > f_best) {
          f_best = f;
          save_models(vm["model_file_prefix"].as<string>(), d, td, model);
        }
        save_models(vm["model_file_prefix"].as<string>() + "_" + to_string(report), d, td, model);
        if (vm["evaluate_test"].as<bool>()){
          predict_and_evaluate(modelone, test, vm["sample_num"].as<unsigned int>(), vm["model_file_prefix"].as<string>(), annel_value, "TEST");
        }
      }
    }
    delete sgd;
  }else{
    use_pretrained_embeding = false;
    Model model;
    cnn::Dict d;
    cnn::Dict td;
    load_dicts(vm["model_file_prefix"].as<string>(), d, td);
    ModelOne<LSTMBuilder> modelone(model, d, td);
    load_models(vm["model_file_prefix"].as<string>(), model);
    vector<pair<vector<int>,vector<int>>> test;
    read_file(vm["test_file"].as<string>(), d, td, test);

    if(vm["test"].as<bool>()){
      vector<vector<int>> test_reranking;
      string reranking_file_name;
      if(vm.count("reranking_file")){
        reranking_file_name = vm["reranking_file"].as<string>();
      }else{
        reranking_file_name = vm["test_file"].as<string>() + ".reranking";
      }
      read_reranking_file(reranking_file_name, d, td, test_reranking);
      test_and_evaluate(modelone, test, test_reranking, vm["sample_num"].as<unsigned int>(), d, td, vm["model_file_prefix"].as<string>());
    }else if(vm["sample"].as<bool>()){
      // sample_only(modelone, test, vm["sample_num"].as<unsigned int>());
      unsigned int sample_num = vm["sample_num"].as<unsigned int>();
      vector<vector<int>> best_score_set;
      vector<vector<int>> oracle_set;
      vector<vector<int>> y_golds;
      cerr << "start sampling" << endl;
      int sent_id = 0;
      for(auto sent : test){
        cerr << "sample " << sent_id++ << endl;
        vector<vector<int>> y_samples(sample_num);
        vector<float> y_scores;
        modelone.SampleParallel(sent.first, y_samples, y_scores, sample_num, vm["model_file_prefix"].as<string>(), annel_value, false, 0);
        int best_oracle = 0;
        unsigned int best_oracle_ind = 0;
        float best_score = 9e99;
        unsigned int best_score_ind = 0;
        auto y_gold = sent.second;
        y_golds.push_back(y_gold);
        for(int i = 0; i < y_samples.size(); i++){
          auto y_sample = y_samples[i];
          assert(y_sample.size() == y_gold.size());
          int oracle_correct = 0;
          string output_s = "";
          for(int j = 0; j < y_sample.size(); j++){
            if(y_sample[j] == y_gold[j]) oracle_correct++;
            output_s += (td.Convert(y_sample[j]) + " ");
          }
          boost::algorithm::trim(output_s);
          cout << output_s << endl;
          if(oracle_correct > best_oracle){
            best_oracle = oracle_correct;
            best_oracle_ind = i;
          }
          if(y_scores[i] < best_score){
            best_score = y_scores[i];
            best_score_ind = i;
          }
        }
        best_score_set.push_back(y_samples[best_score_ind]);
        oracle_set.push_back(y_samples[best_oracle_ind]);
      }
      cerr << "decoding score: " << endl;
      evaluate(best_score_set, y_golds, d, td);
      cerr << "oracle score: " << endl;
      evaluate(oracle_set, y_golds, d, td);
      cerr << "evaluation end" << endl;
    }else if(vm["compute_normalize_z"].as<bool>()){
      float total_len = 0;
      float total_z = 0;
      for(auto sent : test){
        ComputationGraph cg;
        vector<Expression> xins = modelone.ConstructInput(sent.first, cg);
        total_z += modelone.ComputeZ(xins, sent.second, cg);
        total_len += (sent.first.size() + 1);
      }
      cerr << "normalize_z: " <<  (total_z/total_len) << endl;
    }
  }
}



