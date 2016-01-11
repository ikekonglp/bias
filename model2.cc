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

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>


using namespace std;
using namespace cnn;
const int TOKENSOS = 0;
const int TOKENEOS = 1;
const int TAGSOS = 0;
const int TAGEOS = 1;

namespace po = boost::program_options;

unsigned LAYERS = 1;
unsigned INPUT_DIM = 128;
unsigned XCRIBE_DIM = 128;
unsigned TAG_RNN_HIDDEN_DIM = 32;
unsigned TAG_DIM = 32;
unsigned WORD_HIDDEN_DIM = 128;
bool use_pretrained_embeding = false;
bool ner_tagging = false;
string pretrained_embeding = "";

// returns embeddings of labels
struct SymbolEmbedding {
  SymbolEmbedding(Model& m, unsigned n, unsigned dim) {
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
    return lookup(*cg, p_labels, label_id);
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

  vector<Expression> transcribe(ComputationGraph& cg, const vector<Expression>& x, Expression start, Expression end, bool use_dropout, float dropout_rate = 0) {
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

    l2rbuilder.add_input(start);
    for (int i = 0; i < len; ++i)
      fwd[i] = l2rbuilder.add_input(x[i]);

    r2lbuilder.add_input(end);
    for (int i = len - 1; i >= 0; --i)
      rev[i] = r2lbuilder.add_input(x[i]);

    for (int i = 0; i < len; ++i)
      res[i] = affine_transform({cb, f2c, fwd[i], r2c, rev[i]});
    return res;
  }
};

template <class Builder>
struct ModelTwo {
  SymbolEmbedding* xe;
  SymbolEmbedding* ye;
  BiTrans<Builder> bt;
  Builder tagrnn;

  Parameters* p_bias;
  Parameters* p_R;

  Parameters* p_thbias;
  Parameters* p_yth;

  Parameters* p_tbias;
  Parameters* p_th2t;

  cnn::Dict d;
  cnn::Dict td;
  explicit ModelTwo(Model& model, cnn::Dict& d_, cnn::Dict& td_) :
      bt(model),
      tagrnn(LAYERS, TAG_DIM, TAG_RNN_HIDDEN_DIM, &model){
    d = d_;
    td = td_;
    xe = new SymbolEmbedding(model, d.size(), INPUT_DIM);
    if (use_pretrained_embeding) {
       xe->load_embedding(d, pretrained_embeding);
    }
    ye = new SymbolEmbedding(model, td.size(), TAG_DIM);
    
    p_bias = model.add_parameters({td.size()});
    p_R = model.add_parameters({td.size(), TAG_RNN_HIDDEN_DIM});
    p_thbias = model.add_parameters({WORD_HIDDEN_DIM});
    p_yth = model.add_parameters({WORD_HIDDEN_DIM, TAG_RNN_HIDDEN_DIM});
    p_tbias = model.add_parameters({d.size()});
    p_th2t = model.add_parameters({d.size(), WORD_HIDDEN_DIM});

  }

  // return Expression of total loss
  Expression ComputeLoss(const vector<int>& x,
                         const vector<int>& y,
                            ComputationGraph& cg,
                            bool use_dropout,
                            float dropout_rate = 0) {
    int len = x.size();

    Expression i_bias = parameter(cg , p_bias);
    Expression i_R = parameter(cg , p_R);
    Expression i_thbias = parameter(cg , p_thbias);
    Expression i_yth = parameter(cg , p_yth);
    Expression i_tbias = parameter(cg , p_tbias);
    Expression i_th2t = parameter(cg , p_th2t);

    vector<Expression> errs;
    // First we have an rnn generates y
    tagrnn.new_graph(cg);
    if(use_dropout){
      tagrnn.set_dropout(dropout_rate);
    }else{
      tagrnn.disable_dropout();
    }
    tagrnn.start_new_sequence();
    ye->new_graph(cg);
    tagrnn.add_input(ye->embed(TAGSOS));

    // in the last step, we want to generate a EOS tag
    for(unsigned int t = 0; t < len+1; ++t){
      Expression h_t = tagrnn.back();
      Expression u_t = affine_transform({i_bias, i_R, h_t});
      // NOTE HERE: THERE IS NO NON-LINEAR TRANSFORMATION AT THIS STEP?
      // Here we predict the tag next time
      Expression i_err = pickneglogsoftmax(u_t, t < len ? y[t] : TAGEOS);
      errs.push_back(i_err);
    }

    // Second we generate independent emissions from the y
    for(unsigned int t = 0; t < len; ++t){
      Expression y_t = ye->embed(y[t]);
      Expression i_th = tanh(affine_transform({i_thbias, i_yth, y_t}));
      Expression i_t = affine_transform({i_tbias, i_th2t, i_th});

      Expression i_err = pickneglogsoftmax(i_t, x[t]);
      errs.push_back(i_err);
    }
    return sum(errs);
  }

  void RandomSample(int max_len = 150){
    cerr << endl;

    vector<int> gen_y;
    vector<int> gen_x;

    ComputationGraph cg;
    Expression i_bias = parameter(cg , p_bias);
    Expression i_R = parameter(cg , p_R);
    Expression i_thbias = parameter(cg , p_thbias);
    Expression i_yth = parameter(cg , p_yth);
    Expression i_tbias = parameter(cg , p_tbias);
    Expression i_th2t = parameter(cg , p_th2t);
    tagrnn.new_graph(cg);
    tagrnn.start_new_sequence();
    ye->new_graph(cg);
    int len = 0;
    int cur = TAGSOS;
    while(len < max_len) {
      ++len;
      tagrnn.add_input(ye->embed(cur));
      Expression h_t = tagrnn.back();
      Expression u_t = affine_transform({i_bias, i_R, h_t});
      // NOTE HERE: THERE IS NO NON-LINEAR TRANSFORMATION AT THIS STEP?
      Expression ydist = softmax(u_t);

      unsigned w = 0;
      do {
        auto dist = as_vector(cg.get_value(ydist.i));
        double p = rand01();
        for (; w < dist.size(); ++w) {
          p -= dist[w];
          if (p < 0.0) { break; }
        }
        // this should not really happen?
        if (w == dist.size()) w = dist.size() - 1; 
      } while((int) w == TAGSOS);

      cur = w;
      if(cur == TAGEOS) break;
      gen_y.push_back(cur);
      // cerr << (len == 1 ? "" : " ") << d.Convert(cur);
      ++len;
    }

    for(unsigned int t = 0; t < gen_y.size(); ++t){
      Expression y_t = ye->embed(gen_y[t]);
      Expression i_th = tanh(affine_transform({i_thbias, i_yth, y_t}));
      Expression i_t = affine_transform({i_tbias, i_th2t, i_th});

      Expression xdist = softmax(i_t);
      unsigned w = 0;
      do {
        auto dist = as_vector(cg.get_value(xdist.i));
        double p = rand01();
        for (; w < dist.size(); ++w) {
          p -= dist[w];
          if (p < 0.0) { break; }
        }
        // this should not really happen?
        if (w == dist.size()) w = dist.size() - 1; 
      } while((int) w == TOKENSOS || (int) w == TOKENEOS);
      gen_x.push_back(w);
    }

    cerr << "Sample: " << endl;
    assert(gen_x.size() == gen_y.size());
    for(unsigned i = 0; i < gen_y.size(); ++i){
      cerr << ((i == 0) ? "" : " ") << d.Convert(gen_x[i]) << "/" << td.Convert(gen_y[i]);
    }
    cerr << endl;
  }
};

pair<vector<int>,vector<int>> ParseTrainingInstance(const std::string& line, cnn::Dict& d, cnn::Dict& td, bool test_only = false) {
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

double evaluate(vector<vector<int>>& y_preds,
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

void test_only(ModelTwo<LSTMBuilder>& modeltwo,
          vector<pair<vector<int>,vector<int>>>&test_set)
{
  // for (auto& sent : test_set) {
  //   ComputationGraph cg;
  //   vector<int> y_pred;
  //   vector<Expression> xins = modeltwo.ConstructInput(sent.first, cg);
  //   modeltwo.ComputeLoss(xins, sent.second, y_pred, cg, false);
  //   unsigned int i;
  //   for(i = 0; i < y_pred.size()-1; ++i){
  //     auto pred = y_pred[i];
  //     cout << pred << " ";
  //   }
  //   if(i >= 0 && (i == y_pred.size()-1)){
  //     auto pred = y_pred[i];
  //     cout << pred;
  //   }
  //   cout << endl;
  // }
}

void read_file(string file_path,
                     cnn::Dict& d,
                     cnn::Dict& td,
                     vector<pair<vector<int>,vector<int>>>&read_set,
                     bool test_only = false)
{
  read_set.clear();
  string line;
  cerr << "Reading data from " << file_path << "...\n";
  {
    ifstream in(file_path);
    assert(in);
    while(getline(in, line)) {
      read_set.push_back(ParseTrainingInstance(line, d, td, test_only));
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

double predict_and_evaluate(ModelTwo<LSTMBuilder>& modeltwo,
                            const vector<pair<vector<int>,vector<int>>>&input_set,
                            string set_name = "DEV"
                            ){
  for (int i = 0; i < 5; ++i){
    modeltwo.RandomSample();
  }
  // vector<vector<int>> y_preds;
  // vector<vector<int>> y_golds;
  // for (auto& sent : input_set) {
  //   ComputationGraph cg;
  //   vector<int> y_pred;
  //   vector<Expression> xins = modeltwo.ConstructInput(sent.first, cg);
  //   modeltwo.ComputeLoss(xins, sent.second, y_pred, cg, false);
  //   y_golds.push_back(sent.second);
  //   y_preds.push_back(y_pred);
  // }
  // double f = evaluate(y_preds, y_golds, modeltwo.d, modeltwo.td);
  // cerr << set_name << endl;
  // return f;
  return 0;
}


int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  unsigned dev_every_i_reports;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help", "produce help message")
      ("train", po::bool_switch()->default_value(false), "the training mode")
      ("load_original_model", po::bool_switch()->default_value(false), "continuing the training by loading the model, only valid during training")
      ("test", po::bool_switch()->default_value(false), "the test mode")
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
    
    float eta_decay_rate = vm["eta_decay_onset_epoch"].as<unsigned>();
    unsigned eta_decay_onset_epoch = vm["eta_decay_rate"].as<float>();

    cerr << "eta_decay_rate: " << eta_decay_rate << endl;
    cerr << "eta_decay_onset_epoch: " << eta_decay_onset_epoch << endl;

    Model model;
    // auto sgd = new SimpleSGDTrainer(&model);
    // auto sgd = new AdamTrainer(&model, 1e-6, 0.0005, 0.01, 0.9999, 1e-8);
    Trainer* sgd = new SimpleSGDTrainer(&model);
    sgd->eta0 = sgd->eta = vm["eta0"].as<float>();

    ModelTwo<LSTMBuilder> modeltwo(model, d, td);

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
          cerr << "**SHUFFLE\n";
          shuffle(order.begin(), order.end(), *rndeng);
        }

        // build graph for this instance
        ComputationGraph cg;
        auto& sent = training[order[si]];
        ++si;
        modeltwo.ComputeLoss(sent.first, sent.second, cg, use_dropout, dropout_rate);
        ttags += sent.second.size();
        loss += as_scalar(cg.forward());
        cg.backward();
        sgd->update(1.0);
        ++lines;
      }
      sgd->status();
      cerr << " E = " << (loss / ttags) << " ppl=" << exp(loss / ttags) << " (acc=" << (correct / ttags) << ") ";
      report++;
      if (report % dev_every_i_reports == 0) {
        double f = predict_and_evaluate(modeltwo, dev);
        if (f > f_best) {
          f_best = f;
          save_models(vm["model_file_prefix"].as<string>(), d, td, model);
        }
        if (vm["evaluate_test"].as<bool>()){
          predict_and_evaluate(modeltwo, test, "TEST");
        }
      }
    }
    delete sgd;
  }else if(vm["test"].as<bool>()){
    use_pretrained_embeding = false;
    Model model;
    cnn::Dict d;
    cnn::Dict td;
    load_dicts(vm["model_file_prefix"].as<string>(), d, td);
    ModelTwo<LSTMBuilder> modeltwo(model, d, td);
    load_models(vm["model_file_prefix"].as<string>(), model);
    vector<pair<vector<int>,vector<int>>> test;
    read_file(vm["test_file"].as<string>(), d, td, test, true);
    
    test_only(modeltwo, test);
  }
}


