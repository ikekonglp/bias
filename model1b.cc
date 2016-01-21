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
unsigned TAG_HIDDEN_DIM = 128;
bool use_pretrained_embeding = false;
bool ner_tagging = false;
string pretrained_embeding = "";

int SampleFromDist(vector<float> dist){
  unsigned w = 0;
  double p = rand01();
  for (; w < dist.size(); ++w) {
    p -= dist[w];
    if (p < 0.0) { break; }
  }
  // this should not really happen?
  if (w == dist.size()) w = dist.size() - 1;
  return w;
}

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
struct ModelOne {
  SymbolEmbedding* xe;
  BiTrans<Builder> bt;

  Parameters* p_thbias;
  Parameters* p_cth;
  Parameters* p_tbias;
  Parameters* p_th2t;

  cnn::Dict d;
  cnn::Dict td;
  explicit ModelOne(Model& model, cnn::Dict& d_, cnn::Dict& td_) :
      bt(model){
    d = d_;
    td = td_;
    xe = new SymbolEmbedding(model, d.size(), INPUT_DIM);
    if (use_pretrained_embeding) {
       xe->load_embedding(d, pretrained_embeding);
    }
    p_cth = model.add_parameters({TAG_HIDDEN_DIM, XCRIBE_DIM});
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
                            bool training_mode = true) {
    int len = xins.size();
    y_decode.clear();

    Expression i_thbias = parameter(cg, p_thbias);
    Expression i_cth = parameter(cg, p_cth);
    Expression i_tbias = parameter(cg, p_tbias);
    Expression i_th2t = parameter(cg, p_th2t);

    vector<Expression> errs;

    vector<Expression> c = bt.transcribe(cg, xins, xe->embed(TOKENSOS), xe->embed(TOKENEOS), use_dropout, dropout_rate);

    for(unsigned int t = 0; t < len; ++t){
      // first compute the y tag at this step
      // the factor -- last time h from forward tag rnn
      // the factor -- c from the bi-rnn at this time step
      Expression i_th = tanh(affine_transform({i_thbias, i_cth, c[t]}));
      Expression i_t = affine_transform({i_tbias, i_th2t, i_th});

      // In this block, we decode the y tag at this step and put the corresponding y embedding on to tag rnn
      {
        vector<float> dist = as_vector(cg.get_value(i_t.i));
        double best = -9e99;
        int besti = -1;
        for (int i = 0; i < dist.size(); ++i) {
          if (dist[i] > best) { best = dist[i]; besti = i; }
        }
        assert(besti != TAGSOS);
        y_decode.push_back(besti);
      }
      Expression i_err = pickneglogsoftmax(i_t, y[t]);
      errs.push_back(i_err);
    }
    return sum(errs);
  }

  void Sample(const vector<int> x,
              vector<int>& y_decode){
    y_decode.clear();
    ComputationGraph cg;
    vector<Expression> xins = ConstructInput(x, cg);
    int len = xins.size();

    Expression i_thbias = parameter(cg, p_thbias);
    Expression i_cth = parameter(cg, p_cth);
    Expression i_tbias = parameter(cg, p_tbias);
    Expression i_th2t = parameter(cg, p_th2t);

    Expression alpha = input(cg, 0.8f);

    vector<Expression> c = bt.transcribe(cg, xins, xe->embed(TOKENSOS), xe->embed(TOKENEOS), false, 0.0f);

    for(unsigned int t = 0; t < len; ++t){
      Expression i_th = tanh(affine_transform({i_thbias, i_cth, c[t]}));
      Expression i_t = affine_transform({i_tbias, i_th2t, i_th}) * alpha;

      {
        Expression res = softmax(i_t);
        vector<float> dist = as_vector(cg.get_value(res.i));
        unsigned sample_i = 0;
        do {
          sample_i = SampleFromDist(dist);
        } while((int) sample_i == TOKENSOS || (int) sample_i == TOKENEOS);
        y_decode.push_back((int)sample_i);
      }
    }
    return;
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

void test_only(ModelOne<LSTMBuilder>& modelone,
          vector<pair<vector<int>,vector<int>>>&test_set)
{
  for (auto& sent : test_set) {
    ComputationGraph cg;
    vector<int> y_pred;
    vector<Expression> xins = modelone.ConstructInput(sent.first, cg);
    modelone.ComputeLoss(xins, sent.second, y_pred, cg, false, false);
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

double predict_and_evaluate(ModelOne<LSTMBuilder>& modelone,
                            const vector<pair<vector<int>,vector<int>>>&input_set,
                            string set_name = "DEV"
                            ){
  vector<vector<int>> y_preds;
  vector<vector<int>> y_golds;
  for (auto& sent : input_set) {
    ComputationGraph cg;
    vector<int> y_pred;
    vector<Expression> xins = modelone.ConstructInput(sent.first, cg);
    modelone.ComputeLoss(xins, sent.second, y_pred, cg, false, false);
    y_golds.push_back(sent.second);
    y_preds.push_back(y_pred);
  }
  double f = evaluate(y_preds, y_golds, modelone.d, modelone.td);
  cerr << set_name << endl;
  return f;
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
          cerr << "**SHUFFLE\n";
          shuffle(order.begin(), order.end(), *rndeng);
        }

        // build graph for this instance
        ComputationGraph cg;
        auto& sent = training[order[si]];
        ++si;
        vector<Expression> xins = modelone.ConstructInput(sent.first, cg);
        vector<int> y_pred;
        modelone.ComputeLoss(xins, sent.second, y_pred, cg, use_dropout, dropout_rate);
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
        double f = predict_and_evaluate(modelone, dev);
        if (f > f_best) {
          f_best = f;
          save_models(vm["model_file_prefix"].as<string>(), d, td, model);
        }
        if (vm["evaluate_test"].as<bool>()){
          predict_and_evaluate(modelone, test, "TEST");
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
    read_file(vm["test_file"].as<string>(), d, td, test, true);

    if(vm["test"].as<bool>()){
      test_only(modelone, test);
    }else if(vm["sample"].as<bool>()){
      sample_only(modelone, test, vm["sample_num"].as<unsigned int>());
    }
  }
}



