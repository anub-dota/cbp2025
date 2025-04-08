#ifndef _MULTILOG_PREDICTOR_H_
#define _MULTILOG_PREDICTOR_H_

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include<map>
#include <cmath>
#include <numeric>  // for inner_product
#include <string>
#include<cstdint>

#include "hyperparams.h"

class MultiLogisticBranchPredictor {
    private:
        int num_classifiers;
        int history_length;
        double lr;
    
        std::vector<std::vector<double>> weights;
        std::vector<double> biases;
        std::vector<double> history;
        std::map<std::pair<uint64_t,uint64_t>,double> spec_map;
    
    public:
        MultiLogisticBranchPredictor(int num_classifiers = NUM_PREDICTORS_G, int history_length = HISTORY_LENGTH_G, double learning_rate = 0.1)
            : num_classifiers(num_classifiers),
              history_length(history_length),
              lr(learning_rate),
              weights(num_classifiers, std::vector<double>(history_length, 0.0)),
              biases(num_classifiers, 0.0),
              history(history_length, 0.0)
        {}

        void setup(){}
        void terminate(){}

    
        double sigmoid(double z) {
            return 1.0 / (1.0 + std::exp(-z));
        }

        // return bool
        //uint64_t seq_no, uint8_t piece,uint64_t PC
        //put to spec map

        bool predict(uint64_t seq_no, uint8_t piece, uint64_t PC) {
            int index = PC % num_classifiers;
            const std::vector<double>& w = weights[index];
            double b = biases[index];
    
            double z = std::inner_product(w.begin(), w.end(), history.begin(), 0.0) + b;
            double prob = sigmoid(z);
            spec_map[{seq_no,piece}]=prob;
            return (prob >= 0.5) ;
        }

        // uint64_t seq_no,uint64_t piece,uint64_t PC, bool resolved_dir
        void train(uint64_t seq_no,uint64_t piece,uint64_t PC, bool resolved_dir) {
            int index = PC% num_classifiers;

            if(spec_map.find({seq_no,piece})==spec_map.end()){
                return;
            }

            double prob = spec_map[{seq_no,piece}];
            int y = (resolved_dir) ? 1 : 0;
            double gradient = prob - y;
    
            for (int i = 0; i < history_length; ++i) {
                weights[index][i] -= lr * gradient * history[i];
            }
            biases[index] -= lr * gradient;
    
            for (int i = 0; i < history_length - 1; ++i) {
                history[i] = history[i + 1];
            }
            history[history_length - 1] = (resolved_dir) ? 1.0 : -1.0 ;

            spec_map.erase({seq_no,piece});
        }
};


#endif

static MultiLogisticBranchPredictor multilog_predictor;

