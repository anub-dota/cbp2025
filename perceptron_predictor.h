#ifndef _PERCPEPTRON_PREDICTOR_H_
#define _PERCPEPTRON_PREDICTOR_H_

#include <vector>
#include <stdlib.h>
#include<cstdint>
#include<iostream>
#include <map>

#include "hyperparams.h"

class PerceptronPredictor{
    int history_length;
    int num_perceptrons;
    double threshold;
    std::vector<std::vector<double>> weights;
    std::vector<int> history;
    int pred_cnt=0;
    int upd_cnt=0;
    std::map<std::pair<uint64_t,uint64_t>, double> speculative_map;
    std::map<std::pair<uint64_t,uint64_t>,bool> mpp;
    int hihi=0;
    int jojo=0;


public:
    PerceptronPredictor(int history_length = HISTORY_LENGTH_G, int num_perceptrons = NUM_PREDICTORS_G)
        : history_length(history_length), num_perceptrons(num_perceptrons), 
          threshold(1.93 * history_length + 14), weights(num_perceptrons, std::vector<double>(history_length + 1, 0)),
          history(history_length, 0){}

    void setup(){}
    void terminate(){}


    bool predict(uint64_t seq_no, uint8_t piece,uint64_t PC){
        std::vector<double>& ind_weights = weights[PC%num_perceptrons];
        double y=ind_weights[0];
        for(int i=0;i<history_length;i++){
            y+=(ind_weights[i+1]*history[i]);

        }
        speculative_map[{seq_no,piece}] = y;
        return (y>=0);
    }


    void update(uint64_t seq_no,uint64_t piece,uint64_t PC, bool resolved_dir){

        auto it = speculative_map.find({seq_no,piece});
        if (it == speculative_map.end()) return;  //never

        std::vector<double>& ind_weights = weights[PC%num_perceptrons];
        double y=it->second;
        speculative_map.erase(it);
        int resolved_with_sign = resolved_dir?1:-1;
        if((y>=0?1:-1)!=resolved_with_sign || std::abs(y)<=threshold){
            ind_weights[0]+=resolved_with_sign;
            for(int i=0;i<history_length;i++){
                ind_weights[i+1]+=resolved_with_sign*history[i];
            }
        }

        history.erase(history.begin());
        history.push_back(resolved_with_sign);
        speculative_map.erase({seq_no,piece});
    }

};

#endif

static PerceptronPredictor perceptron_predictor;