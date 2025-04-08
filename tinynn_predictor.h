#ifndef _TINYNN_PREDICTOR_H_
#define _TINYNN_PREDICTOR_H_


#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <numeric>  // for inner_product
#include <string>
#include <map>
#include <cstdint>

#include "hyperparams.h"

const int NUM_PREDICTORS = NUM_PREDICTORS_G;
const int HISTORY_LENGTH = HISTORY_LENGTH_G;
const int HIDDEN_SIZE = 16;
const double LEARNING_RATE = 0.05;

double sigmoid(double z) {return 1.0 / (1.0 + exp(-z));}

double tanh_activate(double x) {return std::tanh(x);}

double tanh_derivative(double x) {
    double t = tanh_activate(x);
    return 1 - t * t;
}


class TinyNN {
public:
    std::vector<std::vector<double>> W1; // [HIDDEN_SIZE][HISTORY_LENGTH]
    std::vector<double> b1;
    std::vector<double> W2; // [HIDDEN_SIZE]
    double b2;

    TinyNN() {
        W1 = std::vector<std::vector<double>>(HIDDEN_SIZE, std::vector<double>(HISTORY_LENGTH, 0.01));
        b1 = std::vector<double>(HIDDEN_SIZE, 0.0);
        W2 = std::vector<double>(HIDDEN_SIZE, 0.01);
        b2 = 0.0;
    }
};


class MultiTinyNNBranchPredictor {
    std::vector<TinyNN> predictors;
    std::vector<double> history;

    //internal vectors
    std::map<std::pair<uint64_t,uint64_t>,std::vector<double>> spec_a1, spec_z1;
    std::map<std::pair<uint64_t,uint64_t>,double> spec_prob;


public:
    MultiTinyNNBranchPredictor() {
        predictors = std::vector<TinyNN>(NUM_PREDICTORS);
        history = std::vector<double>(HISTORY_LENGTH, 0.0);
    }
    void setup(){}
    void terminate(){}

    int hash(const std::string& addr) {
        std::hash<std::string> hasher;
        return hasher(addr) % NUM_PREDICTORS;
    }


    bool predict(uint64_t seq_no, uint8_t piece,uint64_t PC) {
        int idx = PC %  NUM_PREDICTORS;
        TinyNN& nn = predictors[idx];

        std::vector<double> z1(HIDDEN_SIZE), a1(HIDDEN_SIZE);
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            z1[i] = nn.b1[i];
            for (int j = 0; j < HISTORY_LENGTH; ++j)
                z1[i] += nn.W1[i][j] * history[j];
            a1[i] = tanh_activate(z1[i]);
        }

        double z2 = nn.b2;
        for (int i = 0; i < HIDDEN_SIZE; ++i)
            z2 += nn.W2[i] * a1[i];
        double prob = sigmoid(z2);

        spec_z1[{seq_no,piece}] = z1;
        spec_a1[{seq_no,piece}] = a1;
        spec_prob[{seq_no,piece}] = prob;

        return (prob >= 0.5);
    }

    void train(uint64_t seq_no,uint64_t piece,uint64_t PC, bool resolved_dir) {
        TinyNN& nn = predictors[PC % NUM_PREDICTORS];
        double y = (resolved_dir) ? 1.0 : 0.0;
        double error = spec_prob[{seq_no,piece}] - y;

        std::vector<double>& a1 = spec_a1[{seq_no,piece}];
        std::vector<double>& z1 = spec_z1[{seq_no,piece}];

        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            nn.W2[i] -= LEARNING_RATE * error * a1[i];
        }
        nn.b2 -= LEARNING_RATE * error;

        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            double dz = error * nn.W2[i] * tanh_derivative(z1[i]);
            for (int j = 0; j < HISTORY_LENGTH; ++j) {
                nn.W1[i][j] -= LEARNING_RATE * dz * history[j];
            }
            nn.b1[i] -= LEARNING_RATE * dz;
        }

        // Update global history
        for (int i = 0; i < HISTORY_LENGTH - 1; ++i)
            history[i] = history[i + 1];
        history[HISTORY_LENGTH - 1] = (resolved_dir) ? 1.0 : -1.0;

        // Clear the speculative maps
        spec_a1.erase({seq_no,piece});
        spec_z1.erase({seq_no,piece});
        spec_prob.erase({seq_no,piece});
    }
};


#endif

static MultiTinyNNBranchPredictor tinynn_predictor;
