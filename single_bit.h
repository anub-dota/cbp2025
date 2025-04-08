#ifndef _SINGLE_BIT_PREDICTOR_H_
#define _SINGLE_BIT_PREDICTOR_H_

#include <vector>
#include <cstdint>
#include <iostream>

#include "hyperparams.h"

class SingleBitBranchPredictor {
    int num_entries;
    std::vector<bool> prediction_table;

public:
    SingleBitBranchPredictor(int num_entries = NUM_PREDICTORS_G) : num_entries(num_entries), prediction_table(num_entries, true) {}

    void setup() {}
    void terminate() {}

    bool predict(uint64_t PC) {
        return prediction_table[PC % num_entries];
    }

    void update(uint64_t PC, bool resolved_dir) {
        prediction_table[PC % num_entries] = resolved_dir;
    }
};

#endif

static SingleBitBranchPredictor single_bit_predictor;
