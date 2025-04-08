#ifndef _TWO_BIT_PREDICTOR_H_
#define _TWO_BIT_PREDICTOR_H_

#include <vector>
#include <cstdint>
#include <iostream>

#include "hyperparams.h"

class TwoBitBranchPredictor {
    int num_entries;
    std::vector<uint8_t> prediction_table; // 2-bit saturating counters

public:
    TwoBitBranchPredictor(int num_entries = NUM_PREDICTORS_G) : num_entries(num_entries), prediction_table(num_entries, 3) {} // Initialize to strongly taken

    void setup() {}
    void terminate() {}

    bool predict(uint64_t PC) {
        return prediction_table[PC % num_entries] >= 2;
    }

    void update(uint64_t PC, bool resolved_dir) {
        uint8_t &counter = prediction_table[PC % num_entries];
        if (resolved_dir) {
            if (counter == 0) counter = 1;
            else if (counter == 1) counter = 3;
            else if (counter == 2) counter = 3;
            // If already 3 (strongly taken), no change
        } else {
            if (counter == 3) counter = 2;
            else if (counter == 2) counter = 0;
            else if (counter == 1) counter = 0;
            // If already 0 (strongly not taken), no change
        }
    }
};

#endif

static TwoBitBranchPredictor two_bit_predictor;
