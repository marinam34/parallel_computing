#pragma once
#include "grid.hpp"

struct StepStats {
    double err_max;
    StepStats() : err_max(0.0) {}
};

struct Solver {
  double tau;    
  double a;     
  double omega;  

  Solver(double tau_, double a_, double omega_) : tau(tau_), a(a_), omega(omega_) {}

  void start_layers(Grid& g);              
  StepStats step_seq_omp(Grid& g, double t_cur); 
};
