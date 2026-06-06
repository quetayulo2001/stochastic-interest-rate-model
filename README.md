# Stochastic Interest Rate Modelling and Prediction
**Finance Club, IIT Roorkee**

## Overview
This repository contains a production-grade implementation of the Cox-Ingersoll-Ross (CIR) stochastic short-rate model. The objective is to reconstruct an entire out-of-sample yield curve (6 Months to 30 Years) using *only* the daily 3-Month yield as the observable input ($r_t$), while achieving a predictive out-of-sample $R^2 > 0.85$.

## Setup & Execution
1. Ensure Python 3.9+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt