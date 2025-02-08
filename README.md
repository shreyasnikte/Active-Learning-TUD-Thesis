# Active Learning for Dynamic Time-of-Use Tariffs

This repository contains the implementation and research work for a thesis project focused on Active Learning approaches in the context of Dynamic Time-of-Use (dToU) tariffs for electricity consumption.

## Project Overview

This project investigates the application of Active Learning techniques to optimize electricity consumption patterns through dynamic tariff policies. The implementation includes various machine learning models, particularly focusing on Random Forest and XGBoost, to predict and analyze consumer behavior in response to different tariff structures.

## Repository Structure

- `notebooks/`: Contains Jupyter notebooks with various implementations and experiments
  - `active_learning.py`: Core implementation of the Active Learning framework
  - `data_generator.py`: Utilities for generating synthetic data
  - `final_active_learning.ipynb`: Final implementation and results
  - Various other experimental notebooks for different approaches and analyses

- `results/`: Storage for experimental results and outputs
- `UKDA-7857-csv/`: Dataset directory
- `docs/`: Documentation files
- `Lit/`: Literature and reference materials
- `mod_datasets/`: Modified and processed datasets

## Key Features

- Implementation of Active Learning strategies for tariff optimization
- Consumption modeling using XGBoost and Random Forest
- Simulation framework for testing different tariff policies
- Uncertainty sampling and entropy-based sample selection
- Comprehensive data preprocessing and analysis tools

## Setup and Installation

1. Clone the repository
2. Install required dependencies (main dependencies include):
   - numpy
   - pandas
   - scikit-learn
   - xgboost
   - matplotlib
   - seaborn
   - bokeh

3. Run Jupyter Lab using the provided script:
   ```bash
   ./run_jupyterlab.sh
   ```

## Main Components

### ConsumptionModel
- Handles the training and testing of consumption prediction models
- Implements entropy-based uncertainty measurements

### Simulator
- Simulates user behavior under different tariff policies
- Implements fuzzy participation and noise addition features

### ActiveLearner
- Implements the core Active Learning methodology
- Handles sample selection and model updating
- Includes both random and targeted sampling strategies

## Usage

The main workflow is implemented in the notebooks directory, with `final_active_learning.ipynb` being the primary notebook for the complete implementation. The project can be used to:

1. Train consumption prediction models
2. Simulate user responses to different tariff policies
3. Implement and test various Active Learning strategies
4. Analyze and visualize results

## Results

Results and experimental outputs are stored in the `results/` directory. The implementation shows the effectiveness of Active Learning in optimizing tariff policies and improving prediction accuracy with fewer labeled samples.

## Contributing

This is a thesis project repository. While it's primarily for academic purposes, suggestions and improvements are welcome through issues and pull requests.

## License

This project is part of academic research at TU Delft. Please contact the repository owner for usage permissions.

---
For more detailed information about specific components or implementation details, please refer to the individual notebook files and documentation in the `docs/` directory.
