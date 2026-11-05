# F1 Overtake Window Predictor

**Team JanusForge** — Owen, Amine, Joshua, Ayobami

Deep learning project predicting whether a Formula 1 driver will overtake the car directly ahead within the next 60 seconds, using the previous 30 seconds of telemetry.

## Dataset

Data from the [OpenF1 API](https://openf1.org/docs/), covering five 2024 races: Bahrain, Australia, Spain, Austria, and São Paulo.

## Milestone 1

- **Dataset:** 17,719 observations, 16 features
- **Baseline:** Always predicts no overtake
- **Model:** Feedforward Neural Network (FNN)
- **Test set:** São Paulo 2024
- **Results:** 94.65% accuracy, 0.1938 F1, 0.1148 PR-AUC

## Running the Project

1. Open [`milestone1_janusforge.ipynb`](milestone1_janusforge.ipynb) in Google Colab.
2. Run all cells from top to bottom.

The notebook already installs the required dependencies automatically, and raw race data and preprocessing scripts are included in the repository.
