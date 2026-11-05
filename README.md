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
- **Results:** 94.65% accuracy

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Open `milestone1.ipynb` and run all cells from top to bottom.

Data preprocessing scripts are included in the repository.
