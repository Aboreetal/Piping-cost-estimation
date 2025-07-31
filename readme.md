# Pipe Welding Cost Prediction

This project aims to predict the cost of pipe welding based on several key factors, including pipe size, pipe material, welding type, pipe schedule, and welding location. The goal is to help engineers, estimators, and project managers quickly and accurately estimate welding costs for piping projects.

## Features

- **Predicts welding cost** using machine learning (Random Forest)
- Considers multiple input factors:
  - **Pipe Material:** Carbon steel, stainless steel, or alloy steel
  - **Pipe Schedule:** 10, 10S, 20, 30, 40, 40S, STD, 80, 80S, XS, XX, 160
  - **Welding Type:** Socket welding, butt welding, or Olet welding
  - **Welding Location:** Workshop or field
  - **Pipe Size**
- Easily extendable to add new features that might affect cost
- Encodes all categorical variables to numeric for model training

## Data

- The project currently uses a dataset containing various records of pipe welding jobs.
- Pipe material, schedule, welding type, and location are categorical features encoded as numbers.
- More data is continually being gathered to improve model accuracy.

## Model

- **Algorithm:** Random Forest Regressor
- **Preprocessing:** All categorical features are encoded numerically
- **Purpose:** To provide accurate cost predictions and explore feature importance for further improvements

## How to Use

1. Clone this repository:
   ```sh
   git clone https://github.com/your-username/your-repo-name.git
   ```
2. Install dependencies (example for Python):
   ```sh
   pip install -r requirements.txt
   ```
3. Prepare your data and adjust the code as needed.
4. Run the training script or notebook to train the model and make predictions.

## Future Work

- Collect more data to improve prediction accuracy
- Explore additional features that may influence welding costs
- Experiment with other machine learning algorithms
- Build a user-friendly interface for estimators

## License

This project is open source. If you use or adapt it, please give appropriate credit. (Specify your license here, e.g., MIT, if you decide on one.)

## Acknowledgments

- Inspired by challenges faced in construction and piping cost estimation
- Special thanks to all data contributors

---

*Feel free to contribute, suggest improvements, or open issues!*