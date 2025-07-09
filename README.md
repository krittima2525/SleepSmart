# Disorder Detection Classification Project

This project aims to build a classification model for detecting various disorders based on health and lifestyle data. The model will utilize machine learning techniques to analyze the dataset and provide insights into disorder detection.

## Project Structure

- `data/`: Contains the dataset used for training and evaluation.
  - `README.md`: Documentation about the dataset, including its source, structure, and preprocessing steps.
  - `Sleep_health_and_lifestyle_dataset.csv` from Kaggle dataset (https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
  
- `notebooks/`: Contains Jupyter notebooks for data analysis and visualization.
  - `exploratory_analysis.ipynb`: Notebook for exploratory data analysis (EDA) to understand the dataset and identify patterns.

- `src/`: Contains source code for data processing, model training, and evaluation.
  - `data_preprocessing.py`: Functions for loading and preprocessing the dataset.
  - `model_training.py`: Code for building and training the classification model.
  - `model_evaluation.py`: Functions for evaluating the model's performance.
  - `utils.py`: Utility functions used across the project.

- `requirements.txt`: Lists the dependencies required for the project.

- `.gitignore`: Specifies files and directories to be ignored by Git.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd disorder-detection-classification
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Preprocess the data using `data_preprocessing.py`.
2. Train the model using `model_training.py`.
3. Evaluate the model's performance using `model_evaluation.py`.
4. Use the Jupyter notebook in `notebooks/` for exploratory data analysis.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.