# J.P-Morgan-ML-Project
In this project, I will solve the Question 3: Discrete choice model and credit card offers. This repository contains a report in pdf format, and Python code that implements the Deep Context-Dependent Choice Model ([Zhang et al. 2025](https://openreview.net/forum?id=bXTBtUjb0c)). The Python code inherits the "ChoiceModel" class from choice_learn package.

## 1. Features

*   **Core DeepHalo Model**: Full implementation of the Deep Context-Dependent Choice Model.
*   **Real-world Data Analysis**: Demonstrates the model's application on a real-world dataset (e.g., LPMC data).
*   **Synthetic Data Analysis**: Illustrates DeepHalo's capability to capture and model high-order effects using synthetic data.
*   **Comprehensive Testing**: Includes unit and integration tests to ensure code reliability and correctness.

## 2. Installation

DeepHalo relies on the `choice_learn` package, which provides foundational tools for choice modeling. User should install `choice_learn` package first:
    ```
    pip install choice_learn
    ```

## 3. File Structure

The repository is organized as follows:

*   `DeepHalo.py`: Contains the core implementation of the Deep Context-Dependent Choice Model.
*   `tests/`: Directory to store the Python testing code `test_DeepHalo.py`, whichi provides unit and integration tests for the DeepHalo model.
*   `LPMC Data.ipynb`: A Jupyter Notebook demonstrating the application of DeepHalo on real-world data (e.g., LPMC dataset) as analyzed in the paper.
*   `Synthetic Data.ipynb`: A Jupyter Notebook showcasing DeepHalo's performance on synthetic data specifically designed to exhibit high-order contextual effects.
*   `data/`: Directory to store dataset LPMC. The synthetic data set are too large, so a Python file that generates the dataset is given in the dataset. The original data generation code can be found on the github (https://github.com/Asimov-Chuang/DeepHalo/blob/main/DeepHalo-Code/Data/Synthetic_High_Order_Data.ipynb)
*   `README.md`: This README file.

