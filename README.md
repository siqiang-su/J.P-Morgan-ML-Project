# J.P-Morgan-ML-Project
In this project, I will solve the Question 3: Discrete choice model and credit card offers. This repository contains a report in pdf format, and Python code that implements the Deep Context-Dependent Choice Model ([Zhang et al. 2025](https://openreview.net/forum?id=bXTBtUjb0c)). The Python code inherits the "ChoiceModel" class from choice_learn package.

## 2. Features

*   **Core DeepHalo Model**: Full implementation of the Deep Context-Dependent Choice Model.
*   **Real-world Data Analysis**: Demonstrates the model's application on a real-world dataset (e.g., LPMC data).
*   **Synthetic Data Analysis**: Illustrates DeepHalo's capability to capture and model high-order effects using synthetic data.
*   **Comprehensive Testing**: Includes unit and integration tests to ensure code reliability and correctness.

## 3. Installation

To get started with DeepHalo, follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/DeepHalo.git
    cd DeepHalo
    ```

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate
    ```

3.  **Install Core Dependencies**:
    DeepHalo relies on the `choice_learn` package, which provides foundational tools for choice modeling.
    ```bash
    pip install choice_learn
    ```

4.  **Install Other Project Dependencies**:
    Install any other necessary Python packages (e.g., `numpy`, `pandas`, `scikit-learn`, `tensorflow`/`pytorch` if used, `jupyter`). It's recommended to create a `requirements.txt` file in your repository. If you have one, you can install them like this:
    ```bash
    pip install -r requirements.txt
    ```
    *(If you don't have a `requirements.txt` yet, you might need to manually list common data science packages or generate one using `pip freeze > requirements.txt` after installing everything.)*

## 4. File Structure

The repository is organized as follows:

*   `DeepHalo.py`: Contains the core implementation of the Deep Context-Dependent Choice Model.
*   `test_DeepHalo.py`: Provides unit and integration tests for the DeepHalo model.
*   `LPMC Data.ipynb`: A Jupyter Notebook demonstrating the application of DeepHalo on real-world data (e.g., LPMC dataset) as analyzed in the paper.
*   `Synthetic Data.ipynb`: A Jupyter Notebook showcasing DeepHalo's performance on synthetic data specifically designed to exhibit high-order contextual effects.
*   `data/` (Optional): Directory to store any necessary datasets (e.g., LPMC data, synthetic data CSVs).
*   `models/` (Optional): Directory to save trained DeepHalo model weights or results.
*   `requirements.txt`: Lists all Python package dependencies.
*   `README.md`: This README file.

