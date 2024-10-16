# Simple Recommendation System

Welcome to the **Simple Recommendation System**! This project is designed to build and evaluate various recommendation algorithms using both simple and advanced approaches. Whether you're a beginner looking to understand the fundamentals or an experienced developer aiming to implement a structured and modular recommendation system, this repository offers comprehensive resources to meet your needs.

## Table of Contents

- [Introduction](#introduction)
  - [Project Structure](#project-structure)
- [Approaches](#approaches)
  - [Simple Approach](#simple-approach)
  - [Advanced Approach](#advanced-approach)
  - [Differences Between Approaches](#differences-between-approaches)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Advanced Approach](#advanced-approach-1)
- [Unit Tests](#unit-tests)
- [Contact](#contact)

## Introduction

The **Recommendation System Project** aims to develop and evaluate different recommendation algorithms using a dataset of titles and user interactions. The project is divided into two main approaches:

1. **Simple Approach**: Utilizes Jupyter Notebooks for Exploratory Data Analysis (EDA), Modeling, and Evaluation.
2. **Advanced Approach**: Employs structured Python scripts for a more organized and scalable workflow, incorporating multiple algorithms and a modular codebase.


## Approaches

### Simple Approach

The **Simple Approach** is ideal for quick experimentation and understanding the basics of recommendation systems. It leverages Jupyter Notebooks to perform data analysis, build models, and evaluate their performance interactively.

- **Notebooks Included:**
  - `EDA.ipynb`: Conducts Exploratory Data Analysis to understand the dataset, visualize trends, and preprocess data.
  - `Modeling.ipynb`: Implements various recommendation algorithms such as Content-Based Filtering, Collaborative Filtering (SVD), and Hybrid Models.

#### Features

- **Interactive Environment**: Ideal for experimenting with data and models interactively.
- **Visualization**: Easily generate plots and charts to visualize data distributions and model performance.
- **Step-by-Step Implementation**: Simplifies the learning process for beginners.

### Advanced Approach

The **Advanced Approach** is tailored for production-ready environments where scalability, maintainability, and automation are crucial. It utilizes a structured and modular codebase with Python scripts organized into various components.

- **Scripts Included:**
  - `app/logger.py`: Sets up logging for the application.
  - `app/main.py`: The entry point that orchestrates the workflow.
  - `app/parser.py`: Parses command-line arguments for hyperparameters and model options.
  - `app/utils.py`: Contains utility functions for setting seeds, saving/loading models, etc.
  - `src/config.py`: Configuration file defining paths and parameters.
  - `src/data_processing/load_data.py`: Functions to load raw and processed data.
  - `src/data_processing/preprocess.py`: Data cleaning and preprocessing functions.
  - `src/feature_engineering/feature_engineer.py`: Functions for feature engineering like TF-IDF vectorization.
  - `src/models/`: Contains various recommender models:
    - `collaborative_filtering.py`
    - `content_based.py`
    - `hybrid_model.py`
    - `recommender.py`
  - `src/evaluation/metrics.py`: Implements custom evaluation metrics.
  - `src/workflow.py`: Manages the entire workflow from data loading to evaluation.
  
#### Features

- **Modular Design**: Separation of concerns enhances code readability and maintainability.
- **Multiple Algorithms**: Incorporates a variety of recommendation algorithms for comprehensive analysis.
- **Logging**: Detailed logging for monitoring and debugging.
- **Automation**: Scripts can be integrated into automated pipelines for continuous evaluation and deployment.

### Differences Between Approaches

| Feature                | Simple Approach (Notebooks)            | Advanced Approach (Scripts)               |
|------------------------|----------------------------------------|-------------------------------------------|
| **Ease of Use**        | High, suitable for beginners          | Moderate, requires familiarity with scripts |
| **Interactivity**      | Interactive and visual through notebooks | Script-based, less interactive          |
| **Scalability**        | Limited by notebook environment        | Not highly scalable but more structured  |
| **Maintainability**    | Less maintainable for large projects   | More maintainable with modular code      |
| **Automation**         | Manual execution through notebooks     | Can be automated using scripts            |
| **Flexibility**        | Easy to tweak and experiment           | Structured for robust development         |
| **Performance**        | Suitable for small to medium datasets  | Better performance with multiple algorithms |
| **Logging & Monitoring** | Basic or none                          | Comprehensive logging and monitoring      |

## Getting Started

### Prerequisites

Ensure you have the following installed on your system:

- **Python 3.7+**
- **pip** (Python package installer)
- **Git** (optional, for cloning the repository)
- **Virtual Environment Tool** (optional but recommended, e.g., `venv` or `conda`)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/recommendation_system.git
   cd recommendation_system
   ```
2. **Create a Virtual Environment**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
   
3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```
   
4. **Make sure you download the titles and title_interactions to /notebooks folder for Simple Approach, /data/raw folder for Advanced Approach **

    ```bash
    pip install -r requirements.txt
    ```

## Advanced Approach

1. **Navigate to the Project Root Directory**

   ```bash
   cd simple_recommendation_system
   ```
2. **Run the Main Script**
The main.py script orchestrates the entire workflow, including data loading, preprocessing, feature engineering, model training, evaluation, and saving results.

    ```bash
    python app/main.py --max_features 5000 --sample_percentage 5 --algorithms SVD --alpha 0.5 --top_k 10
    ```
- **Command-Line Arguments::**
  - `--max_features`: Maximum number of features for TF-IDF Vectorizer (default: 10000).
  - `--sample_percentage`: Percentage of training data to sample (default: 100.0).
  - `--algorithms`: Comma-separated list of collaborative filtering algorithms to use (default: SVD,SVDpp,NMF,KNNBasic,KNNBaseline,KNNWithMeans).
  - `--alpha`: Weighting factor for the hybrid recommender (default: 0.5).
  - `--top_k`: Number of top recommendations to evaluate (default: 10).

3. **Monitor Outputs**

  - `--Logs`: Check the logs/recommender.log file for detailed logs.
  - `--Evaluation`: The evaluation results are saved in outputs/recommendation_evaluation_results.csv.
  - `--Saved Models`: Trained models are saved in the models_saved/ directory.

## Unit Tests

Unit tests are implemented to ensure the correctness of the evaluation metrics and the functionality of the recommendation models.

1. **Navigate to the Project Root Directory**
    
   ```bash
   cd simple_recommendation_system
   ```

2. **Navigate to the Project Root Directory**

    ```bash
   python -m unittest discover -s tests
   ```
   
## Contact
For any questions or suggestions, please contact imamogluubilal@gmail.com .

