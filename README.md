# ML_Fundamentals

This repository contains fundamental machine learning models and visualizations using popular classification and regression algorithms. It includes essential preprocessing steps, model training, evaluation metrics, and performance visualizations.

## 📁 Project Structure

```
ML_Fundamentals/
├── data/                  # Folder for datasets (ignored in Git except structure)
│   └── .gitkeep           # Keeps the empty data folder in Git
├── notebooks/             # Jupyter notebooks for each model
│   ├── 01_Logistic_Regression.ipynb
│   ├── 02_KNN.ipynb
│   ├── 03_Decision_Tree.ipynb
│   ├── 04_Naive_Bayes.ipynb
│   ├── 05_Random_Forest.ipynb
│   ├── 06_SVM.ipynb
│   └── 07_Linear_Regression.ipynb
├── scripts/               # Python (.py) versions of Jupyter Notebooks
│   ├── logistic_regression.py
│   ├── knn.py
│   ├── decision_tree.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   ├── svm.py
│   └── linear_regression.py
├── .gitignore             # Files and folders to be ignored by Git
└── README.md              # Project documentation (this file)
```

## 📌 Features

- Cleaned and organized code for each algorithm
- Evaluates accuracy using `accuracy_score`
- Confusion matrix visualization using `seaborn`
- Uses `train_test_split` for splitting data
- Handles missing values and duplicate entries

## 🧠 Algorithms Implemented

### Classification:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes
- Random Forest
- Support Vector Machine (SVM)

### Regression:

- Simple Linear Regression

## 📊 Visualizations

Each notebook includes:

- Confusion matrix heatmap for classification models
- Accuracy metrics and printouts
- Data inspection with `.info()`, `.describe()`, `.shape`, etc.

## 📂 Data

Datasets are stored in the `data/` folder but are ignored from version control via `.gitignore`. You can add your own datasets here (e.g., `heart.csv`, `Salary_Data.csv`).

## ⚙️ Setup

1. Clone the repository:

```bash
https://github.com/your-username/ML_Fundamentals.git
```

2. Navigate to the project folder:

```bash
cd ML_Fundamentals
```

3. (Optional) Create a virtual environment and activate it.

4. Install the required libraries:

```bash
pip install -r requirements.txt
```

## ▶️ Running Python Scripts

The `scripts/` folder contains `.py` versions of all Jupyter notebooks for easy execution from the terminal or any IDE like VS Code. These scripts are well-structured, modular, and mirror the functionality of the notebooks.

To run a Python script:

```bash
python scripts/logistic_regression.py
```

Replace `logistic_regression.py` with the script you'd like to run.

### 🔧 IDE Usage (e.g., VS Code)

- Open the folder in VS Code.
- Navigate to the `scripts/` directory.
- Open and run any `.py` file directly using the built-in terminal or run button.

This approach showcases familiarity with `.py` based ML workflows, ideal for professional and production-grade development.

## 📝 Note

- Make sure to place your datasets inside the `data/` folder.
- If any notebook or script throws a `FileNotFoundError`, check your working directory or update the path accordingly.

---

Built with ❤️ for ML practice and learning.

---

Feel free to contribute or suggest improvements!

📌 Connect with me on [LinkedIn](https://www.linkedin.com/in/shantanumohod)
