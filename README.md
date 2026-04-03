# StrideSim
StrideSim is a project that simulates runner's running patterns and provides insights into their performance. The project is built using Python and utilizes various libraries for data analysis and visualization. It utilizes GCP for data storage and processing, and is designed to be scalable and efficient. The project includes features such as data preprocessing, model training, and performance evaluation. It also provides a user-friendly interface for visualizing the results and gaining insights into the runner's performance.

## Setup
To set up the environment, run the following command in your terminal:
```bash
pip install -r requirements.txt
```
Then run
```bash
pip install -e .
```

## Linting
Linting is done to ensure code quality and consistency. In this project, ruff is used for linting. To run the linter, use the following command:
```bash
ruff check
```
and if you want to automatically fix any issues, you can run:
```bash
ruff check --fix .
```
