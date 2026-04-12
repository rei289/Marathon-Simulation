# StrideSim
StrideSim is a project that simulates runner's running patterns and provides insights into their performance. It takes in past user running data and simulates how the runner would perform under different conditions, such as varying pacing strategies, terrain, weather conditions, and more. The goal of StrideSim is to help runners understand their performance better and make informed decisions about their training and racing strategies.

## Setup
To set up the environment, run the following command in your terminal:
```bash
python -m venv venv
```
This will create a virtual environment named `venv`. To activate the virtual environment, use the
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
