"""Main file for retrieving runs."""
from src.process_runs.run_retriever import retrieve_run
from src.utilis.helper import extract_global_json

if __name__ == "__main__":
    # load configuration from globals.json
    num_runs = extract_global_json("num_runs")
    output_folder = extract_global_json("output_folder")
    # Run the main function
    retrieve_run(num_runs, output_folder)
