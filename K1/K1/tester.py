import sys
import time
import subprocess
import argparse


def _get_index(script_path):
    parts = script_path.split("-")[2:5]
    return "-".join(parts)[:-3]

def _run_execution_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output_lines = process.stdout.readlines()
    for output_line in output_lines:
        print(output_line.decode("utf-8"))

    if len(output_lines) > 0:
        return output_lines[-1].decode("utf-8").rstrip()
    else:
        return "No output!"

def execute_script(script_path, data_path):
    print(f"*** Executing script {script_path}")

    index = _get_index(script_path)    
    start_time = time.time()
    command = f"python {script_path} {data_path}"
    result = _run_execution_command(command)

    duration = time.time() - start_time
    print(f"*** Index: {index} - Result: {str(result)} - Duration: {duration} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("script_path", help="Path to the script")
    parser.add_argument("data_path", help="Path to the dataset")

    args = parser.parse_args()

    execute_script(args.script_path, args.data_path)
