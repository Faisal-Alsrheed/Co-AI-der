# Purpose: This script is used to run tests on a Python file and generate a results file with the test results and the contents of the Python file and the test file.
# Author: Faisal Alsrheed
# v2

# Python
import os
import subprocess
from datetime import datetime

# Get the Python file path from the user
python_file = input("Enter the full path to the Python file: ")

# Derive the test file name and output file name from the Python file name
test_file = os.path.splitext(python_file)[0] + "_test.py"

# Get current date and time
now = datetime.now()

# Format as a string
now_str = now.strftime("%Y%m%d_%H%M%S")

# Append to filename
output_file = os.path.splitext(python_file)[0] + f"_results_{now_str}.txt"

backend_dict = {
    "1": "jax",
    "2": "torch",
    "3": "tensorflow",
    "4": "mlx",
    "all": ["jax", "torch", "tensorflow", "mlx"]
}

output_choice = input("Do you want the output to include 'all' (the code, test, and results), or 'test only'? Enter 'all' or 'test only': ")

with open(output_file, 'w') as f:
    if output_choice.lower() == 'all':
        # 1st: Write Python file contents
        with open(python_file, 'r') as pf:
            f.write(f"\n\n-- Python File ({python_file}) Contents --\n")
            f.write(pf.read())

        # 2nd: Write Test file contents
        with open(test_file, 'r') as tf:
            f.write(f"\n\n-- Test File ({test_file}) Contents --\n")
            f.write(tf.read())

# 3rd: Run the tests and redirect the output to the output file
backend_input = input("Enter the backend you want to use (1 for jax, 2 for torch, 3 for tensorflow, 4 for mlx, 'all' for all): ")

if backend_input.lower() == 'all':
    for backend in backend_dict['all']:
        with open(output_file, 'a') as f:
            f.write(f"\n\n-- Running tests with backend: {backend} --\n")
        subprocess.run(f"KERAS_BACKEND={backend} pytest {test_file} >> {output_file} 2>&1", shell=True)
else:
    backend = backend_dict.get(backend_input)
    if backend is None:
        print("Input not recognized. Please enter a number from 1 to 4, or 'all'.")
    else:
        with open(output_file, 'a') as f:
            f.write(f"\n\n-- Running tests with backend: {backend} --\n")
        subprocess.run(f"KERAS_BACKEND={backend} pytest {test_file} >> {output_file} 2>&1", shell=True)

# Final: Write the help message
with open(output_file, 'a') as f:
    f.write("\n\nPlease review the failed tests and provide suggestions to improve the functions for better results.")

print(f"Test results and file contents have been generated and appended to {output_file}.")