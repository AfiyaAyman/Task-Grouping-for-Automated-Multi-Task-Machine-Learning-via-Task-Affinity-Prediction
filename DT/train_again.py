import sys
import os
import subprocess

def run_script(script_path, params):
    try:
        subprocess.run(['python', script_path, params[0]], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the script: {e}")

script_path = 'cluster_training_DT.py'

for dataset in ['School','Landmine', 'Parkinsons']:

    # Example usage
    run_script(script_path, [dataset])