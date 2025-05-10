import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

def read_json_files(directory):
    # Dictionary to store all data
    all_data = {}
    
    # Get all tasks and metrics first to create the multi-index columns
    tasks = set()
    metrics = set()
    
    # First pass to collect all tasks and metrics
    for file in os.listdir(directory):
        if file.endswith('.json'):
            with open(os.path.join(directory, file), 'r') as f:
                data = json.load(f)
                for task in data.keys():
                    tasks.add(task)
                    for metric in data[task].keys():
                        metrics.add(metric)
    
    # Second pass to collect all data
    for file in os.listdir(directory):
        if file.endswith('.json'):
            model_name = file.replace('_final_report.json', '')
            with open(os.path.join(directory, file), 'r') as f:
                data = json.load(f)
                
                # Initialize model data dictionary
                model_data = {}
                
                # Process each task and metric
                for task in tasks:
                    if task in data:
                        for metric in metrics:
                            if metric in data[task]:
                                mean = data[task][metric]['mean']
                                std = data[task][metric]['std']
                                
                                # Format differently based on metric type
                                if metric in ['accuracy', 'instruction_followed']:
                                    # Convert to percentage for these metrics
                                    mean_pct = mean * 100
                                    std_pct = std * 100
                                    model_data[(task, metric)] = f"{mean_pct:.2f}% ± {std_pct:.1f}"
                                else:
                                    # Keep as absolute values for other metrics
                                    model_data[(task, metric)] = f"{mean:.2f} ± {std:.2f}"
                            else:
                                model_data[(task, metric)] = "N/A"
                    else:
                        for metric in metrics:
                            model_data[(task, metric)] = "N/A"
                
                all_data[model_name] = model_data
    
    return all_data, sorted(list(tasks)), sorted(list(metrics))

def create_dataframe(all_data, tasks, metrics):
    # Create multi-index columns
    columns = pd.MultiIndex.from_product([tasks, metrics], names=['Task', 'Metric'])
    
    # Create DataFrame
    df = pd.DataFrame(index=all_data.keys(), columns=columns)
    
    # Fill DataFrame with data
    for model, model_data in all_data.items():
        for (task, metric), value in model_data.items():
            df.loc[model, (task, metric)] = value
    
    return df

def export_to_excel(df, output_path):
    # Create a Pandas Excel writer using openpyxl as the engine
    writer = pd.ExcelWriter(output_path, engine='openpyxl')
    
    # Write the DataFrame to an Excel sheet
    df.to_excel(writer, sheet_name='Consolidated Results')
    
    # Save the Excel file
    writer.close()
    print(f"Results exported to {output_path}")

def main():
    directory = "D:/Spring 2025/NLP/Project/Final_results" # path to the directory containing JSON files
    all_data, tasks, metrics = read_json_files(directory)
    df = create_dataframe(all_data, tasks, metrics)
    
    # Display the DataFrame
    print("DataFrame created successfully. Here's a preview:")
    print(df.head())
    
    # Export to Excel
    output_path = "D:/Spring 2025/NLP/Project/consolidated_results.xlsx" # path to save the Excel file
    export_to_excel(df, output_path)
    
    print(f"\nProcess completed successfully!")
    print(f"- Read JSON files from: {directory}")
    print(f"- Created DataFrame with {len(df)} models and {df.shape[1]} metrics")
    print(f"- Exported results to: {output_path}")

if __name__ == "__main__":
    main()
