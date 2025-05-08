#!/usr/bin/env python3
# Modified script to create a single graph that shows the average accuracy across all tasks
# for multiple models, with temperature on the x-axis ranging from 0.0 to 1.6.
# The script accepts multiple model directories and plots their average accuracy on a single graph.

import os
import json
import glob
import argparse
import matplotlib.pyplot as plt
import re
import sys


def collect_data(model_dir, metric, base_dir='.'):
    """
    Collect data from final_report.json files in the specified model directory.
    
    Args:
        model_dir (str): Path to the model directory (e.g., "Qwen2.5-3B-Instruct")
        metric (str): Metric to extract (e.g., "accuracy", "output_tokens")
        base_dir (str): Base directory where model directories are located
    
    Returns:
        dict: Data organized by task, temperature, and the specified metric
    """
    # Use the model directory directly, without hardcoding the parent path
    base_path = os.path.join(base_dir, model_dir)
    print(f"Looking for data in: {base_path}")
    
    # Dictionary to store data for each task and temperature
    data = {
        "sorting_8": {},
        "sorting_16": {},
        "comparison": {},
        "even_count_8": {},
        "even_count_16": {},
        "find_minimum_8": {},
        "find_minimum_16": {},
        "mean_8": {},
        "mean_16": {}
    }
    
    # Find all temperature directories
    temp_pattern = os.path.join(base_path, "temp_*")
    temp_dirs = glob.glob(temp_pattern)
    print(f"Looking for temperature directories with pattern: {temp_pattern}")
    print(f"Found {len(temp_dirs)} temperature directories: {[os.path.basename(d) for d in temp_dirs]}")
    
    # If no directories found, try listing the base directory to debug
    if not temp_dirs:
        try:
            print(f"Listing contents of {base_path} to debug:")
            if os.path.exists(base_path):
                contents = os.listdir(base_path)
                print(f"  Contents: {contents}")
            else:
                print(f"  Directory {base_path} does not exist")
        except Exception as e:
            print(f"  Error listing directory: {e}")
    
    if not temp_dirs:
        print(f"ERROR: No temperature directories found in {base_path}")
        print(f"Please check if the model directory '{model_dir}' is correct")
        sys.exit(1)
    
    # Sort temperature directories by their temperature value to ensure ordered processing
    temp_dirs_with_values = []
    for temp_dir in temp_dirs:
        temp_match = re.search(r'temp_(\d+\.\d+)', temp_dir)
        if temp_match:
            temp_value = float(temp_match.group(1))
            temp_dirs_with_values.append((temp_dir, temp_value))
        else:
            print(f"Warning: Could not extract temperature from directory name: {temp_dir}")
    
    # Sort by temperature value
    temp_dirs_with_values.sort(key=lambda x: x[1])
    print(f"Sorted temperature directories: {[(os.path.basename(d), v) for d, v in temp_dirs_with_values]}")
    
    # Process directories in order
    for temp_dir, temp_value in temp_dirs_with_values:
        # Extract temperature value from directory name
        temp_match = re.search(r'temp_(\d+\.\d+)', temp_dir)
        if not temp_match:
            print(f"Warning: Could not extract temperature from directory name: {temp_dir}")
            continue
        
        temp_value = float(temp_match.group(1))
        print(f"\nProcessing temperature directory: {os.path.basename(temp_dir)} (value: {temp_value})")
        
        # Find final_report.json file
        report_files = glob.glob(os.path.join(temp_dir, "**", "final_report.json"), recursive=True)
        
        if not report_files:
            print(f"Warning: No final_report.json found in {temp_dir}")
            continue
        
        # Use the first report file found
        report_file = report_files[0]
        print(f"Found report file: {report_file}")
        
        try:
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            
            # Extract metric for each task
            for task in data.keys():
                if task in report_data and metric in report_data[task]:
                    # Store the mean value of the metric
                    data[task][temp_value] = report_data[task][metric]["mean"]
                    print(f"  - Extracted {metric} for {task}: {data[task][temp_value]}")
                else:
                    if task not in report_data:
                        print(f"  - Task '{task}' not found in report data")
                    elif metric not in report_data[task]:
                        print(f"  - Metric '{metric}' not found for task '{task}'")
        
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Error processing {report_file}: {e}")
    
    # Print summary of collected data
    print("\nData collection summary:")
    for task in data:
        if data[task]:
            print(f"  - {task}: {len(data[task])} data points at temperatures {sorted(data[task].keys())}")
        else:
            print(f"  - {task}: No data collected")
    
    return data


def plot_tasks(data, tasks, title, metric, model_name, output_file=None):
    """
    Create a plot for the specified tasks.
    
    Args:
        data (dict): Data organized by task, temperature, and metric
        tasks (list): List of tasks to include in the plot
        title (str): Plot title
        metric (str): Metric being plotted
        model_name (str): Name of the model being visualized
        output_file (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Check if we have data for any of the tasks
    has_data = False
    
    # Set colors for different tasks
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'x']
    
    for i, task in enumerate(tasks):
        # Sort temperatures to ensure correct line order
        temps = sorted(data[task].keys())
        
        if not temps:
            print(f"Warning: No data for task '{task}' in {title} plot")
            continue
            
        values = [data[task][t] for t in temps]
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.plot(temps, values, marker=marker, color=color, label=task, linewidth=2, markersize=8)
        has_data = True
    
    if not has_data:
        print(f"ERROR: No data to plot for {title}")
        if output_file:
            # Create an empty plot with a message
            plt.text(0.5, 0.5, "No data available", horizontalalignment='center', 
                     verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
    else:
        # Set appropriate axis limits
        if metric == "accuracy":
            plt.ylim(0, 1.05)  # Accuracy is typically between 0 and 1
        
        # Set x-axis limits with some padding
        all_temps = []
        for task in tasks:
            all_temps.extend(data[task].keys())
        
        if all_temps:
            min_temp = min(all_temps) if all_temps else 0
            max_temp = max(all_temps) if all_temps else 2
            plt.xlim(min_temp - 0.1, max_temp + 0.1)
            
            # Add more x-axis ticks
            plt.xticks([t for t in sorted(set(all_temps))], rotation=45)
    
    plt.xlabel('Temperature', fontsize=12)
    plt.ylabel(f'{metric.capitalize()}', fontsize=12)
    plt.title(f'{model_name} - {title} - Temperature vs {metric.capitalize()}', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_file:
        print(f"Saving plot to: {output_file}")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')


def calculate_average_accuracy(data):
    """
    Calculate the average accuracy across all tasks for each temperature.
    
    Args:
        data (dict): Data organized by task, temperature, and metric
    
    Returns:
        dict: Average accuracy for each temperature
    """
    # Dictionary to store sum and count for each temperature
    temp_sums = {}
    temp_counts = {}
    
    # Iterate through all tasks and temperatures
    for task, temp_data in data.items():
        for temp, value in temp_data.items():
            if temp not in temp_sums:
                temp_sums[temp] = 0
                temp_counts[temp] = 0
            
            temp_sums[temp] += value
            temp_counts[temp] += 1
    
    # Calculate average for each temperature
    avg_accuracy = {}
    for temp in temp_sums:
        if temp_counts[temp] > 0:
            avg_accuracy[temp] = temp_sums[temp] / temp_counts[temp]
    
    return avg_accuracy


def display_mean_calculations(avg_accuracy_by_model, metric):
    """
    Display the mean calculations for each model and temperature in a formatted table.
    
    Args:
        avg_accuracy_by_model (dict): Dictionary containing average accuracy data for each model
        metric (str): Metric being displayed (e.g., "accuracy")
    """
    print("\n" + "="*80)
    print(f"MEAN {metric.upper()} CALCULATIONS")
    print("="*80)
    
    # Get all unique temperatures across all models
    all_temps = set()
    for model_data in avg_accuracy_by_model.values():
        all_temps.update(model_data.keys())
    
    # Sort temperatures
    all_temps = sorted(all_temps)
    
    # Calculate column widths
    model_width = max(len(model) for model in avg_accuracy_by_model.keys()) + 2
    temp_width = 10  # Width for temperature columns
    
    # Print header
    header = f"{'Model':{model_width}}"
    for temp in all_temps:
        header += f"| {temp:<{temp_width-2}} "
    print(header)
    print("-" * len(header))
    
    # Print data for each model
    for model, data in avg_accuracy_by_model.items():
        row = f"{model:{model_width}}"
        for temp in all_temps:
            if temp in data:
                row += f"| {data[temp]:<{temp_width-2}.4f} "
            else:
                row += f"| {'N/A':<{temp_width-2}} "
        print(row)
    
    print("="*80)


def plot_consolidated(model_data, metric, output_file=None):
    """
    Create a consolidated plot showing average accuracy across all tasks for multiple models.
    
    Args:
        model_data (dict): Data organized by model, containing average accuracy for each temperature
        metric (str): Metric being plotted (e.g., "accuracy")
        output_file (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Set colors and markers for different models
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'x']
    
    # Check if we have data for any of the models
    has_data = False
    
    for i, (model_name, avg_data) in enumerate(model_data.items()):
        # Sort temperatures to ensure correct line order
        temps = sorted(avg_data.keys())
        
        if not temps:
            print(f"Warning: No data for model '{model_name}' in consolidated plot")
            continue
            
        values = [avg_data[t] for t in temps]
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.plot(temps, values, marker=marker, color=color, label=model_name, linewidth=2, markersize=8)
        has_data = True
    
    if not has_data:
        print("ERROR: No data to plot for consolidated view")
        if output_file:
            # Create an empty plot with a message
            plt.text(0.5, 0.5, "No data available", horizontalalignment='center', 
                     verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
    else:
        # Set y-axis limits for accuracy
        if metric == "accuracy":
            plt.ylim(0, 1.05)  # Accuracy is typically between 0 and 1
        
        # Set x-axis limits to 0.0 to 1.6 as requested
        plt.xlim(0.0, 1.6)
        
        # Add more x-axis ticks
        plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    
    plt.xlabel('Temperature', fontsize=12)
    plt.ylabel(f'Average {metric.capitalize()} Across All Tasks', fontsize=12)
    plt.title(f'Temperature vs Average {metric.capitalize()} Across All Tasks', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_file:
        print(f"Saving consolidated plot to: {output_file}")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')


def visualize_temperature_data(model_dirs, metric, base_dir='.', output_dir=None, debug=False):
    """
    Visualize temperature experiment results for multiple models.
    
    Args:
        model_dirs (list): List of model directories (e.g., ["Qwen2.5-3B-Instruct", "Llama-2-7b"])
        metric (str): Metric to visualize (e.g., "accuracy", "output_tokens")
        base_dir (str, optional): Base directory where model directories are located. Defaults to '.'.
        output_dir (str, optional): Directory to save plots. Defaults to None, which will use "consolidated_plots".
        debug (bool, optional): Enable debug output and individual plots. Defaults to False.
    
    Returns:
        dict: Dictionary containing average accuracy data for each model
    """
    print(f"Visualizing {metric} for models: {', '.join(model_dirs)}")
    print(f"Using base directory: {base_dir}")
    
    # Set default output directory if not specified
    if output_dir is None:
        output_dir = os.path.join(base_dir, "consolidated_plots")
    
    print(f"Plots will be saved to: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store data for all models
    all_models_data = {}
    # Dictionary to store average accuracy for all models
    avg_accuracy_by_model = {}
    
    # Collect data for each model
    for model_dir in model_dirs:
        print(f"\nProcessing model: {model_dir}")
        
        # Collect data for this model
        model_data = collect_data(model_dir, metric, base_dir)
        
        # Check if we have any data for this model
        has_model_data = False
        for task_data in model_data.values():
            if task_data:
                has_model_data = True
                break
        
        if not has_model_data:
            print(f"WARNING: No data was collected for model {model_dir}. Skipping this model.")
            continue
        
        # Store the data for this model
        all_models_data[model_dir] = model_data
        
        # Calculate average accuracy across all tasks for this model
        avg_accuracy = calculate_average_accuracy(model_data)
        avg_accuracy_by_model[model_dir] = avg_accuracy
        
        # Create individual plots for this model if desired
        if debug:
            model_output_dir = os.path.join(output_dir, model_dir)
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Define plot configurations
            plot_configs = [
                {
                    'tasks': ['sorting_8', 'sorting_16'],
                    'title': 'Sorting Tasks',
                    'filename': f'{model_dir}_sorting_plot_{metric}.png'
                },
                {
                    'tasks': ['comparison'],
                    'title': 'Comparison Task',
                    'filename': f'{model_dir}_comparison_plot_{metric}.png'
                },
                {
                    'tasks': ['even_count_8', 'even_count_16'],
                    'title': 'Even Count Tasks',
                    'filename': f'{model_dir}_even_count_plot_{metric}.png'
                },
                {
                    'tasks': ['find_minimum_8', 'find_minimum_16'],
                    'title': 'Find Minimum Tasks',
                    'filename': f'{model_dir}_find_minimum_plot_{metric}.png'
                },
                {
                    'tasks': ['mean_8', 'mean_16'],
                    'title': 'Mean Tasks',
                    'filename': f'{model_dir}_mean_plot_{metric}.png'
                }
            ]
            
            # Create plots
            print(f"\nGenerating individual plots for {model_dir}...")
            for config in plot_configs:
                print(f"  Creating plot: {config['title']}")
                output_file = os.path.join(model_output_dir, config['filename'])
                plot_tasks(model_data, config['tasks'], config['title'], metric, model_dir, output_file)
    
    # Check if we have data for any models
    if not avg_accuracy_by_model:
        print("ERROR: No data was collected for any of the specified models.")
        sys.exit(1)
    
    # Display mean calculations
    display_mean_calculations(avg_accuracy_by_model, metric)
    
    # Create consolidated plot
    print("\nGenerating consolidated plot for all models...")
    consolidated_output_file = os.path.join(output_dir, f"consolidated_plot_{metric}.png")
    plot_consolidated(avg_accuracy_by_model, metric, consolidated_output_file)
    
    print("\nAll plots have been generated and saved.")
    print("Close the plot window to exit the program.")
    
    # Show all plots
    plt.show()
    
    return avg_accuracy_by_model


def main():
    """
    Main function with hardcoded input values.
    """
    # Hardcoded input values
    model_dirs = ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct"]  # Example model directories
    metric = "accuracy"  # Metric to visualize
    base_dir = "."  # Base directory where model directories are located
    output_dir = None  # Use default (will be set to "./consolidated_plots")
    debug = False  # Enable debug output and individual plots
    
    # Call the visualization function with hardcoded values
    visualize_temperature_data(model_dirs, metric, base_dir, output_dir, debug)


if __name__ == "__main__":
    main()
