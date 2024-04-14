import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Constants
DATA_WITH_CONTEXT_FILE = 'predictions.json'
DATA_WITHOUT_CONTEXT_FILE = 'predictions_gpt.json'
THRESHOLDS = np.arange(0.0, 1.0, 0.001)

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)['data']


def calculate_accuracies(data, thresholds):
    metrics = []
    for threshold in thresholds:
        correct_predictions_jaccard = sum(1 for item in data if item["similarity_score_jaccard"] >= threshold)
        correct_predictions_cosine = sum(1 for item in data if item["similarity_score_cosine"] >= threshold)
        total_questions = len(data)
        metrics.append({
            'Threshold': threshold,
            'Jaccard Accuracy': correct_predictions_jaccard / total_questions,
            'Cosine Accuracy': correct_predictions_cosine / total_questions,
            'Total Correct Jaccard': correct_predictions_jaccard,
            'Total Correct Cosine': correct_predictions_cosine
        })
    return pd.DataFrame(metrics)

def update_plot(threshold, data_with_context, data_without_context, fig, ax):
    # Clear the previous plot
    ax.clear()

    df_metrics_with_context = calculate_accuracies(data_with_context, [threshold])
    df_metrics_without_context = calculate_accuracies(data_without_context, [threshold])

    counts = [
        df_metrics_with_context['Total Correct Jaccard'].values[0],
        df_metrics_with_context['Total Correct Cosine'].values[0],
        df_metrics_without_context['Total Correct Jaccard'].values[0],
        df_metrics_without_context['Total Correct Cosine'].values[0]
    ]

    metrics = ['Jaccard w/ context', 'Cosine w/ context', 'Jaccard w/o context', 'Cosine w/o context']
    barplot = sns.barplot(x=metrics, y=counts, ax=ax)
    ax.set_title(f'Correct Predictions at Threshold {threshold}')
    ax.set_ylabel('Count of Correct Predictions')
    ax.set_xlabel('Metric')

    # Set the y-axis limits and ticks
    ax.set_ylim(0, 20000)  # Set y-axis range from 0 to 20,000
    ax.set_yticks(np.arange(0, 20001, 2000))  # Set y-axis ticks at intervals of 2,000

    bar_label_fontsize = 10
    for bar in barplot.patches:
        ax.text(bar.get_x() + bar.get_width() / 2., 
                bar.get_height(), 
                f'{int(bar.get_height())}', 
                ha='center', va='bottom', 
                fontsize=bar_label_fontsize, weight='bold')
    
    # Redraw the canvas
    fig.canvas.draw()


def main():
    data_with_context = load_data(DATA_WITH_CONTEXT_FILE)
    data_without_context = load_data(DATA_WITHOUT_CONTEXT_FILE)

    # Set up the main Tkinter window
    root = tk.Tk()
    root.title("Correct Predictions Analysis")

    # Create a Figure and a subplot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Initial plot
    update_plot(0.5, data_with_context, data_without_context, fig, ax)

    # Embed the plot on the Tkinter Window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)

    # Slider for setting the threshold
    threshold_slider = tk.Scale(root, from_=0.0, to_=1.0, resolution=0.001, orient='horizontal', label='Threshold', command=lambda val: update_plot(float(val), data_with_context,data_without_context, fig, ax))
    threshold_slider.pack(fill=tk.X)

    root.mainloop()

if __name__ == "__main__":
    plt.ion()
    main()
