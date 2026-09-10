import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_performance():
    # 1. Define the data exactly as provided
    # We use a list of dictionaries to preserve the order of rows
    # Note: Scientific notation in strings (e.g., "2.08883e+06ms") is handled automatically by float conversion
    raw_data = [
        {
            "Label": "N=4 | Procs=16 | Size=20'000",
            "Communication": "1.20622e+06ms", 
            "Compute": "1.2291e+06ms", 
            "INIT fields with Halo": "172057ms"
        },
        {
            "Label": "N=10 | Procs=40 | Size=20000",
            "Communication": "847447ms", 
            "Compute": "844236ms", 
            "INIT fields with Halo": "69529.6ms"
        },
        {
            "Label": "N=1 | Procs=4 | Size=20000",
            "Communication": "1.05381e+06ms", 
            "Compute": "5.07197e+06ms", 
            "INIT fields with Halo": "684792ms"
        }
    ]

    # 2. Create DataFrame
    df = pd.DataFrame(raw_data)

    # 3. Clean the data (Remove 'ms' and convert to float)
    # The columns containing data are all except 'Label'
    data_cols = [col for col in df.columns if col != 'Label']

    for col in data_cols:
        # Strip 'ms', replace ' ' with empty, convert to float (handles scientific notation like 2.08e+06)
        df[col] = df[col].astype(str).str.replace('ms', '').str.strip().astype(float)

    # 4. Sort by Total Time (Descending)
    # Calculate total time for sorting purposes
    df['Total_Time'] = df[data_cols].sum(axis=1)
    
    # Sort the dataframe descending
    df = df.sort_values(by='Total_Time', ascending=False)
    
    # Drop the helper column so it doesn't get plotted
    df = df.drop(columns=['Total_Time'])

    # 5. Plotting
    # We use a Stacked Bar Chart
    # Adjust figure size to fit the labels
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define a color map that is distinct
    # Updated generic colors for the new function names
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot data
    df.plot(
        x='Label', 
        kind='bar', 
        stacked=True, 
        ax=ax, 
        width=0.6, 
        colormap='viridis', 
        edgecolor='black',
        linewidth=0.5
    )

    # 6. Formatting the Plot
    plt.title('Execution Time Breakdown by Function (Sorted by Total Time)', fontsize=16, pad=20)
    plt.xlabel('Configuration (Nodes | Procs | Size)', fontsize=12, labelpad=10)
    plt.ylabel('Time (ms)', fontsize=12)

    # Rotate X-axis labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Add Grid lines (behind the bars)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend placement
    plt.legend(title='Function', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 7. Layout adjustment to prevent clipping of labels
    plt.tight_layout()

    # Save
    plt.savefig('performance_plot.png')
    print("Plot saved as 'performance_plot.png'")

if __name__ == "__main__":
    plot_performance()