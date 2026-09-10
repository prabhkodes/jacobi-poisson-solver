import pandas as pd
from io import StringIO
import re
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data String ---
data = """
Label                               Ranks_per_node MPI_ranks_total OMP_threads EXCHANGE_BOUNDARIES INIT SOLVE_AND_SWAP 
"N=1 | rpn=2  | mpi=2  | omp=56"        2               2               56              1834309               386677                  34524307         
"N=1 | rpn=4  | mpi=4  | omp=28"        4               4               28               958298               195283                  18511096         
"N=1 | rpn=8  | mpi=8  | omp=14"        8               8               14               843351                99449                  17027085         
"N=1 | rpn=14 | mpi=14 | omp=8"        14              14                8              1068969                57677                  17352688         
"N=1 | rpn=28 | mpi=28 | omp=4"        28              28                4               792254                31619                  16650145         
"N=1 | rpn=56 | mpi=56 | omp=2"        56              56                2               997962                17340                  16970285         


"N=2 | rpn=56 | mpi=112 | omp=2"        56             112                2               558735                 8908                   8364774          
"N=2 | rpn=2  | mpi=4  | omp=56"         2               4               56              1813151               193517                  17540511         
"N=2 | rpn=28 | mpi=56 | omp=4"        28              56                4               659602                15662                   8325181          
"N=2 | rpn=4  | mpi=8  | omp=28"         4               8               28              1042787                99132                   9263563         
"N=2 | rpn=14 | mpi=28 | omp=8"        14              28                8              1006469                29460                   8638316          
"N=2 | rpn=8  | mpi=16 | omp=14"         8              16               14               948077                50145                   8495852         


"N=4 | rpn=8  | mpi=32 | omp=14"         8              32               14               503320                25846                   4105777          
"N=4 | rpn=14 | mpi=56 | omp=8"        14              56                8               410120                15563                   4076145          
"N=4 | rpn=28 | mpi=112 | omp=4"        28             112                4               503603                 8356                   4122451         


"N=8 | rpn=2  | mpi=16 | omp=56"         2              16               56               541798                51349                   4355253          
"N=8 | rpn=56 | mpi=448 | omp=2"        56             448                2               344502                 2503                   2026740          
"N=8 | rpn=28 | mpi=224 | omp=4"        28             224                4               328780                 4449                   2014018          
"N=8 | rpn=4  | mpi=32 | omp=28"         4              32               28               563492                25513                   2140872          
"N=8 | rpn=8  | mpi=64 | omp=14"         8              64               14               303225                13558                   2042366          
"N=8 | rpn=14 | mpi=112 | omp=8"        14             112                8               314919                 7832                   2017075          

"N=10 | rpn=14 | mpi=140 | omp=8"        14             140                8               257211                 6336                   1669426         
"N=10 | rpn=8  | mpi=80 | omp=14"         8              80               14               389295                10658                   1687731         
"N=10 | rpn=4  | mpi=40 | omp=28"         4              40               28               666908                20729                   1766101         
"N=10 | rpn=28 | mpi=280 | omp=4"        28             280                4               270665                 3719                   1588536         
"N=10 | rpn=56 | mpi=560 | omp=2"        56             560                2               338686                 2076                   1582320         
"N=10 | rpn=2  | mpi=20 | omp=56"         2              20               56               606333                40161                   2322124         
"""

# --- 2. Data Parsing ---
lines = data.strip().split('\n')
header = lines[0].split()
data_lines = [line.strip() for line in lines[1:] if line.strip()]

parsed_data = []
for line in data_lines:
    # Use regex to find the quoted string (Label) and then split the rest of the numbers.
    match = re.match(r'\"(.*?)\"\s+(.*)', line)
    if match:
        label = match.group(1)
        numbers_str = match.group(2)
        # Split the remaining numbers by one or more spaces and convert to int
        numbers = [int(n) for n in numbers_str.split()]
        parsed_data.append([label] + numbers)

# Create a DataFrame
df = pd.DataFrame(parsed_data, columns=header)

# Extract 'Num of nodes, N' from the 'Label' column
df['N'] = df['Label'].str.extract(r'N=(\d+)', expand=False).astype(int)

# Convert all latency columns to numeric
latency_cols_raw = ['EXCHANGE_BOUNDARIES', 'INIT', 'SOLVE_AND_SWAP']
for col in latency_cols_raw:
    df[col] = pd.to_numeric(df[col])

# --- 3. Prepare for Stacked Plot and File Output ---
latency_components = ['EXCHANGE_BOUNDARIES', 'INIT', 'SOLVE_AND_SWAP']
time_unit = 1000  # Assuming microseconds, convert to milliseconds (ms)

# Convert components to milliseconds
for col in latency_components:
    df[f'{col}_ms'] = df[col] / time_unit


# Sort the DataFrame by 'N' to logically group the bars on the plot.
df_sorted = df.sort_values(by='N').reset_index(drop=True)

# --- 4. Write Data to CSV File ---
output_filename = 'processed_latency_data.csv'
df_sorted.to_csv(output_filename, index=False)
print(f"✅ Processed data saved to: {output_filename}")

# --- 5. Plotting (Stacked Bar Chart) ---
x_labels = df_sorted['Label']
x = np.arange(len(x_labels)) # the label locations
bottom_values = np.zeros(len(x))

plt.figure(figsize=(18, 10))
colors = plt.cm.get_cmap('viridis', len(latency_components))

for i, component in enumerate(latency_components):
    current_latency = df_sorted[f'{component}_ms'].values
    plt.bar(x, current_latency, bottom=bottom_values, label=component.replace('_', ' '), color=colors(i))
    bottom_values += current_latency

# Final plot configuration
plt.ylabel('Latency (micro-seconds)', fontsize=14)
plt.xlabel('Configuration (Nodes=N | Procs/Node=RPN | MPI_RANKS = MPI | OMP/CBLAS Threads=OMP)', fontsize=16)
plt.title('Function-Wise Latency Split for Grid (N*N) where N = 10,000 and NSteps= 1,000', fontsize=16)
plt.xticks(x, x_labels, rotation=90, fontsize=9)
plt.yticks(fontsize=12)

# Add legend and adjust layout
plt.legend(title='Latency Component', fontsize=12, title_fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save and show the plot
plot_filename = 'latency_split_stacked_bar_chart.png'
plt.savefig(plot_filename)
plt.show()
print(f"✅ Plot saved to: {plot_filename}")