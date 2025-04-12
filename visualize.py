import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import re

# Create visualizations directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

# Define the optimizers and corresponding file patterns
optimizers = {
    'adam': 'metrics_client_*_adam.csv',
    'Vanilla SGD': 'metrics_client_*_sgd1.csv',  # SGD without momentum
    'Momentum 0.9': 'metrics_client_*_sgd0.csv'  # SGD with momentum 0.9
}

# Set color palette for consistent colors across plots
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']
OPTIMIZER_COLORS = {opt: color for opt, color in zip(optimizers.keys(), COLORS)}

# Function to read and process data for a specific optimizer
def process_optimizer_data(optimizer, pattern):
    # Get all files matching the pattern
    files = glob.glob(pattern)
    
    all_data = []
    for file in files:
        # Extract client ID from filename using regex
        match = re.search(r'metrics_client_(\d+)_', file)
        if match:
            client_id = int(match.group(1))
            
            # Read CSV file
            df = pd.read_csv(file)            
            # Add round number (row index + 1)
            df['round'] = range(1, len(df) + 1)
            
            # Add client ID and optimizer information
            df['client_id'] = client_id
            df['optimizer'] = optimizer

            all_data.append(df)
            
                
    # Combine all data
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

print("Processing data files...")

# Collect data for all optimizers
data = []
for opt, pattern in optimizers.items():
    print(f"Processing {opt} optimizer files...")
    opt_data = process_optimizer_data(opt, pattern)
    if not opt_data.empty:
        data.append(opt_data)
    else:
        print(f"No data found for {opt} optimizer.")

# Combine all data
if data:
    combined_data = pd.concat(data, ignore_index=True)
    print(f"Successfully processed data. Total records: {len(combined_data)}")
else:
    print("No data files found! Please check the file patterns and directory.")
    exit()

# Basic statistics
print("\nBasic statistics:")
for opt in optimizers.keys():
    opt_data = combined_data[combined_data['optimizer'] == opt]
    if not opt_data.empty:
        final_round = opt_data['round'].max()
        final_accuracy = opt_data[opt_data['round'] == final_round]['accuracy'].mean()
        final_f1 = opt_data[opt_data['round'] == final_round]['f1'].mean()
        
        print(f"{opt}: Final Accuracy = {final_accuracy:.4f}, Final F1 = {final_f1:.4f}")

print("\nGenerating visualizations...")

# Calculate average metrics per round per optimizer
avg_metrics = combined_data.groupby(['optimizer', 'round']).agg({
    'accuracy': 'mean',
    'f1': 'mean'
}).reset_index()

# 1. Line plot: Learning curves - Average accuracy per round for each optimizer
plt.figure(figsize=(12, 6))
for opt in optimizers.keys():
    opt_data = avg_metrics[avg_metrics['optimizer'] == opt]
    if not opt_data.empty:
        plt.plot(opt_data['round'], opt_data['accuracy'], marker='o', 
                 label=opt, color=OPTIMIZER_COLORS[opt], linewidth=2)

plt.title('Learning Curves: Average Accuracy per Round by Optimizer', fontsize=14)
plt.xlabel('Round', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Optimizer', fontsize=10)
plt.tight_layout()
plt.savefig('visualizations/learning_curves_accuracy.png', dpi=300)
plt.close()
print("✓ Created learning curves (accuracy)")

# 2. Line plot: Average F1 score per round for each optimizer
plt.figure(figsize=(12, 6))
for opt in optimizers.keys():
    opt_data = avg_metrics[avg_metrics['optimizer'] == opt]
    if not opt_data.empty:
        plt.plot(opt_data['round'], opt_data['f1'], marker='o', 
                 label=opt, color=OPTIMIZER_COLORS[opt], linewidth=2)

plt.title('Learning Curves: Average F1 Score per Round by Optimizer', fontsize=14)
plt.xlabel('Round', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Optimizer', fontsize=10)
plt.tight_layout()
plt.savefig('visualizations/learning_curves_f1.png', dpi=300)
plt.close()
print("✓ Created learning curves (F1 score)")

# 3. Box plots: Distribution of final accuracy for each optimizer (using matplotlib instead of seaborn)
final_round = combined_data['round'].max()
final_round_data = combined_data[combined_data['round'] == final_round]

plt.figure(figsize=(10, 6))
data_by_optimizer = [final_round_data[final_round_data['optimizer'] == opt]['accuracy'].values 
                     for opt in optimizers.keys()]
box_colors = [OPTIMIZER_COLORS[opt] for opt in optimizers.keys()]

bp = plt.boxplot(data_by_optimizer, patch_artist=True, labels=list(optimizers.keys()))

# Set box colors
for box, color in zip(bp['boxes'], box_colors):
    box.set(facecolor=color, alpha=0.7)
    box.set(edgecolor='black', linewidth=1.5)

plt.title(f'Distribution of Accuracy Across Clients at Round {final_round}', fontsize=14)
plt.xlabel('Optimizer', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visualizations/final_accuracy_distribution.png', dpi=300)
plt.close()
print("✓ Created box plots (accuracy distribution)")

# 4. Box plots: Distribution of final F1 scores for each optimizer (using matplotlib instead of seaborn)
plt.figure(figsize=(10, 6))
data_by_optimizer = [final_round_data[final_round_data['optimizer'] == opt]['f1'].values 
                     for opt in optimizers.keys()]
box_colors = [OPTIMIZER_COLORS[opt] for opt in optimizers.keys()]

bp = plt.boxplot(data_by_optimizer, patch_artist=True, labels=list(optimizers.keys()))

# Set box colors
for box, color in zip(bp['boxes'], box_colors):
    box.set(facecolor=color, alpha=0.7)
    box.set(edgecolor='black', linewidth=1.5)

plt.title(f'Distribution of F1 Scores Across Clients at Round {final_round}', fontsize=14)
plt.xlabel('Optimizer', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visualizations/final_f1_distribution.png', dpi=300)
plt.close()
print("✓ Created box plots (F1 score distribution)")

# 5. Heatmap: Client performance (accuracy) across rounds for each optimizer
for opt in optimizers.keys():
    opt_data = combined_data[combined_data['optimizer'] == opt]
    
    if not opt_data.empty:
        # Create pivot table: clients vs rounds, values are accuracy
        pivot_data = opt_data.pivot_table(index='client_id', columns='round', values='accuracy')
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.3f', linewidths=.5)
        plt.title(f'Accuracy Heatmap: {opt.upper()} Optimizer', fontsize=14)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Client ID', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'visualizations/heatmap_accuracy_{opt}.png', dpi=300)
        plt.close()
    
    print(f"✓ Created heatmap for {opt}")

# 6. Convergence comparison: How quickly do different optimizers converge?
plt.figure(figsize=(12, 6))

for opt in optimizers.keys():
    opt_data = avg_metrics[avg_metrics['optimizer'] == opt]
    
    if not opt_data.empty:
        # Calculate relative improvement from first round
        first_round_acc = opt_data[opt_data['round'] == 1]['accuracy'].iloc[0]
        opt_data['relative_improvement'] = (opt_data['accuracy'] - first_round_acc) / (1 - first_round_acc) if first_round_acc < 1 else 0
        
        plt.plot(opt_data['round'], opt_data['relative_improvement'], marker='o', 
                 label=opt, color=OPTIMIZER_COLORS[opt], linewidth=2)

plt.title('Convergence Rate: Relative Improvement in Accuracy', fontsize=14)
plt.xlabel('Round', fontsize=12)
plt.ylabel('Relative Improvement', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Optimizer', fontsize=10)
plt.tight_layout()
plt.savefig('visualizations/convergence_comparison.png', dpi=300)
plt.close()
print("✓ Created convergence comparison")

# 7. Variation among clients: Standard deviation of accuracy for each optimizer by round
variation_data = combined_data.groupby(['optimizer', 'round'])['accuracy'].std().reset_index()

plt.figure(figsize=(12, 6))
for opt in optimizers.keys():
    opt_variation = variation_data[variation_data['optimizer'] == opt]
    if not opt_variation.empty:
        plt.plot(opt_variation['round'], opt_variation['accuracy'], marker='o', 
                 label=opt, color=OPTIMIZER_COLORS[opt], linewidth=2)

plt.title('Client Variation: Standard Deviation of Accuracy by Round', fontsize=14)
plt.xlabel('Round', fontsize=12)
plt.ylabel('Standard Deviation', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Optimizer', fontsize=10)
plt.tight_layout()
plt.savefig('visualizations/client_variation.png', dpi=300)
plt.close()
print("✓ Created client variation visualization")

# 8. Comparative view: All metrics on a single chart
plt.figure(figsize=(15, 10))

# Split the figure into subplots
gs = GridSpec(2, 2, figure=plt.gcf())

# Accuracy subplot
ax1 = plt.subplot(gs[0, 0])
for opt in optimizers.keys():
    opt_data = avg_metrics[avg_metrics['optimizer'] == opt]
    if not opt_data.empty:
        ax1.plot(opt_data['round'], opt_data['accuracy'], marker='o', 
                 label=opt, color=OPTIMIZER_COLORS[opt], linewidth=2)

ax1.set_title('Average Accuracy Over Rounds', fontsize=13)
ax1.set_xlabel('Round', fontsize=11)
ax1.set_ylabel('Accuracy', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(title='Optimizer')

# F1 Score subplot
ax2 = plt.subplot(gs[0, 1])
for opt in optimizers.keys():
    opt_data = avg_metrics[avg_metrics['optimizer'] == opt]
    if not opt_data.empty:
        ax2.plot(opt_data['round'], opt_data['f1'], marker='o', 
                 label=opt, color=OPTIMIZER_COLORS[opt], linewidth=2)

ax2.set_title('Average F1 Score Over Rounds', fontsize=13)
ax2.set_xlabel('Round', fontsize=11)
ax2.set_ylabel('F1 Score', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(title='Optimizer')

# Convergence subplot 
ax3 = plt.subplot(gs[1, 0])
for opt in optimizers.keys():
    opt_data = avg_metrics[avg_metrics['optimizer'] == opt]
    
    if not opt_data.empty:
        # Calculate relative improvement from first round
        first_round_acc = opt_data[opt_data['round'] == 1]['accuracy'].iloc[0]
        opt_data['relative_improvement'] = (opt_data['accuracy'] - first_round_acc) / (1 - first_round_acc) if first_round_acc < 1 else 0
        
        ax3.plot(opt_data['round'], opt_data['relative_improvement'], marker='o', 
                 label=opt, color=OPTIMIZER_COLORS[opt], linewidth=2)

ax3.set_title('Convergence Rate', fontsize=13)
ax3.set_xlabel('Round', fontsize=11)
ax3.set_ylabel('Relative Improvement', fontsize=11)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend(title='Optimizer')

# Variation subplot
ax4 = plt.subplot(gs[1, 1])
for opt in optimizers.keys():
    opt_variation = variation_data[variation_data['optimizer'] == opt]
    if not opt_variation.empty:
        ax4.plot(opt_variation['round'], opt_variation['accuracy'], marker='o', 
                 label=opt, color=OPTIMIZER_COLORS[opt], linewidth=2)

ax4.set_title('Client Variation (Std Dev)', fontsize=13)
ax4.set_xlabel('Round', fontsize=11)
ax4.set_ylabel('Standard Deviation', fontsize=11)
ax4.grid(True, linestyle='--', alpha=0.7)
ax4.legend(title='Optimizer')

plt.suptitle('Comprehensive Performance Analysis', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('visualizations/comprehensive_comparison.png', dpi=300)
plt.close()
print("✓ Created comprehensive comparison visualization")

# 9. Final performance comparison (bar chart)
final_performance = avg_metrics[avg_metrics['round'] == avg_metrics['round'].max()]

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(optimizers))
width = 0.35

bars1 = ax.bar(x - width/2, final_performance['accuracy'], width, label='Accuracy', 
               color=[OPTIMIZER_COLORS[opt] for opt in final_performance['optimizer']])
bars2 = ax.bar(x + width/2, final_performance['f1'], width, label='F1 Score',
               color=[OPTIMIZER_COLORS[opt] for opt in final_performance['optimizer']], alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(final_performance['optimizer'])
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0, 1.0)  # Assuming metrics are between 0 and 1
ax.set_title(f'Final Performance Comparison at Round {final_round}', fontsize=14)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4, axis='y')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig('visualizations/final_performance_comparison.png', dpi=300)
plt.close()
print("✓ Created final performance comparison")

# 10. Improvement rate per round (bar chart with error bars)
# Calculate improvement rate per round for each optimizer
improvement_data = []

for opt in optimizers.keys():
    opt_data = combined_data[combined_data['optimizer'] == opt]
    
    if not opt_data.empty:
        client_ids = opt_data['client_id'].unique()
        
        for client_id in client_ids:
            client_data = opt_data[opt_data['client_id'] == client_id].sort_values('round')
            
            if len(client_data) > 1:  # Need at least 2 rounds to calculate improvement
                for i in range(1, len(client_data)):
                    prev_acc = client_data.iloc[i-1]['accuracy']
                    curr_acc = client_data.iloc[i]['accuracy']
                    curr_round = client_data.iloc[i]['round']
                    
                    # Improvement per round
                    improvement = curr_acc - prev_acc
                    
                    improvement_data.append({
                        'optimizer': opt,
                        'client_id': client_id,
                        'round': curr_round,
                        'improvement': improvement
                    })

improvement_df = pd.DataFrame(improvement_data)

# Calculate mean and std of improvement for each optimizer and round
improvement_stats = improvement_df.groupby(['optimizer', 'round']).agg({
    'improvement': ['mean', 'std']
}).reset_index()

improvement_stats.columns = ['optimizer', 'round', 'mean_improvement', 'std_improvement']
improvement_stats['std_improvement'] = improvement_stats['std_improvement'].fillna(0)

# Plot bar chart with error bars
plt.figure(figsize=(14, 8))

for i, opt in enumerate(optimizers.keys()):
    opt_stats = improvement_stats[improvement_stats['optimizer'] == opt]
    
    if not opt_stats.empty:
        x = opt_stats['round']
        y = opt_stats['mean_improvement']
        yerr = opt_stats['std_improvement']
        
        plt.bar(x + i*0.25 - 0.25, y, width=0.25, label=opt, 
                color=OPTIMIZER_COLORS[opt], alpha=0.7)
        plt.errorbar(x + i*0.25 - 0.25, y, yerr=yerr, fmt='none', 
                     ecolor='black', capsize=5)

plt.title('Improvement Rate per Round by Optimizer', fontsize=14)
plt.xlabel('Round', fontsize=12)
plt.ylabel('Accuracy Improvement', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4, axis='y')
plt.legend(title='Optimizer')
plt.tight_layout()
plt.savefig('visualizations/improvement_rate.png', dpi=300)
plt.close()
print("✓ Created improvement rate visualization")

print("\nAll visualizations have been saved to the 'visualizations' directory.")
print("Summary of visualizations created:")
print("1. Learning curves (accuracy)")
print("2. Learning curves (F1 score)")
print("3. Final round accuracy distribution")
print("4. Final round F1 score distribution")
print("5. Accuracy heatmaps for each optimizer")
print("6. Convergence comparison")
print("7. Client variation analysis")
print("8. Comprehensive performance comparison")
print("9. Final performance comparison")
print("10. Improvement rate per round")