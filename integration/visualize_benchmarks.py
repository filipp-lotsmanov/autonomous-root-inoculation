import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def load_benchmark_data(results_dir):
    """Load all benchmark result files."""
    results_path = Path(results_dir)
    
    # Find summary file
    summary_files = list(results_path.glob('benchmark_summary_*.csv'))
    if not summary_files:
        raise FileNotFoundError(f"No benchmark summary found in {results_dir}")
    
    summary_file = sorted(summary_files)[-1]  # Most recent
    print(f"Loading: {summary_file.name}")
    
    df_summary = pd.read_csv(summary_file)
    
    # Load aggregate stats if available
    stats_files = list(results_path.glob('aggregate_stats_*.json'))
    aggregate_stats = None
    if stats_files:
        with open(sorted(stats_files)[-1], 'r') as f:
            aggregate_stats = json.load(f)
    
    # Load detailed run data
    detailed_files = list(results_path.glob('run_*_detailed.csv'))
    all_detailed = []
    for df_file in detailed_files:
        df = pd.read_csv(df_file)
        run_num = int(df_file.stem.split('_')[1])
        df['Run'] = run_num
        all_detailed.append(df)
    
    df_detailed = pd.concat(all_detailed, ignore_index=True) if all_detailed else None
    
    return df_summary, df_detailed, aggregate_stats


def create_comprehensive_visualizations(df_summary, df_detailed, aggregate_stats, output_dir):
    """Generate all analysis plots."""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Error Distribution (Histogram)
    ax1 = fig.add_subplot(gs[0, 0])
    if df_detailed is not None:
        all_errors = df_detailed[df_detailed['Success']]['Final Error (mm)']
        ax1.hist(all_errors, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(all_errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {all_errors.mean():.3f}mm')
        ax1.axvline(1.0, color='orange', linestyle=':', linewidth=2, label='Requirement: 1.0mm')
        ax1.set_xlabel('Positioning Error (mm)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Overall Error Distribution (All Runs)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Success Rate per Run
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(df_summary['run_id'], df_summary['success_rate'], color='green', alpha=0.7, edgecolor='black')
    ax2.axhline(95, color='orange', linestyle='--', label='95% target')
    ax2.axhline(100, color='red', linestyle=':', label='100%')
    ax2.set_xlabel('Run Number', fontsize=11)
    ax2.set_ylabel('Success Rate (%)', fontsize=11)
    ax2.set_title('Success Rate per Run', fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Mean Error per Run
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(df_summary['run_id'], df_summary['mean_error_mm'], 'o-', color='steelblue', linewidth=2, markersize=8)
    ax3.axhline(df_summary['mean_error_mm'].mean(), color='red', linestyle='--', label=f'Overall mean: {df_summary["mean_error_mm"].mean():.3f}mm')
    ax3.axhline(1.0, color='orange', linestyle=':', label='Requirement: 1.0mm')
    ax3.fill_between(df_summary['run_id'], 
                      df_summary['mean_error_mm'] - df_summary['std_error_mm'],
                      df_summary['mean_error_mm'] + df_summary['std_error_mm'],
                      alpha=0.3, color='steelblue')
    ax3.set_xlabel('Run Number', fontsize=11)
    ax3.set_ylabel('Mean Error (mm)', fontsize=11)
    ax3.set_title('Mean Error per Run (with Std Dev)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Error Box Plot
    ax4 = fig.add_subplot(gs[1, 0])
    if df_detailed is not None:
        successful_detailed = df_detailed[df_detailed['Success']]
        run_groups = [successful_detailed[successful_detailed['Run'] == r]['Final Error (mm)'].values 
                     for r in sorted(successful_detailed['Run'].unique())]
        bp = ax4.boxplot(run_groups, patch_artist=True)
        ax4.set_xticklabels(sorted(successful_detailed['Run'].unique()))
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        ax4.axhline(1.0, color='orange', linestyle='--', label='Requirement')
        ax4.set_xlabel('Run Number', fontsize=11)
        ax4.set_ylabel('Error (mm)', fontsize=11)
        ax4.set_title('Error Distribution per Run', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Timing Analysis
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(df_summary['mean_steps_per_target'], df_summary['mean_error_mm'], 
               s=100, c=df_summary['success_rate'], cmap='RdYlGn', 
               edgecolors='black', linewidth=1, vmin=80, vmax=100)
    ax5.set_xlabel('Mean Steps per Target', fontsize=11)
    ax5.set_ylabel('Mean Error (mm)', fontsize=11)
    ax5.set_title('Steps vs Accuracy Trade-off', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax5.collections[0], ax=ax5)
    cbar.set_label('Success Rate (%)', fontsize=10)
    
    # 6. Execution Time Distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(df_summary['run_id'], df_summary['execution_time_s'], color='purple', alpha=0.7, edgecolor='black')
    ax6.axhline(df_summary['execution_time_s'].mean(), color='red', linestyle='--', 
               label=f'Mean: {df_summary["execution_time_s"].mean():.2f}s')
    ax6.set_xlabel('Run Number', fontsize=11)
    ax6.set_ylabel('Execution Time (s)', fontsize=11)
    ax6.set_title('Execution Time per Run', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Per-Target Error Analysis
    ax7 = fig.add_subplot(gs[2, :])
    if df_detailed is not None:
        successful = df_detailed[df_detailed['Success']]
        for run in sorted(successful['Run'].unique()):
            run_data = successful[successful['Run'] == run]
            ax7.plot(range(len(run_data)), run_data['Final Error (mm)'].values, 
                    'o-', alpha=0.5, label=f'Run {run}')
        
        ax7.axhline(1.0, color='orange', linestyle='--', linewidth=2, label='Requirement')
        ax7.set_xlabel('Target Number (within run)', fontsize=11)
        ax7.set_ylabel('Error (mm)', fontsize=11)
        ax7.set_title('Per-Target Error Across All Runs', fontweight='bold')
        ax7.legend(ncol=5, fontsize=8)
        ax7.grid(True, alpha=0.3)
    
    plt.suptitle('Task 13: System Performance Benchmark Analysis', fontsize=16, fontweight='bold')
    
    # Save
    viz_path = Path(output_dir) / 'benchmark_analysis.png'
    plt.savefig(viz_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved comprehensive analysis: {viz_path}")
    plt.close()


def create_summary_dashboard(df_summary, aggregate_stats, output_dir):
    """Create executive summary dashboard."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Key Metrics Display
    ax1.axis('off')
    metrics_text = f"""
PERFORMANCE SUMMARY
{'='*40}

Success Rate:    {aggregate_stats['mean_success_rate']:.1f}%
Mean Error:      {aggregate_stats['mean_positioning_error_mm']:.3f} mm
Std Dev Error:   {aggregate_stats['std_positioning_error_mm']:.3f} mm
Best Run:        {aggregate_stats['best_error_mm']:.3f} mm
Worst Run:       {aggregate_stats['worst_error_mm']:.3f} mm

Execution Time:  {aggregate_stats['mean_execution_time_s']:.2f} s/run
Time per Target: {aggregate_stats['mean_time_per_target_s']:.2f} s
Steps per Target: {aggregate_stats['mean_steps_per_target']:.0f}

Requirement Compliance:
  <1mm accuracy:  {aggregate_stats['requirement_compliance_rate']:.1f}%
  Status:         {'PASS ✓' if aggregate_stats['mean_positioning_error_mm'] < 1.0 else 'NEEDS IMPROVEMENT'}
"""
    
    ax1.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax1.set_title('Key Performance Metrics', fontsize=14, fontweight='bold')
    
    # 2. Success Rate Gauge
    success_rate = aggregate_stats['mean_success_rate']
    ax2.barh([0], [success_rate], color='green' if success_rate >= 95 else 'orange', alpha=0.7)
    ax2.barh([0], [100-success_rate], left=[success_rate], color='lightgray', alpha=0.5)
    ax2.set_xlim([0, 100])
    ax2.set_yticks([])
    ax2.set_xlabel('Success Rate (%)', fontsize=11)
    ax2.set_title(f'Overall Success Rate: {success_rate:.1f}%', fontsize=14, fontweight='bold')
    ax2.axvline(95, color='red', linestyle='--', linewidth=2, label='95% target')
    ax2.legend()
    
    # 3. Error Statistics
    errors = [aggregate_stats['best_error_mm'], 
             aggregate_stats['mean_positioning_error_mm'],
             aggregate_stats['worst_error_mm']]
    labels = ['Best', 'Mean', 'Worst']
    colors = ['green', 'blue', 'red']
    
    bars = ax3.bar(labels, errors, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.axhline(1.0, color='orange', linestyle='--', linewidth=2, label='1mm requirement')
    ax3.set_ylabel('Error (mm)', fontsize=11)
    ax3.set_title('Error Statistics', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, errors):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}mm', ha='center', va='bottom', fontweight='bold')
    
    # 4. Compliance Matrix
    ax4.axis('off')
    
    requirement_status = {
        'Accuracy (<1mm mean)': '✓ PASS' if aggregate_stats['mean_positioning_error_mm'] < 1.0 else '✗ FAIL',
        'Success Rate (>95%)': '✓ PASS' if aggregate_stats['mean_success_rate'] > 95 else '✗ FAIL',
        'Consistency (<0.2mm std)': '✓ PASS' if aggregate_stats['std_positioning_error_mm'] < 0.2 else '✗ FAIL',
        'Total Runs': f"{aggregate_stats['total_runs']} executed"
    }
    
    compliance_text = "\nREQUIREMENT COMPLIANCE\n" + "="*40 + "\n\n"
    for req, status in requirement_status.items():
        compliance_text += f"{req:.<30} {status}\n"
    
    ax4.text(0.1, 0.5, compliance_text, fontsize=13, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax4.set_title('Requirement Compliance', fontsize=14, fontweight='bold')
    
    plt.suptitle('Task 13: Benchmark Summary Dashboard', fontsize=16, fontweight='bold')
    
    # Save
    output_path = Path(output_dir) / 'benchmark_dashboard.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved dashboard: {output_path}")
    plt.close()


def create_detailed_analysis(df_summary, df_detailed, output_dir):
    """Create detailed multi-panel analysis."""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Error vs Run Number (Trend)
    ax = axes[0, 0]
    ax.errorbar(df_summary['run_id'], df_summary['mean_error_mm'], 
               yerr=df_summary['std_error_mm'], fmt='o-', capsize=5, capthick=2,
               color='steelblue', ecolor='gray', linewidth=2, markersize=8)
    ax.axhline(1.0, color='orange', linestyle='--', label='Requirement')
    ax.set_xlabel('Run Number')
    ax.set_ylabel('Mean Error (mm)')
    ax.set_title('Positioning Error Trend')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Success Rate Stability
    ax = axes[0, 1]
    ax.plot(df_summary['run_id'], df_summary['success_rate'], 'o-', 
           color='green', linewidth=2, markersize=8)
    ax.axhline(100, color='red', linestyle=':', label='100%')
    ax.axhline(95, color='orange', linestyle='--', label='95% target')
    ax.fill_between(df_summary['run_id'], df_summary['success_rate'], 100, 
                     alpha=0.3, color='green')
    ax.set_xlabel('Run Number')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate Consistency')
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Execution Time Analysis
    ax = axes[0, 2]
    ax.scatter(df_summary['total_targets'], df_summary['execution_time_s'],
              s=100, c=df_summary['success_rate'], cmap='RdYlGn', 
              edgecolors='black', vmin=80, vmax=100)
    ax.set_xlabel('Number of Targets')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Execution Time vs Target Count')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Success Rate (%)')
    
    # 4. Per-Target Performance (if detailed data available)
    # Group by the within-run target position. benchmark_system.py writes this
    # as 'Target Number'; fall back to 'Plant ID' for older result files.
    ax = axes[1, 0]
    position_column = next(
        (col for col in ('Target Number', 'Plant ID')
         if df_detailed is not None and col in df_detailed.columns),
        None
    )
    if position_column is not None:
        target_errors = df_detailed.groupby(position_column)['Final Error (mm)'].agg(['mean', 'std', 'count'])
        x_pos = range(len(target_errors))
        ax.bar(x_pos, target_errors['mean'], yerr=target_errors['std'],
              capsize=5, color='skyblue', alpha=0.7, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(target_errors.index)
        ax.set_xlabel('Plant Position')
        ax.set_ylabel('Mean Error (mm)')
        ax.set_title('Error by Plant Position (Averaged Across Runs)')
        ax.axhline(1.0, color='orange', linestyle='--')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No per-target data available', ha='center', va='center')
    
    # 5. Steps vs Error Correlation
    ax = axes[1, 1]
    ax.scatter(df_summary['mean_steps_per_target'], df_summary['mean_error_mm'],
              s=150, alpha=0.6, c='purple', edgecolors='black', linewidth=1.5)
    ax.set_xlabel('Mean Steps per Target')
    ax.set_ylabel('Mean Error (mm)')
    ax.set_title('Convergence Speed vs Accuracy')
    ax.grid(True, alpha=0.3)
    
    # Calculate correlation
    if len(df_summary) > 2:
        corr = np.corrcoef(df_summary['mean_steps_per_target'], df_summary['mean_error_mm'])[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 6. Cumulative Success Distribution
    ax = axes[1, 2]
    if df_detailed is not None:
        successful = df_detailed[df_detailed['Success']]
        sorted_errors = np.sort(successful['Final Error (mm)'].values)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        ax.plot(sorted_errors, cumulative, linewidth=3, color='steelblue')
        ax.axvline(1.0, color='orange', linestyle='--', linewidth=2, label='1mm requirement')
        
        # Mark percentiles
        pct_under_1mm = (sorted_errors < 1.0).sum() / len(sorted_errors) * 100
        ax.text(1.0, pct_under_1mm, f'{pct_under_1mm:.1f}%', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        ax.set_xlabel('Error (mm)')
        ax.set_ylabel('Cumulative Percentage (%)')
        ax.set_title('Cumulative Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Detailed Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir) / 'detailed_analysis.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved detailed analysis: {output_path}")
    plt.close()


def main():
    """Main visualization execution."""
    if len(sys.argv) < 2:
        print("Usage: python visualize_benchmarks.py <benchmark_results_directory>")
        print("\nExample: python visualize_benchmarks.py benchmark_results")
        sys.exit(1)
    
    results_dir = sys.argv[1]

    
    # Load data
    print(f"\nLoading data from: {results_dir}")
    df_summary, df_detailed, aggregate_stats = load_benchmark_data(results_dir)
    
    print(f"  Runs loaded: {len(df_summary)}")
    if df_detailed is not None:
        print(f"  Total targets: {len(df_detailed)}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    create_summary_dashboard(df_summary, aggregate_stats, results_dir)
    create_detailed_analysis(df_summary, df_detailed, results_dir)
    create_comprehensive_visualizations(df_summary, df_detailed, aggregate_stats, results_dir)
    
    print(f"All visualizations saved to: {results_dir}")
    print("\nGenerated files:")
    print("  - benchmark_dashboard.png (executive summary)")
    print("  - detailed_analysis.png (deep dive)")
    print("  - benchmark_analysis.png (comprehensive)")


if __name__ == "__main__":
    main()
