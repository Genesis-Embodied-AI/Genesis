#!/usr/bin/env python3
"""
Comprehensive Analysis of Genesis LiDAR Benchmark Results

This script provides detailed analysis and visualization of the LiDAR benchmark data.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_benchmark_results(filepath='/tmp/lidar_complete_benchmark_final.json'):
    """Load and process benchmark results from JSON file."""
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        print(f"Loaded {len(results)} benchmark results from {filepath}")
        return results
    except FileNotFoundError:
        print(f"Error: Could not find benchmark results file at {filepath}")
        print("Please run the benchmark first to generate results.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse JSON file: {e}")
        return None


def create_dataframe(results):
    """Convert benchmark results to pandas DataFrame for analysis."""
    data = []
    
    for result in results:
        if result.get('success', False):
            config = result['config']
            performance = result['performance']
            memory = result.get('memory', {})
            hit_stats = result.get('hit_stats', {})
            
            # Calculate total rays and efficiency
            total_rays = config['n_envs'] * config['n_scan_lines'] * config.get('n_points_per_line', 32)
            efficiency = total_rays / performance['avg_read_time_ms']
            
            row = {
                'n_envs': int(config['n_envs']),
                'n_scan_lines': int(config['n_scan_lines']),
                'n_obstacles': int(config['n_obstacles']),
                'n_points_per_line': int(config.get('n_points_per_line', 32)),
                'total_rays': int(total_rays),
                'read_time_ms': float(performance['avg_read_time_ms']),
                'std_read_time_ms': float(performance.get('std_read_time_ms', 0)),
                'build_time_s': float(performance['build_time_s']),
                'efficiency_rays_per_ms': float(efficiency),
                'initial_memory_mb': float(memory.get('initial_mb', 0)),
                'final_memory_mb': float(memory.get('final_mb', 0)),
                'memory_increase_mb': float(memory.get('build_increase_mb', 0)),
                'hit_rate_percent': float(hit_stats.get('hit_rate_percent', 0)),
                'benchmark_steps': int(performance.get('benchmark_steps', 50))
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    print(f"Created DataFrame with {len(df)} successful benchmark results")
    print(f"Columns: {list(df.columns)}")
    
    return df


def analyze_benchmark_results(results):
    """Perform comprehensive analysis of benchmark results."""
    if not results:
        return None
    
    # Create DataFrame
    df = create_dataframe(results)
    if df.empty:
        print("No successful results to analyze")
        return None
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE BENCHMARK ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    print(f"\nDataset Overview:")
    print(f"Total successful configurations: {len(df)}")
    print(f"Environment counts tested: {sorted(df['n_envs'].unique())}")
    print(f"Scan line counts tested: {sorted(df['n_scan_lines'].unique())}")
    print(f"Obstacle counts tested: {sorted(df['n_obstacles'].unique())}")
    
    # Performance statistics
    print(f"\nPerformance Summary:")
    print(f"Read time range: {df['read_time_ms'].min():.2f} - {df['read_time_ms'].max():.2f} ms")
    print(f"Mean read time: {df['read_time_ms'].mean():.2f} ± {df['read_time_ms'].std():.2f} ms")
    print(f"Efficiency range: {df['efficiency_rays_per_ms'].min():.0f} - {df['efficiency_rays_per_ms'].max():.0f} rays/ms")
    print(f"Mean efficiency: {df['efficiency_rays_per_ms'].mean():.0f} ± {df['efficiency_rays_per_ms'].std():.0f} rays/ms")
    
    # Best and worst performers
    fastest = df.loc[df['read_time_ms'].idxmin()]
    slowest = df.loc[df['read_time_ms'].idxmax()]
    most_efficient = df.loc[df['efficiency_rays_per_ms'].idxmax()]
    least_efficient = df.loc[df['efficiency_rays_per_ms'].idxmin()]
    
    print(f"\nBest Performers:")
    print(f"Fastest: {fastest['read_time_ms']:.2f}ms - {int(fastest['n_envs'])} envs × {int(fastest['n_scan_lines'])} lines × {int(fastest['n_obstacles'])} obstacles")
    print(f"Most efficient: {most_efficient['efficiency_rays_per_ms']:.0f} rays/ms - {int(most_efficient['n_envs'])} envs × {int(most_efficient['n_scan_lines'])} lines × {int(most_efficient['n_obstacles'])} obstacles")
    
    print(f"\nWorst Performers:")
    print(f"Slowest: {slowest['read_time_ms']:.2f}ms - {int(slowest['n_envs'])} envs × {int(slowest['n_scan_lines'])} lines × {int(slowest['n_obstacles'])} obstacles")
    print(f"Least efficient: {least_efficient['efficiency_rays_per_ms']:.0f} rays/ms - {int(least_efficient['n_envs'])} envs × {int(least_efficient['n_scan_lines'])} lines × {int(least_efficient['n_obstacles'])} obstacles")
    
    # Scaling analysis
    print(f"\n" + "=" * 60)
    print("SCALING ANALYSIS")
    print("=" * 60)
    
    # Environment scaling (fix the format string issue)
    print(f"\nEnvironment Scaling (64 scan lines, 32 obstacles):")
    env_scaling = df[(df['n_scan_lines'] == 64) & (df['n_obstacles'] == 32)].sort_values('n_envs')
    
    if len(env_scaling) > 1:
        baseline = env_scaling.iloc[0]
        for _, row in env_scaling.iterrows():
            if row['n_envs'] == baseline['n_envs']:
                print(f"  {int(row['n_envs']):4d} envs: {row['read_time_ms']:6.2f}ms (baseline)")
            else:
                parallel_efficiency = (baseline['read_time_ms'] * row['n_envs']) / row['read_time_ms']
                print(f"  {int(row['n_envs']):4d} envs: {row['read_time_ms']:6.2f}ms (parallel efficiency: {parallel_efficiency:.2f}x)")
    
    # Scan line scaling
    print(f"\nScan Line Scaling (1 environment, 32 obstacles):")
    line_scaling = df[(df['n_envs'] == 1) & (df['n_obstacles'] == 32)].sort_values('n_scan_lines')
    
    for _, row in line_scaling.iterrows():
        time_per_line = row['read_time_ms'] / row['n_scan_lines']
        print(f"  {int(row['n_scan_lines']):4d} lines: {row['read_time_ms']:6.2f}ms ({time_per_line:.4f}ms per line)")
    
    # Obstacle scaling
    print(f"\nObstacle Scaling (1 environment, 64 scan lines):")
    obstacle_scaling = df[(df['n_envs'] == 1) & (df['n_scan_lines'] == 64)].sort_values('n_obstacles')
    
    for _, row in obstacle_scaling.iterrows():
        time_per_obstacle = row['read_time_ms'] / row['n_obstacles'] if row['n_obstacles'] > 0 else 0
        print(f"  {int(row['n_obstacles']):4d} obstacles: {row['read_time_ms']:6.2f}ms ({time_per_obstacle:.4f}ms per obstacle)")
    
    # Memory analysis
    print(f"\nMemory Usage Analysis:")
    print(f"Memory increase range: {df['memory_increase_mb'].min():.1f} - {df['memory_increase_mb'].max():.1f} MB")
    print(f"Mean memory increase: {df['memory_increase_mb'].mean():.1f} ± {df['memory_increase_mb'].std():.1f} MB")
    
    # Correlation analysis
    print(f"\nCorrelation Analysis:")
    correlations = df[['n_envs', 'n_scan_lines', 'n_obstacles', 'total_rays', 'read_time_ms', 'efficiency_rays_per_ms']].corr()
    print(f"Read time correlation with:")
    print(f"  Environments: {correlations.loc['read_time_ms', 'n_envs']:.3f}")
    print(f"  Scan lines: {correlations.loc['read_time_ms', 'n_scan_lines']:.3f}")
    print(f"  Obstacles: {correlations.loc['read_time_ms', 'n_obstacles']:.3f}")
    print(f"  Total rays: {correlations.loc['read_time_ms', 'total_rays']:.3f}")
    
    return df


def create_visualizations(df, output_dir='/home/zifanw/Genesis/examples/sensors/lidar/'):
    """Create comprehensive visualizations of benchmark results."""
    if df is None or df.empty:
        print("No data to visualize")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a comprehensive figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Read time vs total rays (log-log scale)
    ax1 = plt.subplot(3, 3, 1)
    scatter = ax1.scatter(df['total_rays'], df['read_time_ms'], 
                         c=df['n_envs'], s=60, alpha=0.7, cmap='viridis')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Total Rays per Frame')
    ax1.set_ylabel('LiDAR Read Time (ms)')
    ax1.set_title('Read Time vs Total Rays')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Environments')
    
    # 2. Efficiency vs environments
    ax2 = plt.subplot(3, 3, 2)
    for lines in df['n_scan_lines'].unique():
        subset = df[df['n_scan_lines'] == lines]
        ax2.plot(subset['n_envs'], subset['efficiency_rays_per_ms'], 
                'o-', label=f'{int(lines)} lines', alpha=0.7)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Number of Environments')
    ax2.set_ylabel('Efficiency (rays/ms)')
    ax2.set_title('Efficiency vs Environment Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Read time vs obstacles
    ax3 = plt.subplot(3, 3, 3)
    for envs in sorted(df['n_envs'].unique()):
        subset = df[df['n_envs'] == envs]
        if len(subset) > 1:
            ax3.plot(subset['n_obstacles'], subset['read_time_ms'], 
                    'o-', label=f'{int(envs)} envs', alpha=0.7)
    ax3.set_xlabel('Number of Obstacles')
    ax3.set_ylabel('LiDAR Read Time (ms)')
    ax3.set_title('Read Time vs Obstacle Count')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Memory usage vs configuration complexity
    ax4 = plt.subplot(3, 3, 4)
    complexity = df['n_envs'] * df['n_scan_lines'] * df['n_obstacles']
    ax4.scatter(complexity, df['memory_increase_mb'], 
               c=df['n_envs'], s=60, alpha=0.7, cmap='plasma')
    ax4.set_xlabel('Configuration Complexity (envs × lines × obstacles)')
    ax4.set_ylabel('Memory Increase (MB)')
    ax4.set_title('Memory Usage vs Complexity')
    ax4.grid(True, alpha=0.3)
    
    # 5. Environment scaling heatmap
    ax5 = plt.subplot(3, 3, 5)
    pivot_data = df.pivot_table(values='read_time_ms', 
                               index='n_scan_lines', 
                               columns='n_envs', 
                               aggfunc='mean')
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax5)
    ax5.set_title('Read Time Heatmap (ms)')
    ax5.set_xlabel('Number of Environments')
    ax5.set_ylabel('Scan Lines')
    
    # 6. Efficiency distribution
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(df['efficiency_rays_per_ms'], bins=20, alpha=0.7, edgecolor='black')
    ax6.axvline(df['efficiency_rays_per_ms'].mean(), color='red', 
               linestyle='--', label=f'Mean: {df["efficiency_rays_per_ms"].mean():.0f}')
    ax6.set_xlabel('Efficiency (rays/ms)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Efficiency Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Build time vs read time
    ax7 = plt.subplot(3, 3, 7)
    ax7.scatter(df['build_time_s'], df['read_time_ms'], 
               c=df['n_obstacles'], s=60, alpha=0.7, cmap='coolwarm')
    ax7.set_xlabel('Build Time (s)')
    ax7.set_ylabel('Read Time (ms)')
    ax7.set_title('Build Time vs Read Time')
    ax7.grid(True, alpha=0.3)
    
    # 8. Parallel efficiency
    ax8 = plt.subplot(3, 3, 8)
    single_env_data = df[df['n_envs'] == 1]
    if len(single_env_data) > 0:
        for _, single in single_env_data.iterrows():
            # Find matching multi-env configurations
            matching = df[(df['n_scan_lines'] == single['n_scan_lines']) & 
                         (df['n_obstacles'] == single['n_obstacles']) &
                         (df['n_envs'] > 1)]
            
            if len(matching) > 0:
                parallel_efficiency = (single['read_time_ms'] * matching['n_envs']) / matching['read_time_ms']
                ax8.plot(matching['n_envs'], parallel_efficiency, 'o-', alpha=0.7,
                        label=f'{int(single["n_scan_lines"])}L×{int(single["n_obstacles"])}O')
    
    ax8.axhline(y=1.0, color='red', linestyle='--', label='Perfect scaling')
    ax8.set_xlabel('Number of Environments')
    ax8.set_ylabel('Parallel Efficiency')
    ax8.set_title('Parallel Scaling Efficiency')
    ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax8.grid(True, alpha=0.3)
    
    # 9. Performance vs complexity scatter
    ax9 = plt.subplot(3, 3, 9)
    complexity_score = df['total_rays'] / 1000 + df['n_obstacles'] / 10
    ax9.scatter(complexity_score, df['efficiency_rays_per_ms'], 
               c=df['read_time_ms'], s=60, alpha=0.7, cmap='RdYlBu_r')
    ax9.set_xlabel('Complexity Score')
    ax9.set_ylabel('Efficiency (rays/ms)')
    ax9.set_title('Efficiency vs Complexity')
    ax9.set_yscale('log')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    output_path = Path(output_dir) / 'lidar_benchmark_comprehensive_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComprehensive analysis plot saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    # Create additional detailed plots
    create_detailed_scaling_plots(df, output_dir)


def create_detailed_scaling_plots(df, output_dir):
    """Create detailed scaling analysis plots."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Environment scaling
    ax1.set_title('Environment Scaling Analysis')
    for scan_lines in sorted(df['n_scan_lines'].unique()):
        for obstacles in sorted(df['n_obstacles'].unique()):
            subset = df[(df['n_scan_lines'] == scan_lines) & (df['n_obstacles'] == obstacles)]
            if len(subset) > 1:
                ax1.plot(subset['n_envs'], subset['read_time_ms'], 
                        'o-', label=f'{int(scan_lines)}L×{int(obstacles)}O', alpha=0.7)
    
    ax1.set_xlabel('Number of Environments')
    ax1.set_ylabel('Read Time (ms)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Scan line scaling
    ax2.set_title('Scan Line Scaling Analysis')
    for envs in sorted(df['n_envs'].unique()):
        subset = df[df['n_envs'] == envs]
        if len(subset) > 1:
            ax2.plot(subset['n_scan_lines'], subset['read_time_ms'], 
                    'o-', label=f'{int(envs)} envs', alpha=0.7)
    
    ax2.set_xlabel('Number of Scan Lines')
    ax2.set_ylabel('Read Time (ms)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Obstacle scaling
    ax3.set_title('Obstacle Count Impact')
    for envs in sorted(df['n_envs'].unique()):
        subset = df[df['n_envs'] == envs]
        if len(subset) > 1:
            ax3.plot(subset['n_obstacles'], subset['read_time_ms'], 
                    'o-', label=f'{int(envs)} envs', alpha=0.7)
    
    ax3.set_xlabel('Number of Obstacles')
    ax3.set_ylabel('Read Time (ms)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Efficiency vs total rays
    ax4.set_title('Efficiency vs Total Ray Count')
    scatter = ax4.scatter(df['total_rays'], df['efficiency_rays_per_ms'], 
                         c=df['n_envs'], s=60, alpha=0.7, cmap='viridis')
    ax4.set_xlabel('Total Rays per Frame')
    ax4.set_ylabel('Efficiency (rays/ms)')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    plt.colorbar(scatter, ax=ax4, label='Environments')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save detailed scaling plots
    output_path = Path(output_dir) / 'lidar_benchmark_scaling_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Detailed scaling analysis plot saved to: {output_path}")
    
    plt.show()


def generate_performance_report(df, output_dir='/home/zifanw/Genesis/examples/sensors/lidar/'):
    """Generate a comprehensive performance report in markdown format."""
    if df is None or df.empty:
        print("No data to generate report")
        return
    
    report_path = Path(output_dir) / 'lidar_benchmark_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Genesis LiDAR Sensor Performance Benchmark Report\n\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total configurations tested**: {len(df)}\n")
        f.write(f"- **Performance range**: {df['read_time_ms'].min():.2f} - {df['read_time_ms'].max():.2f} ms\n")
        f.write(f"- **Efficiency range**: {df['efficiency_rays_per_ms'].min():.0f} - {df['efficiency_rays_per_ms'].max():.0f} rays/ms\n")
        f.write(f"- **Best configuration**: {int(df.loc[df['read_time_ms'].idxmin(), 'n_envs'])} envs × {int(df.loc[df['read_time_ms'].idxmin(), 'n_scan_lines'])} lines × {int(df.loc[df['read_time_ms'].idxmin(), 'n_obstacles'])} obstacles\n\n")
        
        # Detailed Results
        f.write("## Detailed Results\n\n")
        f.write("### Top 10 Fastest Configurations\n\n")
        top_10_fastest = df.nsmallest(10, 'read_time_ms')[['n_envs', 'n_scan_lines', 'n_obstacles', 'read_time_ms', 'efficiency_rays_per_ms']]
        f.write(top_10_fastest.to_markdown(index=False, floatfmt='.2f'))
        f.write("\n\n")
        
        f.write("### Top 10 Most Efficient Configurations\n\n")
        top_10_efficient = df.nlargest(10, 'efficiency_rays_per_ms')[['n_envs', 'n_scan_lines', 'n_obstacles', 'read_time_ms', 'efficiency_rays_per_ms']]
        f.write(top_10_efficient.to_markdown(index=False, floatfmt='.2f'))
        f.write("\n\n")
        
        # Scaling Analysis
        f.write("## Scaling Analysis\n\n")
        
        # Environment scaling
        f.write("### Environment Scaling\n\n")
        env_scaling = df[(df['n_scan_lines'] == 64) & (df['n_obstacles'] == 32)].sort_values('n_envs')
        if len(env_scaling) > 1:
            baseline = env_scaling.iloc[0]
            f.write("| Environments | Read Time (ms) | Parallel Efficiency |\n")
            f.write("|--------------|----------------|---------------------|\n")
            for _, row in env_scaling.iterrows():
                if row['n_envs'] == baseline['n_envs']:
                    f.write(f"| {int(row['n_envs'])} | {row['read_time_ms']:.2f} | 1.00x (baseline) |\n")
                else:
                    parallel_efficiency = (baseline['read_time_ms'] * row['n_envs']) / row['read_time_ms']
                    f.write(f"| {int(row['n_envs'])} | {row['read_time_ms']:.2f} | {parallel_efficiency:.2f}x |\n")
        f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("### For Best Performance\n")
        f.write("- Use fewer environments for latency-critical applications\n")
        f.write("- Balance scan lines vs points per line based on required resolution\n")
        f.write("- Consider obstacle density impact on performance\n\n")
        
        f.write("### For Best Throughput\n")
        f.write("- Use higher environment counts for batch processing\n")
        f.write("- Monitor memory usage with large configurations\n")
        f.write("- Consider adaptive ray count based on environment complexity\n\n")
        
        # Technical Details
        f.write("## Technical Details\n\n")
        f.write("### Correlation Analysis\n\n")
        correlations = df[['n_envs', 'n_scan_lines', 'n_obstacles', 'read_time_ms']].corr()
        f.write("| Factor | Correlation with Read Time |\n")
        f.write("|--------|----------------------------|\n")
        f.write(f"| Environments | {correlations.loc['read_time_ms', 'n_envs']:.3f} |\n")
        f.write(f"| Scan Lines | {correlations.loc['read_time_ms', 'n_scan_lines']:.3f} |\n")
        f.write(f"| Obstacles | {correlations.loc['read_time_ms', 'n_obstacles']:.3f} |\n")
        f.write("\n")
    
    print(f"Performance report saved to: {report_path}")


def main():
    """Main function to run comprehensive benchmark analysis."""
    print("Genesis LiDAR Benchmark Comprehensive Analysis")
    print("=" * 60)
    
    # Load results
    results = load_benchmark_results()
    if not results:
        return
    
    # Analyze results
    df = analyze_benchmark_results(results)
    if df is None:
        return
    
    # Create visualizations
    create_visualizations(df)
    
    # Generate report
    generate_performance_report(df)
    
    print(f"\nAnalysis complete! Check the output files for detailed results.")


if __name__ == "__main__":
    main()
