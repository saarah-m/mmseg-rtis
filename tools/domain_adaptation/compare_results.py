#!/usr/bin/env python3
"""
Compare results from domain adaptation experiments.
"""

import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Compare domain adaptation results')
    parser.add_argument('--results-dir', 
                       default='work_dirs/domain_adaptation',
                       help='directory containing experiment results')
    parser.add_argument('--output-dir', 
                       default='work_dirs/domain_adaptation/comparison',
                       help='directory to save comparison results')
    parser.add_argument('--format', 
                       choices=['json', 'csv', 'html', 'all'],
                       default='all',
                       help='output format for comparison results')
    
    return parser.parse_args()

def load_experiment_results(results_dir):
    """Load results from all experiments"""
    experiments = {}
    
    # Load from results file if it exists
    results_file = os.path.join(results_dir, 'experiment_results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            experiments = json.load(f)
    
    # Load individual experiment results
    for exp_dir in Path(results_dir).iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith('experiment'):
            exp_name = exp_dir.name
            log_file = exp_dir / 'log.json'
            
            if log_file.exists():
                # Load training logs
                with open(log_file, 'r') as f:
                    logs = json.load(f)
                
                # Extract final metrics
                if 'val' in logs and 'mIoU' in logs['val']:
                    final_miou = logs['val']['mIoU'][-1] if logs['val']['mIoU'] else 0
                    experiments[exp_name] = {
                        'name': exp_name,
                        'final_miou': final_miou,
                        'work_dir': str(exp_dir)
                    }
    
    return experiments

def create_comparison_table(experiments):
    """Create a comparison table of results"""
    data = []
    
    for exp_name, exp_data in experiments.items():
        data.append({
            'Experiment': exp_data.get('name', exp_name),
            'mIoU': exp_data.get('final_miou', 0),
            'Work Directory': exp_data.get('work_dir', 'N/A')
        })
    
    df = pd.DataFrame(data)
    return df

def create_comparison_plots(experiments, output_dir):
    """Create comparison plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    exp_names = []
    miou_scores = []
    
    for exp_name, exp_data in experiments.items():
        if 'final_miou' in exp_data:
            exp_names.append(exp_data.get('name', exp_name))
            miou_scores.append(exp_data['final_miou'])
    
    if not exp_names:
        print("No valid experiment data found for plotting")
        return
    
    # Create mIoU comparison bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(exp_names, miou_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title('Domain Adaptation Experiments - mIoU Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Experiment', fontsize=12)
    plt.ylabel('mIoU Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, score in zip(bars, miou_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'miou_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {output_dir}")

def generate_html_report(df, experiments, output_dir):
    """Generate HTML report"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Domain Adaptation Experiments Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            .best {{ background-color: #d4edda; }}
            .worst {{ background-color: #f8d7da; }}
        </style>
    </head>
    <body>
        <h1>Domain Adaptation Experiments Comparison</h1>
        
        <h2>Results Summary</h2>
        {df.to_html(classes='table', escape=False, index=False)}
        
        <h2>Experiment Details</h2>
        <ul>
    """
    
    for exp_name, exp_data in experiments.items():
        html_content += f"<li><strong>{exp_name}:</strong> {exp_data.get('name', 'N/A')}</li>"
    
    html_content += """
        </ul>
        
        <h2>Analysis</h2>
        <p>This comparison shows the performance of different domain adaptation strategies:</p>
        <ul>
            <li><strong>CityScapes → RailSem19:</strong> Direct adaptation from urban scenes</li>
            <li><strong>Mapillary → RailSem19:</strong> Direct adaptation from diverse street scenes</li>
            <li><strong>Mapillary → CityScapes → RailSem19:</strong> Multi-stage adaptation</li>
        </ul>
        
        <p><em>Generated on: {datetime.now()}</em></p>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, 'comparison_report.html'), 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {os.path.join(output_dir, 'comparison_report.html')}")

def main():
    args = parse_args()
    
    print("Loading experiment results...")
    experiments = load_experiment_results(args.results_dir)
    
    if not experiments:
        print("No experiment results found!")
        return
    
    print(f"Found {len(experiments)} experiments")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create comparison table
    df = create_comparison_table(experiments)
    
    # Save results in different formats
    if args.format in ['json', 'all']:
        results_file = os.path.join(args.output_dir, 'comparison_results.json')
        with open(results_file, 'w') as f:
            json.dump(experiments, f, indent=2)
        print(f"JSON results saved to {results_file}")
    
    if args.format in ['csv', 'all']:
        csv_file = os.path.join(args.output_dir, 'comparison_results.csv')
        df.to_csv(csv_file, index=False)
        print(f"CSV results saved to {csv_file}")
    
    if args.format in ['html', 'all']:
        generate_html_report(df, experiments, args.output_dir)
    
    # Create comparison plots
    create_comparison_plots(experiments, args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    
    # Find best performing experiment
    if not df.empty and 'mIoU' in df.columns:
        best_exp = df.loc[df['mIoU'].idxmax()]
        print(f"\n🏆 Best performing experiment: {best_exp['Experiment']} (mIoU: {best_exp['mIoU']:.3f})")

if __name__ == '__main__':
    main()
