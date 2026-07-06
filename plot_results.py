import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import seaborn as sns

# Professional journal styling
# sns.set_theme(style="ticks")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'grid.linestyle': '--',
    'grid.alpha': 0.7
})

res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Ensure output directory exists
os.makedirs(fig_dir, exist_ok=True)

def plot_s1():
    df = pd.read_csv(os.path.join(res_dir, "S1_Spatial.csv"))
    plt.figure(figsize=(7, 4.5))
    
    # Plot curves
    plt.plot(df['step'], df['R_mis'], label='Proposed MW-$k$/$n$:G', linewidth=2.5, color='#d62728')
    plt.plot(df['step'], df['R_b1'], label='Baseline 1', linewidth=2, color='#1f77b4', linestyle='--')
    plt.plot(df['step'], df['R_b2'], label='Baseline 2', linewidth=2, color='#ff7f0e', linestyle='-.')
    plt.plot(df['step'], df['R_b3'], label='Baseline 3', linewidth=2, color='#2ca02c', linestyle=':')
    
    # Plot 95% CIs
    plt.fill_between(df['step'], df['R_mis'] - 1.96*df['R_mis_se'], df['R_mis'] + 1.96*df['R_mis_se'], color='#d62728', alpha=0.1)
    plt.fill_between(df['step'], df['R_b1'] - 1.96*df['R_b1_se'], df['R_b1'] + 1.96*df['R_b1_se'], color='#1f77b4', alpha=0.1)
    plt.fill_between(df['step'], df['R_b2'] - 1.96*df['R_b2_se'], df['R_b2'] + 1.96*df['R_b2_se'], color='#ff7f0e', alpha=0.1)
    plt.fill_between(df['step'], df['R_b3'] - 1.96*df['R_b3_se'], df['R_b3'] + 1.96*df['R_b3_se'], color='#2ca02c', alpha=0.1)
    
    plt.title("Scenario S1: Spatial Degradation (Baseline Comparison)")
    plt.xlabel("Mission Time Step")
    plt.ylabel("Mission Reliability ($R(t)$)")
    plt.ylim(-0.05, 1.05)
    plt.xlim(0, 120)
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig3_S1_spatial_results.pdf"), dpi=300)
    plt.savefig(os.path.join(fig_dir, "fig3_S1_spatial_results.png"), dpi=300)
    plt.close()

def plot_s2():
    df = pd.read_csv(os.path.join(res_dir, "S2_Temporal.csv"))
    plt.figure(figsize=(7, 4.5))
    
    # Plot curves
    plt.plot(df['step'], df['R_mis'], label='Proposed MW-$k$/$n$:G', linewidth=2.5, color='#d62728')
    plt.plot(df['step'], df['R_b1'], label='Baseline 1', linewidth=2, color='#1f77b4', linestyle='--')
    plt.plot(df['step'], df['R_b2'], label='Baseline 2', linewidth=2, color='#ff7f0e', linestyle='-.')
    plt.plot(df['step'], df['R_b3'], label='Baseline 3', linewidth=2, color='#2ca02c', linestyle=':')
    
    # Plot 95% CIs
    plt.fill_between(df['step'], df['R_mis'] - 1.96*df['R_mis_se'], df['R_mis'] + 1.96*df['R_mis_se'], color='#d62728', alpha=0.1)
    plt.fill_between(df['step'], df['R_b1'] - 1.96*df['R_b1_se'], df['R_b1'] + 1.96*df['R_b1_se'], color='#1f77b4', alpha=0.1)
    plt.fill_between(df['step'], df['R_b2'] - 1.96*df['R_b2_se'], df['R_b2'] + 1.96*df['R_b2_se'], color='#ff7f0e', alpha=0.1)
    plt.fill_between(df['step'], df['R_b3'] - 1.96*df['R_b3_se'], df['R_b3'] + 1.96*df['R_b3_se'], color='#2ca02c', alpha=0.1)
    
    plt.title("Scenario S2: Temporal Degradation (Baseline Comparison)")
    plt.xlabel("Mission Time Step")
    plt.ylabel("Mission Reliability ($R(t)$)")
    plt.ylim(-0.05, 1.05)
    plt.xlim(0, 120)
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig4_S2_temporal_results.pdf"), dpi=300)
    plt.savefig(os.path.join(fig_dir, "fig4_S2_temporal_results.png"), dpi=300)
    plt.close()

def plot_s3():
    s1 = pd.read_csv(os.path.join(res_dir, "S1_Spatial.csv"))
    s2 = pd.read_csv(os.path.join(res_dir, "S2_Temporal.csv"))
    s3_indep = pd.read_csv(os.path.join(res_dir, "S3_Joint_Indep.csv"))
    s3_gauss = pd.read_csv(os.path.join(res_dir, "S3_Joint_Gauss.csv"))
    s3_t = pd.read_csv(os.path.join(res_dir, "S3_Joint_T.csv"))
    s3_clayton = pd.read_csv(os.path.join(res_dir, "S3_Joint_Clayton.csv"))
    s3_gumbel = pd.read_csv(os.path.join(res_dir, "S3_Joint_Gumbel.csv"))
    
    plt.figure(figsize=(8.5, 5))
    
    # Baselines
    plt.plot(s1['step'], s1['R_mis'], label='S1 (Spatial Baseline)', linestyle='--', color='#7f7f7f', alpha=0.7)
    plt.plot(s2['step'], s2['R_mis'], label='S2 (Temporal Baseline)', linestyle='-.', color='#bcbd22', alpha=0.7)
    
    # Product of S1 and S2 (Hypothetical Independence)
    product_r = s1['R_mis'] * s2['R_mis']
    plt.plot(s1['step'], product_r, label='S1 x S2 Product (Hypothetical)', color='black', linestyle=':', linewidth=2)
    
    # Joint Scenarios
    plt.plot(s3_indep['step'], s3_indep['R_mis'], label='S3 (Joint, Independent)', color='#2ca02c', linewidth=2)
    plt.fill_between(s3_indep['step'], s3_indep['R_mis'] - 1.96*s3_indep['R_mis_se'], s3_indep['R_mis'] + 1.96*s3_indep['R_mis_se'], color='#2ca02c', alpha=0.1)
    
    plt.plot(s3_gauss['step'], s3_gauss['R_mis'], label='S3 (Joint, Gaussian Copula)', color='#1f77b4', linewidth=2)
    plt.fill_between(s3_gauss['step'], s3_gauss['R_mis'] - 1.96*s3_gauss['R_mis_se'], s3_gauss['R_mis'] + 1.96*s3_gauss['R_mis_se'], color='#1f77b4', alpha=0.1)
    
    plt.plot(s3_t['step'], s3_t['R_mis'], label='S3 (Joint, Student-t Copula)', color='#d62728', linewidth=2)
    plt.fill_between(s3_t['step'], s3_t['R_mis'] - 1.96*s3_t['R_mis_se'], s3_t['R_mis'] + 1.96*s3_t['R_mis_se'], color='#d62728', alpha=0.1)
    
    plt.plot(s3_clayton['step'], s3_clayton['R_mis'], label='S3 (Joint, Clayton Copula)', color='#9467bd', linewidth=2)
    plt.fill_between(s3_clayton['step'], s3_clayton['R_mis'] - 1.96*s3_clayton['R_mis_se'], s3_clayton['R_mis'] + 1.96*s3_clayton['R_mis_se'], color='#9467bd', alpha=0.1)
    
    plt.plot(s3_gumbel['step'], s3_gumbel['R_mis'], label='S3 (Joint, Gumbel Copula)', color='#ff7f0e', linewidth=2)
    plt.fill_between(s3_gumbel['step'], s3_gumbel['R_mis'] - 1.96*s3_gumbel['R_mis_se'], s3_gumbel['R_mis'] + 1.96*s3_gumbel['R_mis_se'], color='#ff7f0e', alpha=0.1)
    
    plt.title("Scenario S3: Joint Spatio-Temporal Copula Coupling")
    plt.xlabel("Mission Time Step")
    plt.ylabel("Mission Reliability ($R_{mis}$)")
    plt.ylim(-0.05, 1.05)
    plt.xlim(0, 120)
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig5_S3_joint_results.pdf"), dpi=300)
    plt.savefig(os.path.join(fig_dir, "fig5_S3_joint_results.png"), dpi=300)
    plt.close()

def plot_s3_baselines():
    df = pd.read_csv(os.path.join(res_dir, "S3_Joint_T.csv"))
    plt.figure(figsize=(7, 4.5))
    
    # Plot curves
    plt.plot(df['step'], df['R_mis'], label='Proposed MW-$k$/$n$:G', linewidth=2.5, color='#d62728')
    plt.plot(df['step'], df['R_b1'], label='Baseline 1', linewidth=2, color='#1f77b4', linestyle='--')
    plt.plot(df['step'], df['R_b2'], label='Baseline 2', linewidth=2, color='#ff7f0e', linestyle='-.')
    plt.plot(df['step'], df['R_b3'], label='Baseline 3', linewidth=2, color='#2ca02c', linestyle=':')
    
    # Plot 95% CIs
    plt.fill_between(df['step'], df['R_mis'] - 1.96*df['R_mis_se'], df['R_mis'] + 1.96*df['R_mis_se'], color='#d62728', alpha=0.1)
    plt.fill_between(df['step'], df['R_b1'] - 1.96*df['R_b1_se'], df['R_b1'] + 1.96*df['R_b1_se'], color='#1f77b4', alpha=0.1)
    plt.fill_between(df['step'], df['R_b2'] - 1.96*df['R_b2_se'], df['R_b2'] + 1.96*df['R_b2_se'], color='#ff7f0e', alpha=0.1)
    plt.fill_between(df['step'], df['R_b3'] - 1.96*df['R_b3_se'], df['R_b3'] + 1.96*df['R_b3_se'], color='#2ca02c', alpha=0.1)
    
    plt.title("Scenario S3: Joint Spatio-Temporal (Baseline Comparison)")
    plt.xlabel("Mission Time Step")
    plt.ylabel("Mission Reliability ($R(t)$)")
    plt.ylim(-0.05, 1.05)
    plt.xlim(0, 120)
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig5b_S3_baselines_comparison.pdf"), dpi=300)
    plt.savefig(os.path.join(fig_dir, "fig5b_S3_baselines_comparison.png"), dpi=300)
    plt.close()

def plot_mttf():
    files = ["S2_Temporal.csv", "S3_Joint_Indep.csv", "S3_Joint_Gauss.csv", "S3_Joint_T.csv", "S3_Joint_Clayton.csv", "S3_Joint_Gumbel.csv"]
    labels = ["Temporal Only\n(S2 Baseline)", "Joint (Indep)\n(S3)", "Joint (Gaussian)\n(S3)", "Joint (Student-t)\n(S3)", "Joint (Clayton)\n(S3)", "Joint (Gumbel)\n(S3)"]
    mttfs = []
    
    # Calculate MTTFs
    for f in files:
        df = pd.read_csv(os.path.join(res_dir, f))
        # MTTF is the area under survival curve: sum of R(t)
        mttf = df['R_mis'].sum()
        mttfs.append(mttf)
        
    plt.figure(figsize=(9.5, 4.5))
    bars = plt.bar(labels, mttfs, color=['#bcbd22', '#2ca02c', '#1f77b4', '#d62728', '#9467bd', '#ff7f0e'], edgecolor='black', alpha=0.85)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
        
    plt.ylabel("Mean Time to Failure (MTTF in Steps)")
    plt.ylim(0, max(mttfs) * 1.15)
    plt.title("Mean Time to Failure (MTTF) Comparison")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig6_mttf_comparison.pdf"), dpi=300)
    plt.savefig(os.path.join(fig_dir, "fig6_mttf_comparison.png"), dpi=300)
    plt.savefig(os.path.join(fig_dir, "fig6_mttf_comparison.png"), dpi=300)
    plt.close()

def plot_sensitivity():
    sens_file = os.path.join(res_dir, "sensitivity_analysis.csv")
    if not os.path.exists(sens_file):
        print(f"Sensitivity data not found at {sens_file}. Skipping plot.")
        return
        
    df = pd.read_csv(sens_file)
    
    plt.figure(figsize=(8.5, 6))
    
    # Gaussian Copula (df is NaN)
    df_gauss = df[df['Copula'] == 'Gaussian']
    plt.plot(df_gauss['rho'], df_gauss['MTTF'], marker='s', markersize=8, linewidth=2.5, 
             label='Gaussian Copula', color='#1f77b4', linestyle='--')
             
    # Student-t Copula for different df
    dfs = sorted(df[df['Copula'] == 'Student-t']['df'].unique())
    t_colors = ['#d62728', '#2ca02c', '#8c564b', '#e377c2']
    
    for i, d in enumerate(dfs):
        df_t = df[(df['Copula'] == 'Student-t') & (df['df'] == d)]
        plt.plot(df_t['rho'], df_t['MTTF'], marker='o', markersize=7, linewidth=2,
                 label=f'Student-t Copula (df={int(d)})', color=t_colors[i % len(t_colors)])
                 
    # Clayton Copula
    df_clayton = df[df['Copula'] == 'Clayton']
    plt.plot(df_clayton['rho'], df_clayton['MTTF'], marker='^', markersize=8, linewidth=2.5,
             label='Clayton Copula (Lower-Tail)', color='#9467bd', linestyle='-.')
             
    # Gumbel Copula
    df_gumbel = df[df['Copula'] == 'Gumbel']
    plt.plot(df_gumbel['rho'], df_gumbel['MTTF'], marker='v', markersize=8, linewidth=2.5,
             label='Gumbel Copula (Upper-Tail)', color='#ff7f0e', linestyle=':')
                 
    plt.title("MTTF Sensitivity to Copula Correlation ($\\rho$) and Tail Dependence ($df$)")
    plt.xlabel("Correlation Coefficient ($\\rho$)")
    plt.ylabel("Mean Time to Failure (MTTF in Steps)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', ncol=1, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig7_sensitivity_analysis.pdf"), dpi=300)
    plt.savefig(os.path.join(fig_dir, "fig7_sensitivity_analysis.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_s1()
    plot_s2()
    plot_s3()
    plot_s3_baselines()
    plot_mttf()
    plot_sensitivity()
    print("Plots generated successfully.")
