"""
================================================================================
INTERACTIVE RESEARCH PAPER FIGURE & MATRIX VIEWER
================================================================================
This script allows you to view all pictorial graphs, curves, and confusion 
matrices generated for the research paper on screen.

Usage:
    python src\show_paper_figures.py           # Opens the interactive gallery
    python src\show_paper_figures.py --dashboard # Shows the 4-in-1 HOD presentation dashboard
    python src\show_paper_figures.py --matrix   # Shows the Confusion Matrices (Figure 5)
    python src\show_paper_figures.py --curve    # Shows the Prediction Curves (Figures 6, 7, 8)
================================================================================
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE_DIR = r"D:\AI_Research_Paper"
GRAPHS_DIR = os.path.join(BASE_DIR, "results", "graphs")

FIGURE_LIST = [
    ("fig_presentation_dashboard.png", "Master 4-in-1 Presentation Dashboard (Matrices, Regression, ROC, Importance)"),
    ("fig5_confusion_matrices.png", "Figure 5: Normalized Confusion Matrices (Decision Tree, LR, Random Forest, SVM)"),
    ("fig8_actual_vs_predicted_grades.png", "Figure 8: Actual vs. Predicted Performance (Continuous Math Grade Regression)"),
    ("fig6_roc_curves.png", "Figure 6: Receiver Operating Characteristic (ROC) Curves & AUC"),
    ("fig7_precision_recall_curves.png", "Figure 7: Precision-Recall Curves Across Operating Thresholds"),
    ("fig10_feature_importance.png", "Figure 10: Top Predictive Features by Gini Impurity Attribution"),
    ("fig4_model_comparison_metrics.png", "Figure 4: Comparative Evaluation Metrics Across Evaluated Models"),
    ("fig1_class_distribution.png", "Figure 1: Empirical Class Distribution of Student Academic Outcomes"),
    ("fig2_correlation_heatmap.png", "Figure 2: Spearman Rank Correlation Heatmap of Student Predictors"),
    ("fig3_feature_distributions.png", "Figure 3: Predictor Distributions Stratified by Final Academic Outcome"),
    ("fig9_prediction_probabilities.png", "Figure 9: Calibrated Posterior Passing Probability Densities"),
    ("fig11_learning_curves.png", "Figure 11: Empirical Learning Curves Depicting Convergence")
]

def show_single_figure(filename, title):
    filepath = os.path.join(GRAPHS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"[Error] Figure not found: {filepath}")
        return
    
    img = mpimg.imread(filepath)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title, fontsize=12, weight='bold', pad=10)
    plt.tight_layout()
    plt.show()

def show_dashboard():
    show_single_figure("fig_presentation_dashboard.png", "Master Research Presentation Dashboard")

def show_confusion_matrices():
    show_single_figure("fig5_confusion_matrices.png", "Figure 5: Normalized Confusion Matrices (All 4 Models)")

def show_prediction_curves():
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    
    p8 = os.path.join(GRAPHS_DIR, "fig8_actual_vs_predicted_grades.png")
    if os.path.exists(p8):
        axs[0].imshow(mpimg.imread(p8))
        axs[0].axis('off')
        axs[0].set_title("Figure 8: Actual vs. Predicted Grade Curve", weight='bold')

    p6 = os.path.join(GRAPHS_DIR, "fig6_roc_curves.png")
    if os.path.exists(p6):
        axs[1].imshow(mpimg.imread(p6))
        axs[1].axis('off')
        axs[1].set_title("Figure 6: ROC Discrimination Trajectories", weight='bold')

    plt.tight_layout()
    plt.show()

def interactive_gallery():
    current_idx = [0]
    
    fig, ax = plt.subplots(figsize=(13, 8.5))
    plt.subplots_adjust(bottom=0.15)
    
    def update_plot():
        ax.clear()
        fname, desc = FIGURE_LIST[current_idx[0]]
        fpath = os.path.join(GRAPHS_DIR, fname)
        if os.path.exists(fpath):
            img = mpimg.imread(fpath)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"[{current_idx[0] + 1}/{len(FIGURE_LIST)}] {desc}", fontsize=11, weight='bold')
        else:
            ax.text(0.5, 0.5, f"File not found: {fname}", ha='center', va='center')
        fig.canvas.draw_idle()
    
    def on_key(event):
        if event.key in ['right', 'n', ' ']:
            current_idx[0] = (current_idx[0] + 1) % len(FIGURE_LIST)
            update_plot()
        elif event.key in ['left', 'p']:
            current_idx[0] = (current_idx[0] - 1) % len(FIGURE_LIST)
            update_plot()
        elif event.key in ['q', 'escape']:
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)
    update_plot()
    
    print("\n" + "=" * 70)
    print(" INTERACTIVE FIGURE VIEWER CONTROLS:")
    print("   -> Press [RIGHT ARROW] or [N] or [SPACE] for NEXT figure")
    print("   -> Press [LEFT ARROW]  or [P] for PREVIOUS figure")
    print("   -> Press [Q] or [ESC] to exit")
    print("=" * 70 + "\n")
    
    plt.show()

if __name__ == '__main__':
    if "--dashboard" in sys.argv:
        show_dashboard()
    elif "--matrix" in sys.argv:
        show_confusion_matrices()
    elif "--curve" in sys.argv:
        show_prediction_curves()
    else:
        interactive_gallery()
