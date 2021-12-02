# imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, auc, plot_confusion_matrix
import matplotlib.pyplot as plt
from pandas import DataFrame

# functions
def evaluate(model, X_tr, y_tr, X_te, y_te, grid_search=False, save_fig_path=False):
    """
    Calculate classification metrics for train and test data given a fitted model.
    Plot the confusion matrix, ROC curve, and precision-recall curve for test data.
    ROC curve plotting code inspired by https://github.com/learn-co-curriculum/dsc-roc-curves-and-auc from Flatiron School
    
    Inputs:
        model: sklearn-like model object
            The fitted model.
        X_tr: array-like
            The input samples for train set.
        y_tr: array-like
            The target values for train set.
        X_te: array-like
            The input samples for test set.
        y_te: array-like
            The target values for test set.
        grid_search: boolean, default=False
            False: The model is not a grid search object.
            True: The model is a grid search object. The best parameters and the CV results will be displayed.
        save_fig_path: boolean or str, default=False
            The path to save the visualizations
            False: don't save the visualizations
            str: save the visualizations to the path (also renames them to "Holdout Data")
        
    Outputs:
        model: sklearn-like model object
            The model chosen by the grid search based on the best CV score.
    """
    
    # predictions
    trn_preds = model.predict(X_tr)
    tst_preds = model.predict(X_te)
    tst_proba = model.predict_proba(X_te)[:,1]
    
    # roc auc calcs
    fpr, tpr, _ = roc_curve(y_te, tst_proba)
    roc_auc_text = f"AUC score: {auc(fpr, tpr):.3f}"

    # precision-recall curve calcs
    precision_c, recall_c, _ = precision_recall_curve(y_te, tst_proba)
    pr_auc_text = f"AUC score: {auc(recall_c, precision_c):.3f}"
    # predict all 1's
    dummy_preds = sum(y_te==1)/len(y_te)
    
    
    print("Training Metrics")
    # Accuracy
    print(f"Accuracy: {accuracy_score(y_tr, trn_preds):.3f}")
    # Precision
    print(f"Precision: {precision_score(y_tr, trn_preds):.3f}")
    # Recall
    print(f"Recall: {recall_score(y_tr, trn_preds):.3f}")
    # f1
    print(f"f1: {f1_score(y_tr, trn_preds):.3f}")
    print('-'*10)
    print("Testing Metrics")
    # Accuracy
    print(f"Accuracy: {accuracy_score(y_te, tst_preds):.3f}")
    # Precision
    print(f"Precision: {precision_score(y_te, tst_preds):.3f}")
    # Recall
    print(f"Recall: {recall_score(y_te, tst_preds):.3f}")
    # f1
    print(f"f1: {f1_score(y_te, tst_preds):.3f}")
    
    
    # create fig, axes
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8, 18))
    
    # confusion matrix
    plot_confusion_matrix(model, X_te, y_te, cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix for Test Data')
    ax1.set_xticklabels(['not hateful', 'hateful'])
    ax1.set_yticklabels(['not hateful', 'hateful'])
    ax1.grid(False)
    
    # roc curve
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_title('Receiver operating characteristic (ROC) Curve for Test Data')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.text(x=0.72, y=0.15, s=roc_auc_text, fontsize=14)
    ax2.legend(loc='lower right', fontsize=14)

    # precision-recall curve
    ax3.plot(recall_c, precision_c, color='darkorange', lw=2, label='Precision-Recall curve')
    ax3.plot([0, 1], [dummy_preds, dummy_preds], color='navy', lw=2, linestyle='--')
    ax3.set_title('Precision-Recall Curve for Test Data')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.text(x=0.05, y=0.15, s=pr_auc_text, fontsize=14)
    ax3.legend(loc='lower left', fontsize=14)

    # save figs
    if save_fig_path:
        ax1.set_title('Confusion Matrix for Holdout Data')
        ax2.set_title('Receiver operating characteristic (ROC) Curve for Holdout Data')
        ax3.set_title('Precision-Recall Curve for Holdout Data')
        fig.savefig(save_fig_path)
    
    # display grid search results
    if grid_search:
        print(f"\nBest Parameters\n{model.best_params_}")
        results = DataFrame(model.cv_results_)
        display(results.sort_values('rank_test_f1'))
        return model