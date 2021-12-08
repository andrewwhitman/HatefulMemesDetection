# imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, auc, plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from string import punctuation
from nltk import word_tokenize

# functions
def evaluate(model, X_tr, y_tr, X_te, y_te, grid_search=False, save_fig_path=False):
    """
    Calculate classification metrics for train and test data given a fitted model.
    Plot the confusion matrix and ROC curve for test data.
    ROC curve plotting code derived from Flatiron School's https://github.com/learn-co-curriculum/dsc-roc-curves-and-auc
    
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
    """
    
    # predictions
    trn_preds = model.predict(X_tr)
    tst_preds = model.predict(X_te)
    tst_proba = model.predict_proba(X_te)[:,1]
    
    # roc auc calcs
    fpr, tpr, _ = roc_curve(y_te, tst_proba)
    roc_auc_text = f"AUC score: {auc(fpr, tpr):.3f}"
    
    # metrics
    print("Training Metrics")
    print(f"Accuracy: {accuracy_score(y_tr, trn_preds):.3f}")
    print(f"Precision: {precision_score(y_tr, trn_preds):.3f}")
    print(f"Recall: {recall_score(y_tr, trn_preds):.3f}")
    print(f"f1: {f1_score(y_tr, trn_preds):.3f}")
    print('-'*10)
    print("Testing Metrics")
    print(f"Accuracy: {accuracy_score(y_te, tst_preds):.3f}")
    print(f"Precision: {precision_score(y_te, tst_preds):.3f}")
    print(f"Recall: {recall_score(y_te, tst_preds):.3f}")
    print(f"f1: {f1_score(y_te, tst_preds):.3f}")
    
    # create fig, axes
    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 12))
    
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

    # save figs, rename to Holdout Data
    if save_fig_path:
        ax1.set_title('Confusion Matrix for Holdout Data')
        ax2.set_title('Receiver operating characteristic (ROC) Curve for Holdout Data')
        ax3.set_title('Precision-Recall Curve for Holdout Data')
        fig.savefig(save_fig_path)
    
    # display grid search results, sort by roc_auc
    if grid_search:
        print(f"\nBest Parameters\n{model.best_params_}")
        results = DataFrame(model.cv_results_)
        display(results.sort_values('rank_test_roc_auc'))


def generate_glove(vocab, path='data/glove.6B/glove.6B.300d.txt'):
    """
    Generate word vectors from a pre-trained word embedding, like GloVe.
    Derived from Flatiron School's https://github.com/learn-co-curriculum/dsc-classification-with-word-embeddings-codealong

    Inputs:
        vocab: collection (set, list, tuple, dict)
            The vocabulary, or unique tokens, of a corpus.
        path: str, default='data/glove.6B/glove.6B.300d.txt'
            The relative path to the downloaded GloVe vectors. Default is 300 dimensional word vectors.
    Output:
        glove: dict
            Maps a token to a word embedding
    """
    glove = {}
    with open(path, 'rb') as f:
        for line in f:
            parts = line.split()
            word = parts[0].decode('utf-8')
            if word in vocab:
                vector = np.array(parts[1:], dtype=np.float32)
                glove[word] = vector
    return glove


def strip_punctuation(doc, embeddings):
    """
    Remove leading and trailing punctuation from document tokens not found in a word embedding.

    Inputs:
        doc: list-like
            A document which has been tokenized.
        embeddings: collection
            A collection of tokens from a pre-trained word embedding, like GLoVe
    Output:
        token: str
            The output comes from yielding, rather than returning, token(s). This results in a generator.
    """
    for token in doc:
        if token not in embeddings:
            yield token.strip(punctuation)
        else:
            yield token


# classes
class W2vTokenizer(object):
    """
    Tokenize a document to prepare for vectorization via word embeddings. Compatible with scikit-learn Pipelines.
    Derived from Flatiron School's https://github.com/learn-co-curriculum/dsc-classification-with-word-embeddings-codealong

    Parameters:
        w2v: dict
            A dictionary of words and vectors from a pre-trained word embedding, like GloVe
    Methods:
        fit: dummy method required for pipeline functionality
        transform: tokenize a document
    """
    def __init__(self, w2v):
        self.w2v = w2v
    
    def fit(self, X, y):
        return self
            
    def transform(self, X):
        return X.map(word_tokenize).map(lambda x: list(strip_punctuation(x, self.w2v)))


class W2vVectorizer(object):
    """
    Vectorize a tokenized document using word embeddings.
    Derived from Flatiron School's https://github.com/learn-co-curriculum/dsc-classification-with-word-embeddings-codealong

    Parameters:
        w2v: dict
            A dictionary of words and vectors from a pre-trained word embedding, like GloVe
    Methods:
        fit: dummy method required for pipeline functionality
        transform: vectorize a document by calculating its mean word vector
    """
    def __init__(self, w2v):
        self.w2v = w2v
        if len(w2v) == 0:
            self.dimensions = 0
        else:
            self.dimensions = len(w2v[next(iter(w2v))])

    def fit(self, X, y):
        return self
    # 
    def transform(self, X):
        return np.array([np.mean([self.w2v[token] for token in doc if token in self.w2v]
                                 or [np.zeros(self.dimensions)], axis=0)
                         for doc in X])