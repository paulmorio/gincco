# Utility for baseline SVM model
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, roc_auc_score
# Baseline SVM
def rbf_svm_classify(X_train, X_test, Y_train, Y_test):
    """
    Parameters
    ----------
    X_train : numpy.ndarray 
        training feature vectors
    X_test : numpy.ndarray 
        testing feature vectors
    Y_train : numpy.ndarray 
        training set labels
    Y_test : numpy.ndarray 
        test set labels

    Returns
    -------
    tuple
        (acc, precision, recall, fbeta_score)
    """
    params = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}
    classifier = GridSearchCV(SVC(gamma="scale", class_weight="balanced"), params, cv=5, scoring='balanced_accuracy', verbose=0, n_jobs=-1)
    
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    precision, recall, fbeta_score, support = precision_recall_fscore_support(Y_test, Y_pred) # fbeta aka f1_score, f-measure

    return (acc, precision, recall, fbeta_score)


def rbf_svm_classify_roc(X_train, X_test, Y_train, Y_test):
    """
    Parameters
    ----------
    X_train : numpy.ndarray 
        training feature vectors
    X_test : numpy.ndarray 
        testing feature vectors
    Y_train : numpy.ndarray 
        training set labels
    Y_test : numpy.ndarray 
        test set labels

    Returns
    -------
    tuple
        (acc, precision, recall, fbeta_score)
    """
    params = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}
    classifier = GridSearchCV(SVC(gamma="scale", class_weight="balanced", probability=True), params, cv=5, scoring='balanced_accuracy', verbose=0, n_jobs=-1)
    
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    Y_probs = classifier.predict_proba(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    precision, recall, fbeta_score, support = precision_recall_fscore_support(Y_test, Y_pred) # fbeta aka f1_score, f-measure
    roc_auc = roc_auc_score(Y_test, Y_probs, multi_class="ovr")

    return (acc, precision, recall, fbeta_score, roc_auc)

def rbf_svm_classify_roc_binary(X_train, X_test, Y_train, Y_test):
    """
    Parameters
    ----------
    X_train : numpy.ndarray 
        training feature vectors
    X_test : numpy.ndarray 
        testing feature vectors
    Y_train : numpy.ndarray 
        training set labels
    Y_test : numpy.ndarray 
        test set labels

    Returns
    -------
    tuple
        (acc, precision, recall, fbeta_score)
    """
    params = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}
    classifier = GridSearchCV(SVC(gamma="scale", class_weight="balanced", probability=True), params, cv=5, scoring='balanced_accuracy', verbose=0, n_jobs=-1)
    
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    Y_probs = classifier.predict_proba(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    precision, recall, fbeta_score, support = precision_recall_fscore_support(Y_test, Y_pred) # fbeta aka f1_score, f-measure
    roc_auc = roc_auc_score(Y_test, Y_pred)

    return (acc, precision, recall, fbeta_score, roc_auc)