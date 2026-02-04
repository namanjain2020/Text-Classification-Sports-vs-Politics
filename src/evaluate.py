from sklearn.metrics import classification_report, accuracy_score

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['SPORTS','POLITICS'], digits=4)
    return accuracy, report, y_pred