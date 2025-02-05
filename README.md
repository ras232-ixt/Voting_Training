# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# Ensure `data_final_vot` is defined
X = data_final_vot.loc[:, data_final_vot.columns != 'ERCOUNT']
y = data_final_vot['ERCOUNT']

# Splitting data into training and testing sets (Only once!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Handling class imbalance with SMOTE (Applied only to training data)
smote = SMOTE(random_state=0)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Create a 10-fold cross-validation scheme
kfold = KFold(n_splits=10, random_state=7, shuffle=True)

# Create the individual models
estimators = [
    ('logistic', LogisticRegression(random_state=0)),
    ('cart', DecisionTreeClassifier(random_state=0)),
    ('svm', SVC(probability=True, random_state=0))
]

# Create the ensemble model using Soft Voting
ensemble = VotingClassifier(estimators, voting='soft')

# Perform cross-validation
cv_results = cross_val_score(ensemble, X_train_resampled, y_train_resampled, cv=kfold)
print(f"Cross-validation Mean Accuracy: {cv_results.mean():.4f}")

# Train the ensemble model on resampled training data
ensemble.fit(X_train_resampled, y_train_resampled)

# Make predictions on the original (untouched) test set
y_pred = ensemble.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot ROC Curve
roc_auc = roc_auc_score(y_test, ensemble.predict(X_test))
fpr, tpr, _ = roc_curve(y_test, ensemble.predict_proba(X_test)[:, 1])

plt.figure()
plt.plot(fpr, tpr, label=f'LR+DT+SVM (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('LR+DT+SVM_ROC')
plt.show()
