import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
import joblib

#GLOBAL VARIABLS#
data = 1 #0 -> DementiaNet; 1 -> DementaBank;
rf = 1
svm = 0
log_reg = 0
outliers = 0
feat_sel = 0
pronunciation = 1
prosody = 1
save = True


def reshape_data(X):
    'Reshape 3D data to 2D'
    return X.reshape(X.shape[0], -1)

def remove_outliers(data, threshold=3):
    if not isinstance(data, np.ndarray):
        raise TypeError("data should be a numpy array")
    
    # Initialize a mask to keep track of non-outliers
    mask = np.ones(data.shape, dtype=bool)
    
    # Iterate over each feature dimension
    for i in range(data.shape[-1]):
        feature_data = data[..., i]
        mean = np.mean(feature_data)
        std = np.std(feature_data)
        z_scores = (feature_data - mean) / std
        mask[..., i] = np.abs(z_scores) < threshold
    
    # Set outliers to NaN
    filtered_data = np.where(mask, data, np.nan)
    return filtered_data

def get_score(model):
    print("extracting coeffiecents...")
    coefficients = model.coef_[0]
    score_sum = np.sum(coefficients)
    score_avg = np.mean(coefficients)

    print("Coefficient weights:", coefficients)
    print("Score (sum of coefficients):", score_sum)
    print("Score (average of coefficients):", score_avg)


def train_model_balanced(X_train, y_train, X_val, y_val, feature_type, estimator, sampling_strat=None, scale_features=True):
    'Train and evaluate a classifier model -- pass "smote" or "undersample" for sampling_strat' 
    
    # Reshape the data
    X_train_2d = reshape_data(X_train)
    X_val_2d = reshape_data(X_val)

    if scale_features:
        # Scale the features
        print('scaling features')
        scaler = StandardScaler()
        X_train_2d = scaler.fit_transform(X_train_2d)
        X_val_2d = scaler.transform(X_val_2d)
    else:
        scaler = None

    if sampling_strat == 'smote':
        smote = SMOTE(sampling_strategy=1.0, random_state=117)
        X_train_2d, y_train = smote.fit_resample(X_train_2d, y_train)
    if sampling_strat == 'undersample':
        rus = RandomUnderSampler(random_state=117)
        X_train_2d, y_train = rus.fit_resample(X_train_2d, y_train)
    elif sampling_strat == "combine":
        rus = RandomUnderSampler(sampling_strategy=0.5, random_state=117)
        X_train_2d, y_train = rus.fit_resample(X_train_2d, y_train)
        smote = SMOTE(sampling_strategy=1.0, random_state=117)
        X_train_2d, y_train = smote.fit_resample(X_train_2d, y_train)

    #model = estimator.fit(X_train_2d, y_train)
    #y_pred = model.predict(X_val_2d)

        # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, 
                               cv=3, n_jobs=-1, verbose=2, scoring='f1')

    # Fit the grid search to the data
    grid_search.fit(X_train_2d, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters for {feature_type} features:", best_params)

    # Get the best model
    model = grid_search.best_estimator_

    # Make predictions on the validation set
    y_pred = model.predict(X_val_2d)

    unique, counts = np.unique(y_train, return_counts=True)
    print("Current Class Balance: \n", np.asarray((unique, counts)).T)
    #print(f"Class distribution: {np.bincount(y_train)}")

    # Print the confusion matrix
    print(f"\nConfusion Matrix for {feature_type} features:")
    print(confusion_matrix(y_val, y_pred))

    # Print the classification report
    print(f"\nClassification Report for {feature_type} features:")
    print(classification_report(y_val, y_pred))

    return model, scaler

def feature_select(X, y, k_features):
    #Univariate Feature Selection - Mutual Info 
    print("\n--- Feature Selection ON ---")
    sel=SelectKBest(mutual_info_classif, k=k_features)
    fit_mod=sel.fit(X, y)    
    sel_idx=fit_mod.get_support()
    print ('Univariate Select - Mutual info classif: \n')

    selected_feat_indices = np.where(sel_idx)[0]
    print("Selected Features: ", selected_feat_indices,"\n")

    #updated data with selected features
    X = X[:, selected_feat_indices]

    print("New Data Shape: ", X.shape)

    return X

if data == 0:
    print("---\nDementiaNet LOADING\n---")
    prosody_features = np.load('data/DementiaNet/prosody_features.npy')
    pronunciation_features = np.load('data/DementiaNet/pronunciation_features.npy')
    labels_encoded = np.load('data/DementiaNet/prosody_features.npy')
if data == 1:
    print("---\nDementiaBank LOADING\n---")
    prosody_features = np.load('data\DementiaBank\prosody_features_sub_4.npy')
    pronunciation_features = np.load('data\DementiaBank\pronunciation_features_sub_4.npy')
    labels_encoded = np.load('data\DementiaBank\labels_encoded_sub_4.npy')
if data == 3:
    print("---\nDementiaNet LOADING\n---")
    prosody_dn = np.load('data/DementiaNet/prosody_features.npy')
    pronunciation_dn = np.load('data/DementiaNet/pronunciation_features.npy')
    labels_dn = np.load('data/DementiaNet/prosody_features.npy')
    print("---\nDementiaBank LOADING\n---")
    prosody_db = np.load('data\DementiaBank\prosody_features_sub_4.npy')
    pronunciation_db = np.load('data\DementiaBank\pronunciation_features_sub_4.npy')
    labels_db = np.load('data\DementiaBank\labels_encoded_sub_4.npy')
    #combining data
    pronunciation_features = np.vstack((pronunciation_dn, pronunciation_db))
    prosody_features = np.vstack((prosody_dn, prosody_db))
    labels_encoded = np.concatenate((labels_dn, labels_db))

if feat_sel == 1 and prosody == 1:
    prosody_features = reshape_data(prosody_features)
    prosody_features = feature_select(prosody_features, labels_encoded, 10)

if feat_sel == 1 and pronunciation == 1:
    pronunciation_features = reshape_data(pronunciation_features)
    pronunciation_features = feature_select(pronunciation_features, labels_encoded, 10)

if outliers == 1:
    print("---\nImputings Outliers with NaN...")
    prosody_features = remove_outliers(prosody_features, 3)
    pronunciation_features = remove_outliers(pronunciation_features, 3)

    prosody_features = reshape_data(prosody_features)
    pronunciation_features = reshape_data(pronunciation_features)

    print("Imputing NaN values with mean vals...")
    imp = SimpleImputer(missing_values = np.nan, strategy='mean')
    prosody_features = imp.fit_transform(prosody_features)
    pronunciation_features = imp.fit_transform(pronunciation_features)

print("Train/Test split ON\n---")
prosody_train, prosody_val, pronunciation_train, pronunciation_val, labels_train, labels_val = train_test_split(
    prosody_features, pronunciation_features, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42)


print("\nDataset preparation completed.")
print(f"Total data points: {len(labels_encoded)}")
print(f"Prosody features shape: {prosody_features.shape}")
print(f"Pronunciation features shape: {pronunciation_features.shape}")
print(f"Labels shape: {labels_encoded.shape}")
print(f"Class distribution: {np.bincount(labels_encoded)}\n")

unique, counts = np.unique(labels_train, return_counts=True)
print("Training Class Balance: \n", np.asarray((unique, counts)).T)



if rf == 1 and prosody == 1:
    prosody_rf = RandomForestClassifier(criterion='gini', n_estimators=200, n_jobs=-1, min_samples_split=2, min_samples_leaf=1, max_depth=30, random_state=117, class_weight='balanced_subsample')
    print("Training Model for Prosody Features")
    rf_prosody, scaler_prosody = train_model_balanced(prosody_train, labels_train, prosody_val, labels_val, "Prosody", prosody_rf, sampling_strat="combine", scale_features=True)
if rf == 1 and pronunciation == 1:
    pronunciation_rf = RandomForestClassifier(criterion='gini', n_estimators=200, n_jobs=-1, min_samples_split=2, min_samples_leaf=1, max_depth=30, random_state=117, class_weight='balanced_subsample')
    print("Training Model for Pronunciation Features")
    rf_prosody, scaler_prosody = train_model_balanced(prosody_train, labels_train, prosody_val, labels_val, "Pronunciation", pronunciation_rf, sampling_strat='combine', scale_features=True)

if svm == 1 and prosody == 1:
    prosody_svm = SVC(C = 0.12, kernel='rbf', gamma='auto', random_state=117)
    print("Training SVM Model for prosody")
    svm_prosody, scaler_prosody = train_model_balanced(prosody_train, labels_train, prosody_val, labels_val, "Prosody", prosody_svm, sampling_strat="combine", scale_features=True)
if svm == 1 and pronunciation == 1:
    pronunciation_svm = SVC(C = 0.12, kernel = 'rbf', gamma='auto', random_state=117)
    print("Training SVM Model for pronunciation")
    svm_pronunciation, scaler_pronunciation = train_model_balanced(pronunciation_train, labels_train, pronunciation_val, labels_val, "Pronunciation", pronunciation_svm, sampling_strat="combine", scale_features=True)    

if log_reg == 1 and prosody == 1:
    prosody_logreg = LogisticRegression(C=0.12, max_iter=1000)
    print('Training Logistic Regression Model for prosody')
    logreg_prosody, scaler_prosody = train_model_balanced(prosody_train, labels_train, prosody_val, labels_val, "Prosody", prosody_logreg, sampling_strat='smote', scale_features=True)
    get_score(logreg_prosody)
if log_reg == 1 and pronunciation == 1:
    pronunciation_logreg = LogisticRegression(C=0.12, max_iter=1000)
    print("Training logistic regression model for pronunciation")
    logreg_pronunciation, scaler_pronunciation = train_model_balanced(pronunciation_train, labels_train, pronunciation_val, labels_val, "Pronunciation", pronunciation_logreg, sampling_strat='smote', scale_features=True)
    get_score(logreg_pronunciation)

if save == True:
    joblib.dump(prosody_rf, 'prosody_rf.pkl')
    joblib.dump(pronunciation_rf, 'pronunciation_rf.pkl')