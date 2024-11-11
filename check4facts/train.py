import os
import time

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from joblib import dump
from sklearn.preprocessing import MinMaxScaler
from check4facts.config import DirConf
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split



class Trainer:

    def __init__(self, **kwargs):
        self.classifiers_params = kwargs['classifiers']
        self.gs_params = kwargs['gs']
        self.features = kwargs['features']
        self.best_model = None
        #16/10 added stacking model variable to store the stacking model
        self.stacking_model = None
        


    def save_best_model(self, path):
        #16/10 save stacking model instead of the best estimator
        dump(self.best_model['best_estimator'], path)
        #dump(self.stacking_model, path)
        

 



    def gs(self, X, y):
        
        #X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        # y_train = y_train.reset_index(drop=True)
        # y_hold = y_hold.reset_index(drop=True)
        
        gs_results = []
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
        for clf in self.classifiers_params:
            gs = GridSearchCV(
                estimator=globals()[clf['class']](),
                param_grid=clf['params'],
                scoring=self.gs_params['scoring'],
                refit=self.gs_params['refit'],
                cv=skf, error_score='raise', n_jobs=-1).fit(X, y)
            gs_results.append({
                **{'clf': clf['name'], 'best_estimator': gs.best_estimator_,
                   'best_params': gs.best_params_},
                **{scorer: gs.cv_results_[f'mean_test_{scorer}'][
                    gs.best_index_] for scorer in self.gs_params['scoring']}})
        
        gs_results_df = pd.DataFrame(gs_results)

        #sort gs results based on accuracy
        gs_results_df = gs_results_df.sort_values(
            by=self.gs_params['refit'], ascending=False).reset_index(drop=True)
        gs_results_df.dropna(inplace=True)
        self.best_model = gs_results_df.iloc[0]

        
        ##16/10 stacking model initialization
        # final_estimator = xgb.XGBClassifier(eval_metric='error', random_state=42, 
        #                                     reg_alpha=0.5, reg_lambda=15, learning_rate=0.1)
        # final_estimator =  xgb.XGBClassifier(max_depth=3, n_estimators=50, learning_rate=0.1, 
        #                                      subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5,  
        #                                      reg_lambda=15, random_state=42)
        
        # self.stacking_model = StackingClassifier(estimators=list((name, est) for name, est in 
        #                                          zip(gs_results_df['clf'], gs_results_df['best_estimator'])),
        #                                          final_estimator=final_estimator, cv=5)
      

        # # Cross-validation setup
        # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # # Collect training and validation scores
        # train_scores = []
        # val_scores = []
        # #self.stacking_model = gs_results_df.loc[0,'best_estimator']
        
      
        # # Perform cross-validation
        # for train_index, val_index in cv.split(X_hold, y_hold):
        #     X_train, X_val = X_hold[train_index], X_hold[val_index]
        #     y_train, y_val = y_hold[train_index], y_hold[val_index]
            
        #     # Fit the stacking model on the training fold
        #     self.stacking_model.fit(X_train, y_train)
        #     #self.best_model.fit(X_train, y_train)
            
        #     # Predict on training data
        #     y_train_pred = self.stacking_model.predict(X_train)
        #     train_accuracy = accuracy_score(y_train, y_train_pred)
        #     train_scores.append(train_accuracy)
            
        #     # Predict on validation data
        #     y_val_pred = self.stacking_model.predict(X_val)
        #     val_accuracy = accuracy_score(y_val, y_val_pred)
        #     val_scores.append(val_accuracy)

        # # Convert scores to DataFrame for easier visualization
        # scores_df = pd.DataFrame({'Fold': range(1, 6), 'Train Accuracy': train_scores, 'Validation Accuracy': val_scores})

        # # Print the scores
        # print(scores_df)

        # # Plotting the training and validation scores
        # plt.figure(figsize=(10, 5))
        # plt.plot(scores_df['Fold'], scores_df['Train Accuracy'], marker='o', label='Train Accuracy')
        # plt.plot(scores_df['Fold'], scores_df['Validation Accuracy'], marker='o', label='Validation Accuracy')
        # plt.title('Training and Validation Accuracy per Fold')
        # plt.xlabel('Fold')
        # plt.ylabel('Accuracy')
        # plt.xticks(scores_df['Fold'])
        # plt.legend()
        # plt.grid()
        # plt.show(block=True)
        
        # self.stacking_model.fit(X_hold,y_hold)
        return pd.DataFrame(gs_results)

    def run(self, X, y):
        gs_results_df = self.gs(X, y).sort_values(
            by=self.gs_params['refit'], ascending=False).reset_index(drop=True)
        print(gs_results_df)
        print()
        print(gs_results_df.loc[:,['clf', 'accuracy']])
        self.best_model = gs_results_df.iloc[0]
       
        return

    def run_dev(self):
        start_time = time.time()
        if not os.path.exists(DirConf.TRAINER_RESULTS_DIR):
            os.mkdir(DirConf.TRAINER_RESULTS_DIR)
        # statement_df = pd.read_csv(DirConf.CSV_FILE).head(200)
        statement_df = pd.read_csv(DirConf.TRAIN_CSV_FILE)
        #10/10. Index is associated with the statement_id for debugging purposes
        df_idx = statement_df['Fact id'].to_list()
        features_df = pd.DataFrame([pd.read_json(os.path.join(
            DirConf.FEATURES_RESULTS_DIR, f'{s_id}.json'), typ='series')
            for s_id in statement_df['Fact id']], columns=self.features,
            index= df_idx)
        

        # Drop statements with no resources
        idx1 = list(features_df.dropna().index)
        # Drop statements with 'UNKNOWN' label
        #idx2 = list(statement_df[~statement_df['Verdict'].isin(
            #['UNKNOWN', 'Unknown'])].index)
        idx2 = statement_df[statement_df['Verdict'] != 2]['Fact id'].tolist()
        idx = list(set(idx1) & set(idx2))
       
        #changed .index to ['Fact id']
        statement_df = statement_df[statement_df['Fact id'].isin(idx)]
        features_df = features_df[features_df.index.isin(idx)]
        X = np.vstack(features_df.apply(np.hstack, axis=1)).astype(np.float64) #changed from float
        y = statement_df.set_index('Fact id')['Verdict'].astype(int) #.replace({'TRUE': 1.0, 'FALSE': 0.0})
        y=y.reset_index(drop=True)
        


        # Initialize new array to store reduced features
        # X_reduced = np.zeros((num_samples, reduced_num_features))
        
        # # Loop over samples, apply PCA, and store results in X_reduced
        # for i, x in enumerate(X):
        #     x = x.reshape(2, len(x) // 2)       # Reshape x
        #     x = np.transpose(x)                  # Transpose x
        #     X_reduced[i] = pca.fit_transform(x).flatten()  # Apply PCA and save flattened result to X_reduced
        # print(X_reduced.shape)
        # X = X_reduced


            
        
        

        

        
        
        #10/10 implement SMOTE method to combat class imbalance
        if 'llm' in features_df.columns[0]:
            pass
            # undersampler = RandomUnderSampler(random_state=42)
            # X, y = undersampler.fit_resample(X, y)
            # print(f'Before: {y.shape}')
            # smote = BorderlineSMOTE(random_state=42)
            # X, y = smote.fit_resample(X,y)
            #y = y.reset_index(drop=True)
            # print(f'After: {y.shape}')

        self.run(X, y)

        #8/10 changed file name
        # fname = 'clf_' + self.best_model['clf'] + '_' + time.strftime(
        #     '%Y-%m-%d-%H:%M') + '.joblib'
        fname = 'clf'+'.joblib'
        path = os.path.join(DirConf.TRAINER_RESULTS_DIR, fname)
        self.save_best_model(path)
        stop_time = time.time()
        print(f'Model training done in {stop_time-start_time:.2f} secs.')
        print(f'Best model was: {self.best_model['clf']}')
