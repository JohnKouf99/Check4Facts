import os
import time

import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from check4facts.config import DirConf
from check4facts.metrics import accuracy, f1
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import precision_recall_curve

def proba_to_rating(true_proba):
    if true_proba < 0.25:
        rating = 1
    elif true_proba < 0.5:
        rating = 2
    elif true_proba < 0.75:
        rating = 3
    else:
        rating = 4
    return rating


class Predictor:

    def __init__(self, **kwargs):
        self.model_params = kwargs['model']
        self.features = kwargs['features']
        self.model = load(self.model_params['path'])
        
        
        


    def prepare_data(self, features_list):
        #create an index on the predict dataset, so it starts from the first element it predicts
        #this makes the statement_id column to be equivalent to the Fact id column for better examination
        #additionally, we remove the 'uknown' label from the statements df
        df = pd.read_csv(DirConf.PREDICT_CSV_FILE) #pd.read_csv(DirConf.CSV_FILE).tail(40)
        
        
        #df_idx = df.index[0]

        features_df = pd.DataFrame(features_list, columns=self.features,
                                   index= df['Fact id'].to_list())
        
        #10/10 added the following two lines of code
        # Drop statements with no resources
        # Drop 'uknown' label
        idx = list(features_df.dropna().index)
        #idx2 = df[df['Verdict'] != 2.0].index.tolist()
        #idx = list(set(idx1) & set(idx2))
        features_df = features_df[features_df.index.isin(idx)]
        x = np.vstack(features_df.apply(np.hstack, axis=1)).astype(np.float64)
        #10/10 normalize if we have llm embeddings
        if 'llm' in features_df.columns[0]:
            pass
            # scaler = MinMaxScaler()
            # x = scaler.fit_transform(x)
        return features_df.index, x

    def run(self, features_list):
        idx, x = self.prepare_data(features_list)
        preds = self.model.predict_proba(x)
        ratings = [proba_to_rating(p) for p in preds[:, 1]]

       



        # 'Statement id' col contains the 0-based index of the passed
        # statements for prediction. This is NOT equivalent to the
        # 'Fact id' property.
        result_df = pd.DataFrame(
            {'Statement id': idx, 'pred_0': preds[:, 0],
             'pred_1': preds[:, 1], 'rating': ratings})
        return result_df

    def run_dev(self):
        start_time = time.time()
        if not os.path.exists(DirConf.PREDICTOR_RESULTS_DIR):
            os.mkdir(DirConf.PREDICTOR_RESULTS_DIR)

        #8/10 change .head(60) to .tail(60) or less to generate a train/test split
        statement_df = pd.read_csv(DirConf.PREDICT_CSV_FILE) #pd.read_csv(DirConf.CSV_FILE).tail(40)

        #drop 'uknown' labels 10/10
        # idx1 = statement_df[statement_df['Verdict'] != 2].index.tolist()
        # statement_df = statement_df[statement_df.index.isin(idx1)]

        features_list = [pd.read_json(os.path.join(
            DirConf.FEATURES_RESULTS_DIR, f'{s_id}.json'), typ='series')
            for s_id in statement_df['Fact id']]
        result_df = self.run(features_list)
        path = os.path.join(DirConf.PREDICTOR_RESULTS_DIR, 'clf_results.csv')
        result_df.to_csv(path, index=False)
        stop_time = time.time()
        print(f'Model prediction done in {stop_time-start_time:.2f} secs.')
        pred_labels = result_df['rating'].replace({1:0, 2:0, 3:1, 4:1})

        
        
        true_labels = statement_df.loc[statement_df['Fact id'].isin(result_df['Statement id']),'Verdict']
        acc = accuracy(true_labels,pred_labels)
        precision = precision_score(true_labels,pred_labels)
        recall = recall_score(true_labels,pred_labels)
        print(f"Accuracy:  {acc*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall: {recall*100:.2f}%")

  

        # Create confusion matrix 21/10
        cm = confusion_matrix(true_labels, pred_labels)
        class_names = ['False', 'True']  
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.show(block=True)
        