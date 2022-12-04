from tokenize import String
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pylab as pylab
import time

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.svm import SVC

import sklearn.metrics as metrics
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# from sklearn.metrics import silhouette_score
# from sklearn.metrics import confusion_matrix


import scipy.stats as stats
from scipy.stats import chi2_contingency
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV
# from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

#from lightgbm import LGBMClassifier

import xgboost as xgb
XGBClassifier = xgb.XGBClassifier

from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score 
from numpy import mean
from numpy import std

from pandas_profiling import ProfileReport
import plotly.express as px

import matplotlib.pyplot as plt

import itertools

# global definitions for graphics
sub_plots_title_fontSize = 12
sub_plots_xAxis_fontSize = 10
sub_plots_yAxis_fontSize = 10
sub_plots_label_fontSize = 10
heatmaps_text_fontSize = 8
plots_title_fontSize = 14
plots_title_textColour = 'black'
plots_legend_fontSize = 12
plots_legend_textColour = 'black'

class Functions:

###################################################################################################
    def confusion_matrix(y, y_predicted):
        confusion_matrix = metrics.confusion_matrix(y, y_predicted)
        return metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

###################################################################################################
    def variation(a, b):
        if(a == b):
            return 0
        if(b == 0):
            if(a < 0):
                return -1
            return 1
        return (a-b)/b

###################################################################################################

    def df_shape(X: pd.DataFrame = None,Y: pd.DataFrame = None,description: str = '', y1 = 1, y0 = 0):
        df = pd.DataFrame(columns = [
            'Description',
            'X_Cols',
            'X_Rows',
            'Y_Rows',
            'Y0',
            'Y1',
            'Y1 %'
        ])

        if X is None:
            return df

        df = df.append({
            'Description': description,
            'X_Cols': X.shape[1],
            'X_Rows': X.shape[0],
            'Y_Rows': Y.shape[0],
            'Y0': Y[Y == y0].count(),
            'Y1':Y[Y == y1].count(),
            'Y1 %': round(100 * Y[Y == y1].count() / Y.count(),2)
         },ignore_index=True)
        return df
        

###################################################################################################
    def correlationWithY(X,Y,cols, YcolName):
        df_result = pd.DataFrame(columns = ['Value','P-value','P-value (formated)','Spearman_Correl'])
        temp_correl = X.join(Y).corr(method='spearman')
        for c in cols:
            dfObserved = pd.crosstab(Y,X[c]) 
            chi2, p, dof, expected = stats.chi2_contingency(dfObserved.values)
            df_result = df_result.append({'Value':c,'P-value':p, 'P-value (formated)': '{:.20e}'.format(p), 'Spearman_Correl': temp_correl[YcolName][c]},ignore_index=True)
        return df_result.sort_values(['P-value'])

###################################################################################################
    def analyse_pca_fit(dataframe, cumulativeVarianceTreshold=0.95, titleExtra=""):
        """
        Returns plots at table of Variance explained by PCA components.
        Returns the minimum number of components needed for having cumulativeVarianceTreshold (default 95%)
        `dataframe`: dataframe with data
        `cumulativeVarianceTreshold`: cumulative variance for wicht the minimum number of components needed will be calculated and returned
        `titleExtra`: title added to the plots after "Clusters - "
        Fit the PCA algorithm to data
        """

        pca = PCA().fit(dataframe)

        # Show the variance per component
        pcaevr = ['{:f}'.format(item) for item in pca.explained_variance_ratio_]
        pcaDF = pd.DataFrame({'Component': range(1, len(dataframe.columns)+1),
                            'Eigenvalue': pca.explained_variance_,
                            "Diff.": np.insert(np.diff(pca.explained_variance_), 0, 0),
                            'Var. explained (%)': pcaevr,
                            'Cum. var. explained': np.cumsum(pca.explained_variance_ratio_),
                            
                            'Components': ",".join([str(i) for i in pca.components_])
                            })
        # Plot the cumulative explained variance

        # Draw
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_))
        # Decoration
        sns.despine()
        plt.xlabel('Number of principal components')
        plt.ylabel('Variance explained')
        plt.axhline(y = 0.95, color='k', linestyle='--', label = '95% Explained Variance')
        plt.rc('axes', labelsize=sub_plots_label_fontSize)
        
        plt.title('Explained variance by components - ' +
                titleExtra, fontsize=plots_title_fontSize)
        # plt

        #
        component_number = pcaDF[pcaDF['Cum. var. explained']
                                > cumulativeVarianceTreshold].index[0] + 1
        print(
            f"Number of components for {cumulativeVarianceTreshold} cumulative variance explained: " + str(component_number))

        print(pcaDF)
        return component_number

###################################################################################################
    def select_best_features(X, y, num_vars, cat_vars, n_splits = 10, rfe_n_features_to_select = 0.75):
        rfe_columns = num_vars + cat_vars
        chi2_columns = cat_vars

        rfe_results = [0] * len(rfe_columns)
        chi2_results = {chi2_columns[i]: 0 for i in range(len(chi2_columns))}

        skf = StratifiedKFold(n_splits)
        counter = 0
        for train_index, val_index in skf.split(X, y):
            counter +=1
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            
            # get all numerical variables
            X_train_rfe = X_train[rfe_columns]
            # X_val_num = X_val[rfe_columns]
            
            # get all categorical variables
            X_train_cat = X_train[chi2_columns]
            # X_val_cat = X_val[cat_vars]
            
            # Apply scaling to numerical data
            scaler = MinMaxScaler().fit(X_train_rfe)
            X_train_scaled = pd.DataFrame(scaler.transform(X_train_rfe), columns = X_train_rfe.columns, index = X_train_rfe.index,) # MinMaxScaler in the training data
            
            # Check which features to use using RFE
            model = LogisticRegression()
            rfe = RFE(estimator = model, n_features_to_select= rfe_n_features_to_select)
            X_rfe = rfe.fit_transform(X = X_train_scaled, y = y_train)
            for i,r in enumerate(rfe.support_):
                if r: rfe_results[i] = rfe_results[i] + 1
            # selected_features = pd.Series(rfe.support_, index = X_train_scaled.columns)
            # print(selected_features)
            
            # Check which features to use using Chi-Square
            def TestIndependence(X,y,var,alpha=0.05):        
                dfObserved = pd.crosstab(y,X) 
                chi2, p, dof, expected = stats.chi2_contingency(dfObserved.values)
                dfExpected = pd.DataFrame(expected, columns=dfObserved.columns, index = dfObserved.index)
                if p<alpha:
                    chi2_results[var] = chi2_results[var] + 1
            
            for var in X_train_cat:
                TestIndependence(X_train_cat[var],y_train, var)
        
        
        rfe_results =  pd.DataFrame(rfe_results, index=rfe_columns, columns = ['Qty_Selected_By_Rfe']).copy().sort_values(['Qty_Selected_By_Rfe'], ascending=False)
        chi2_results = pd.DataFrame(chi2_results.values(), index=chi2_results.keys(), columns = ['Qty_Selected_By_Chi2']).copy().sort_values(['Qty_Selected_By_Chi2'], ascending=False)

        # sns.countplot(y = 'Qty_Selected_By_Rfe', data = rfe_results)
        plt.figure(figsize=(10, min(10,len(rfe_results) / 2 )))
        plt.title(f'Feature importance: RFE (StratifiedKFold K = {n_splits}, Feature to select = {rfe_n_features_to_select})')
        sns.barplot(y=rfe_results.index, x='Qty_Selected_By_Rfe', data=rfe_results, palette = ['royalblue' for i in rfe_results.index])
        plt.figure(figsize=(10, min(10,len(chi2_results) / 2 )))
        plt.title(f'Feature importance (categorical): Chi2 (StratifiedKFold K = {n_splits})')
        sns.barplot(y=chi2_results.index, x='Qty_Selected_By_Chi2', data=chi2_results, palette = ['royalblue' for i in chi2_results.index])
        return rfe_results, chi2_results

###################################################################################################
    def plot_kmeans_elbow_and_silhouette(dataframe, nMax=5, titleExtra='', randomState=50):
        """
        Returns plots of 'Elbow' and 'Silhouette' of k-means++ clustering of dataframe
        `dataframe`: dataframe with the data
        `nMax`: max of K clusters to analyse
        `titleExtra`: title added to the plots after "K-means analysis - Elbow/Silhouette methods - "
        `randomState`: random state passed to KMeans (optional default = 50)
        Example:
        plot_kmeans_elbow_and_silhouette(df_satisfaction, 11, "Satisfaction perspective - Without PCA")
        """

        # calculation
        elbowValues = []
        sil = []
        for i in range(1, nMax+1):
            k_means = KMeans(n_clusters=i, init='k-means++',
                            random_state=randomState)
            k_means.fit(dataframe)
            labels = k_means.labels_
            elbowValues.append(k_means.inertia_)
            if (i > 1):
                sil.append(silhouette_score(dataframe, labels, metric='euclidean'))

        # Draw
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        axes[0].plot(np.arange(1, nMax+1), elbowValues, 'bx-')
        axes[1].plot(np.arange(2, nMax+1), sil, 'bx-')

        sns.lineplot(ax=axes[0], x=range(1, nMax+1), y=elbowValues, marker='o')
        sns.lineplot(ax=axes[1], x=range(2, nMax+1), y=sil, marker='o')

        axes[0].set_title('Elbow', fontsize=sub_plots_title_fontSize)
        axes[1].set_title('Silhouette', fontsize=sub_plots_title_fontSize)

        axes[0].set_xlabel('K')
        axes[1].set_xlabel('K')

        axes[0].set_ylabel('Sum of Squared Distances')
        axes[1].set_ylabel('Silhouette score')

        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

        sns.despine()
        fig.tight_layout(pad=3.25)
        plt.rc('axes', labelsize=sub_plots_label_fontSize)
        fig.suptitle("K-means analysis - Elbow/Silhouette methods - " +
                    titleExtra, fontsize=plots_title_fontSize)

###################################################################################################
    def plot_cluster_cardinality_satisfaction(dataframe, clusterIdColumnName, distanceToCentroidColumnName, titleExtra=""):
        """
        Returns plots of 'Cardinality', 'Magnitude' and 'Cardinality / Magnitude' of a cluster result
        `dataframe`: dataframe with data
        `clusterIdColumnName`: name of the column that has the cluster Id
        `distanceToCentroidColumnName`: name of the column that has the distance to the erspective cluster centroid
        `titleExtra`: title added to the plots after "Clusters - "
        Example:
        plot_cluster_cardinality_satisfaction(df, 'cluster_id', 'distanceToCentroid', 'Satisfaction perspective - PCA K2')
        """

        # Sum magnitudes per cluster
        freqByCluster = dataframe[clusterIdColumnName].groupby(
            dataframe[clusterIdColumnName]).count()
        # Sum magnitudes per cluster
        magnitude = dataframe[distanceToCentroidColumnName].groupby(
            dataframe[clusterIdColumnName]).sum()
        # sns.set_theme(style="whitegrid")
        # Draw
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

        ax = sns.countplot(ax=axes[0], x=clusterIdColumnName, data=dataframe)
        ax.bar_label(ax.containers[0], fontsize=sub_plots_label_fontSize)

        ax = sns.barplot(ax=axes[1], x=magnitude.index, y=magnitude.values)
        ax.bar_label(ax.containers[0], fontsize=sub_plots_label_fontSize)

        ax = sns.regplot(ax=axes[2], x=freqByCluster, y=magnitude,
                        scatter=True, seed=123, truncate=False, ci=None)

        axes[0].set_title('Cardinality', fontsize=sub_plots_title_fontSize)
        axes[1].set_title('Magnitude', fontsize=sub_plots_title_fontSize)
        axes[2].set_title('Cardinality / Magnitude',
                        fontsize=sub_plots_title_fontSize)

        axes[0].set_xlabel('Cluster')
        axes[1].set_xlabel('Cluster')
        axes[2].set_xlabel('Cardinality')

        axes[0].set_ylabel('Count')
        axes[1].set_ylabel('Total Magnitude')
        axes[2].set_ylabel('Magnitude')

        sns.despine()
        fig.tight_layout(pad=3.5)
        plt.rc('axes', labelsize=sub_plots_label_fontSize)
        fig.suptitle("Clusters - " + titleExtra, fontsize=plots_title_fontSize)

###################################################################################################
    def plot_pca_preview_2d(perspectives, titleExtra='', randomState=50, size = (2,3)):
        """
        Scatter plot with the first PCA components defined by perspectives configuration
        Fit the PCA algorithm to data
        """

        pca = PCA(n_components=2)
        fig, axes = plt.subplots(nrows=size[0], ncols=size[1], figsize=(size[1] * 6, size[0] * 4))
        for p, subplot in zip(perspectives, axes.flatten()):
            pca.fit(p['df'])
            p_pca = pca.transform(p['df'])

            explained_variance_ratio = pca.explained_variance_ratio_[
                0] + pca.explained_variance_ratio_[1]

            sns.scatterplot(x=p_pca[:, 0], y=p_pca[:, 1], ax=subplot,
                            hue=p['hue'], edgecolor='none', alpha=0.5, cmap='viridis')

            subplot.set_title(
                p['name'] + f"\n(Explained variance 2 components {'{:.1f}'.format(100 * explained_variance_ratio)}%)", fontsize=sub_plots_title_fontSize)
            subplot.set_xlabel('Component 1')
            subplot.set_ylabel('Component 2')

        sns.despine()
        fig.tight_layout(pad=3.25)
        plt.rc('axes', labelsize=sub_plots_label_fontSize)
        fig.suptitle("PCA 2 main components - " + titleExtra,
                    fontsize=plots_title_fontSize)

###################################################################################################
    def plot_pca_2d(pcas, names = None, centers = None, hues = None, titleExtra='', randomState=50, size = (2,3)):
        fig, axes = plt.subplots(nrows=size[0], ncols=size[1], figsize=(size[1] * 6, size[0] * 4))
        for p, subplot,index in zip(pcas, axes.flatten(), range(100)):

            # explained_variance_ratio = p.explained_variance_ratio_[0] + p.explained_variance_ratio_[1]

            if hues != None:
                hue = hues[index]
            else:
                hue = None

            sns.scatterplot(x=p[:, 0], y=p[:, 1], ax=subplot,
                            hue=hue, edgecolor='none', alpha=0.5, cmap='viridis', legend=False)

            if centers != None:
                subplot.scatter(centers[index][:, 0], centers[index][:, 1], c='black', s=200, alpha=0.5)

            if names != None:
                name = names[index] + "\n"
            else:
                name = ''

            subplot.set_title(name , fontsize=sub_plots_title_fontSize)
            subplot.set_xlabel('Component 1')
            subplot.set_ylabel('Component 2')

        sns.despine()
        fig.tight_layout(pad=3.25)
        plt.rc('axes', labelsize=sub_plots_label_fontSize)
        fig.suptitle("PCA 2 main components - " + titleExtra,
                    fontsize=plots_title_fontSize)

###################################################################################################
    def plot_pca_3d(perspective, titleExtra='', randomState=50):
        """
        Scatter plot with the first PCA components defined by perspectives configuration
        Fit the PCA algorithm to data
        """

        pca = PCA(n_components=3)
        pca.fit(perspective['df'])
        p_pca = pca.transform(perspective['df'])

        x = p_pca[:, 0]
        y = p_pca[:, 1]
        z = p_pca[:, 2]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection = '3d')

        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")

        ax.scatter(x, y, z)

###################################################################################################
    def apply_kmeans(dataframe, k, randomState = 50):

        kmeans = KMeans(n_clusters=k, random_state=randomState)
        allDistances = kmeans.fit_transform(dataframe)
        y_kmeans = kmeans.predict(dataframe)

        return (y_kmeans, np.min(allDistances, axis=1), kmeans.cluster_centers_)

###################################################################################################
    def apply_kprototypes(dataframe, k, categorical_columns, randomState = 50,  nJobs = -1):

        kproto = KPrototypes(n_clusters= k, init='Huang', n_jobs = nJobs, random_state=randomState)
        y_kproto = kproto.fit_predict(dataframe, categorical=categorical_columns)

        return y_kproto

###################################################################################################
    def plotMultipleBoxPlot(dataframe, xVariables, hue, title, size=(18,8)):
        d=dataframe[xVariables + [hue]]
        d=pd.melt(d,id_vars=hue, var_name="Variables", value_name="Values")

        # Draw
        fig, axes = plt.subplots(figsize=size)

        sns.boxplot(
                ax=axes,
                x="Variables",
                y="Values",
                hue=hue,
                data=d,
                width=0.5,
                linewidth=1,
                fliersize=2
        )

        sns.despine()
        axes.set_xticklabels(axes.get_xticklabels(),rotation = 45)
        axes.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0)
        fig.tight_layout(pad=3.25)
        fig.suptitle(title, fontsize=plots_title_fontSize)

    def plot_missing_values(df):
        data = [(col, df[col].isnull().sum() / len(df)) 
            for col in df.columns]
        col_names = ['column', 'percent_missing']
        missing_df = pd.DataFrame(data, columns=col_names).sort_values('percent_missing')
        pylab.rcParams['figure.figsize'] = (15, 8)
        plt = missing_df.plot(kind='barh', x='column', y='percent_missing'); 

###################################################################################################
###################################################################################################

    def plot_precision_recall_vs_threshold(title_extra, precisions, recalls, thresholds):
        
        plt.figure(figsize=(8, 8))
        plt.title("Precision and Recall Scores - " + title_extra)
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
        plt.ylabel("Score")
        plt.xlabel("Decision Threshold")
        plt.legend(loc='best')

    def batch_predict(clf,data):
        y_data_pred=[]
        tr_loop=data.shape[0]-data.shape[0]%10000
        for i in range(0,tr_loop,10000):
            y_data_pred.extend(clf.predict_proba(data[i:i+10000])[:,1])

        if data.shape[0]%10000!=0:
            y_data_pred.extend(clf.predict_proba(data[tr_loop:])[:,1])
    
        return y_data_pred

    def best_threshold(thresh,fpr,tpr):
        t=thresh[np.argmax(tpr*(1-fpr))]
    
        #print("the maximum value of tpr*(1-fpr)",max(tpr*(1-fpr)),"for threshold",np.round(t,3))
        return t

    def prediction_(proba,thresh):
        #print("theshold",threshold)
        predictions=[]
        for i in proba:
            if i>=thresh:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions


    def plot_roc_auc_(classifier,train,test,y_train,y_test):
        # %pylab inline
        classifier.fit(train,y_train)
        y_train_predict=Functions.batch_predict(classifier,train)
        y_test_predict=Functions.batch_predict(classifier,test)
        train_fpr,train_tpr,train_threshold=metrics.roc_curve(y_train,y_train_predict)
        test_fpr,test_tpr,test_threshold=metrics.roc_curve(y_test,y_test_predict)
        print("Train Threshold:", train_threshold)
        print("Test Threshold:", test_threshold)
        plt.plot(train_fpr,train_tpr,label='Train ROC Curve')
        plt.plot(test_fpr,test_tpr,label='Test ROC Curve')
        plt.legend()
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('TPR vs FPR graph')
        plt.grid()
        plt.show()

        train_auc = metrics.auc(train_fpr,train_tpr)
        test_auc = metrics.auc(test_fpr,test_tpr)
        print("Train AUC Score",train_auc)
        print("Test AUC Score",test_auc)
    
        fig=plt.figure()
        ax=fig.add_subplot(111)
        best_t= Functions.best_threshold(train_threshold,train_fpr,train_tpr)
        print("Best Threshold:", best_t)
        print("Train Confusion Matrix")
        y_train_predicti=Functions.prediction_(y_train_predict,best_t)
        train_matrix=metrics.confusion_matrix(y_train,y_train_predicti)
        sns.heatmap(train_matrix,annot=True,fmt='d')
        plt.show(ax)
        print()
        fig=plt.figure()
        ax1=fig.add_subplot(111)
        print("Test Cofusion Matrix")
        y_test_predicti=Functions.prediction_(y_test_predict,best_t)
        test_matrix=metrics.confusion_matrix(y_test,y_test_predicti)
        sns.heatmap(test_matrix,annot=True,fmt='d')
        plt.show(ax1)
        return train_auc,test_auc


    def plot_feature_importance(importance_func,names,model_type):
    
        #Create arrays from feature importance and feature names
        #%pylab inline
        feature_importance = np.array(importance_func)
        feature_names = np.array(names)

        #Create a DataFrame using a Dictionary
        data={'feature_names':feature_names,'feature_importance':feature_importance}
        fi_df = pd.DataFrame(data)

        #Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

        plt.figure(figsize=(10,8))
    
        sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])

        plt.title( 'FEATURE IMPORTANCE '+model_type)
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')
