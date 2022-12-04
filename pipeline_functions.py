from imports import *


###################################################################################################
# 
###################################################################################################

def checkIfFeaturesExists(X, feature):
    if isinstance(X, np.ndarray): return False
    return feature in X
class OutliersProcessingStrategy(Enum):
    NONE = 1
    CLIP = 2

class OutliersProcessing(BaseEstimator, TransformerMixin):
    
    def __init__(self, strategy: OutliersProcessingStrategy = OutliersProcessingStrategy.NONE):
        self.strategy = strategy
        
    def fit(self, X, y = None):

        return self
    
    def transform(self, X, y = None):
        if(self.strategy == OutliersProcessingStrategy.CLIP):
            X_ = X.copy()
            if checkIfFeaturesExists(X_,'national_inv'): X_['national_inv'].clip(lower=0, upper=5487, inplace=True)
            if checkIfFeaturesExists(X_,'in_transit_qty'): X_['in_transit_qty'].clip(upper=5510, inplace=True)
            if checkIfFeaturesExists(X_,'forecast_3_month'): X_['forecast_3_month'].clip(upper=2280, inplace=True)
            if checkIfFeaturesExists(X_,'forecast_6_month'): X_['forecast_6_month'].clip(upper=4335, inplace=True)
            if checkIfFeaturesExists(X_,'forecast_9_month'): X_['forecast_9_month'].clip(upper=6316, inplace=True)
            if checkIfFeaturesExists(X_,'sales_1_month'): X_['sales_1_month'].clip(upper=693, inplace=True)
            if checkIfFeaturesExists(X_,'sales_3_month'): X_['sales_3_month'].clip(upper=2229, inplace=True)
            if checkIfFeaturesExists(X_,'sales_6_month'): X_['sales_6_month'].clip(upper=4410, inplace=True)
            if checkIfFeaturesExists(X_,'sales_9_month'): X_['sales_9_month'].clip(upper=6698, inplace=True)
            if checkIfFeaturesExists(X_,'min_bank'): X_['min_bank'].clip(upper=679, inplace=True)

            return X_
        return X

###################################################################################################
# 
###################################################################################################
class CustomScalerType(Enum):
    MINMAX = 2
    STANDARD = 3

class CustomScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, strategy: CustomScalerType = CustomScalerType.MINMAX):
        self.strategy = strategy
        
    def fit(self, X, y = None):

        if(self.strategy == CustomScalerType.MINMAX):
            self.scaler = MinMaxScaler()

        if(self.strategy == CustomScalerType.STANDARD):
            self.scaler = StandardScaler()

        self.scaler.fit(X)
        return self
    
    def transform(self, X, y = None):
        return self.scaler.transform(X)

###################################################################################################
# 
###################################################################################################
class InbalancedDataSamplingStrategy(Enum):
    NONE = 1
    SMOTE = 2
    ADASYN = 3
    NEARMISS = 4

class ImbalancedDataSampling(BaseEstimator):
    """
    """
    
    def __init__(self, strategy: InbalancedDataSamplingStrategy = InbalancedDataSamplingStrategy.NONE):
        self.strategy = strategy
        
    def fit_resample(self, X, y):
        return self.resample(X, y)
        
    def resample(self, X, y):
        
        if(self.strategy == InbalancedDataSamplingStrategy.SMOTE):
            return SMOTE(random_state=config.proj_random_state).fit_resample(X, y)

        if(self.strategy == InbalancedDataSamplingStrategy.ADASYN):
            return ADASYN(random_state=config.proj_random_state).fit_resample(X, y)

        if(self.strategy == InbalancedDataSamplingStrategy.NEARMISS):
            return NearMiss().fit_resample(X, y)

        return X,y


###################################################################################################
# 
################################################################################################### 
def batch_predict(clf,data):
    y_data_pred=[]
    tr_loop=data.shape[0]-data.shape[0]%10000
    for i in range(0,tr_loop,10000):
        y_data_pred.extend(clf.predict_proba(data[i:i+10000])[:,1])
        
    if data.shape[0]%10000!=0:
        y_data_pred.extend(clf.predict_proba(data[tr_loop:])[:,1])
    
    return y_data_pred

def prediction_(proba,thresh):
    predictions=[]
    for i in proba:
        if i>=thresh:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


def calculate_scores(label, y_real, y_predict):
    f1score  = f1_score(y_real, y_predict)
    f1scoreMicro  = f1_score(y_real, y_predict, average='micro')
    mcc = matthews_corrcoef(y_real, y_predict)
    report = classification_report(y_real, y_predict)
    print(label)
    print("F1-score:",f1score, "F1-score-micro:",f1scoreMicro, "MCC:", mcc)
    print(report)

    
def best_threshold_auc(thresh,fpr,tpr):
    # G-mean
    return thresh[np.argmax(tpr*(1-fpr))]

def best_threshold_auc_youdenJ(thresh,fpr,tpr):
    # G-mean
    return thresh[np.argmax(tpr-fpr)]

def best_threshold_auc_tpr(thresh,fpr,tpr):
    return thresh[np.argmax(tpr*tpr*(1-fpr))]

def best_threshold_auc_fpr(thresh,fpr,tpr):
    return thresh[np.argmax(tpr*(1-fpr)*(1-fpr))]

def best_threshold_f1score(thresh,precision,recall):
    t=thresh[np.argmax((2 * precision * recall) / (precision + recall))]
    
    #print("the maximum value of tpr*(1-fpr)",max(tpr*(1-fpr)),"for threshold",np.round(t,3))
    return t

def plot_curves_and_scores(title, classifier,train,test,y_train,y_test):

    classifier.fit(train,y_train)
    y_train_predict=batch_predict(classifier,train)
    y_test_predict=batch_predict(classifier,test)

    # ROC
    train_fpr,train_tpr,train_threshold=roc_curve(y_train,y_train_predict)
    test_fpr,test_tpr,test_threshold=roc_curve(y_test,y_test_predict)
    plt.plot(train_fpr,train_tpr,label='Train ROC Curve')
    plt.plot(test_fpr,test_tpr,label='Test ROC Curve')
    plt.legend()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('TPR vs FPR graph Curve - ' + title)
    plt.grid()
    plt.show()

    train_auc = auc(train_fpr,train_tpr)
    test_auc = auc(test_fpr,test_tpr)
    print("Train AUC Score",train_auc)
    print("Test AUC Score",test_auc)

    best_t= 0.5
    result_for_treshold(title + ' - Treshold AUC 0.5', best_t, y_train, y_train_predict, y_test, y_test_predict)

    best_t= best_threshold_auc(train_threshold,train_fpr,train_tpr)
    result_for_treshold(title + ' - Treshold Best AUC', best_t, y_train, y_train_predict, y_test, y_test_predict)

    best_t= best_threshold_auc_youdenJ(train_threshold,train_fpr,train_tpr)
    result_for_treshold(title + ' - Treshold Best AUC youdenJ', best_t, y_train, y_train_predict, y_test, y_test_predict)

    best_t= best_threshold_auc_tpr(train_threshold,train_fpr,train_tpr)
    result_for_treshold(title + ' - Treshold Best AUC TPR', best_t, y_train, y_train_predict, y_test, y_test_predict)

    best_t= best_threshold_auc_fpr(train_threshold,train_fpr,train_tpr)
    result_for_treshold(title + ' - Treshold Best AUC FPR', best_t, y_train, y_train_predict, y_test, y_test_predict)

    # Precision Recall Curve

    train_precision,train_recall,train_threshold=precision_recall_curve(y_train,y_train_predict)
    test_precision,test_recall,test_threshold=precision_recall_curve(y_test,y_test_predict)
    plt.plot(train_precision,train_recall,label='Train')
    plt.plot(test_precision,test_recall,label='Test')
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - ' + title)
    plt.grid()
    plt.show()

    train_score = average_precision_score(y_train,y_train_predict)
    test_score = average_precision_score(y_train,y_train_predict)
    print("Train average precision Score",train_score)
    print("Test average precision Score",test_score)

    best_t= 0.5
    result_for_treshold(title + ' - F1 - Treshold 0.5', best_t, y_train, y_train_predict, y_test, y_test_predict)

    best_t= best_threshold_f1score(train_threshold,train_precision,train_recall)
    result_for_treshold(title + ' - Treshold Best F1', best_t, y_train, y_train_predict, y_test, y_test_predict)

    return train_auc,test_auc

def confusion_matrix_heatmap(title, y_real, y_predict):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    matrix=confusion_matrix(y_real,y_predict)
    sns.heatmap(matrix,annot=True,fmt='d')
    plt.title("Confusion Matrix  - " + title)
    plt.show(ax)
    print()

def result_for_treshold(title, threshold, y_real, y_prob, y_real_test, y_prob_test):

    print(threshold)
    th = " (TH = " + str(np.round(threshold,3)) + ")"
    y_predict=prediction_(y_prob,threshold)
    confusion_matrix_heatmap('Train -' + title + th, y_real, y_predict)
    y_predict_test=prediction_(y_prob_test,threshold)
    confusion_matrix_heatmap('Test -' + title + th, y_real_test, y_predict_test)
    print()
    calculate_scores('Train - ' + title + th, y_real, y_predict)
    print()
    calculate_scores('Test - ' + title + th, y_real_test, y_predict_test)
    print()

