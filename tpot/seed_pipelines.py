import pandas as pd

from .dataset_describe import Dataset
from collections import OrderedDict
from sklearn.externals import joblib

from os import path

    
def get_metafeatures(df):
    
    select_metafeatures = [u'class_prob_max', u'class_prob_mean', u'class_prob_median',
       u'class_prob_min', u'class_prob_std', u'diversity_fraction',
       u'entropy_dependent', u'kurtosis_kurtosis', u'kurtosis_max',
       u'kurtosis_mean', u'kurtosis_median', u'kurtosis_min', u'kurtosis_skew',
       u'kurtosis_std', u'n_categorical', u'n_classes', u'n_columns',
       u'n_numerical', u'n_rows', u'pca_fraction_95',
       u'ratio_rowcol', u'skew_kurtosis', u'skew_max', u'skew_mean',
       u'skew_median', u'skew_min', u'skew_skew', u'skew_std']
    

    dataset = Dataset(df, prediction_type='classification', dependent_col = 'class')
   
    meta_features = OrderedDict()
    for i in dir(dataset):
        result = getattr(dataset, i)
        # print i
        if not i.startswith('__') and not i.startswith('_') and hasattr(result, '__call__'):
            if i in select_metafeatures:
#                 print i
                meta_features[i] = result()
    
    return pd.DataFrame(meta_features, index=[0])