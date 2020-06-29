# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("C:/Users/SAI VAMSHI/Desktop/Analytics_Vidhya/HR")

df = pd.read_csv("train_LZdllcl.csv")
test_df = pd.read_csv("test_2umaH9m.csv")
df.head()

df.isnull().sum()

cat_var = ["department","region","education","gender","recruitment_channel"]
numeric_variables = ["no_of_trainings","age","previous_year_rating",
                     "length_of_service","KPIs_met >80%","awards_won?",
                     "avg_training_score"]

def imp_education(df1):
    df2 = df1.copy()
    df2.loc[:,"education"] = df2.loc[:,"education"].fillna("None")
    return df2

get_numeric_data = FunctionTransformer(lambda x: x[numeric_variables], validate=False)
get_cat_data = FunctionTransformer(lambda x: x[cat_var], validate=False)
impute_education = FunctionTransformer(imp_education)
create_dummies = FunctionTransformer(lambda x:pd.get_dummies(x,drop_first=True))

prepro_pipeline = FeatureUnion(
    transformer_list = [
        ('numeric_features',Pipeline([
            ('selector',get_numeric_data),
            ('imputer',SimpleImputer(strategy="most_frequent"))
            ])),
        ('text_features',Pipeline([
            ('selector_cat',get_cat_data),
            ('imputer_cat',impute_education),
            ('get_dummies',create_dummies)
            ]))
        ])

# df_prepro = prepro_pipeline.fit_transform(df)

pl = Pipeline([
    ('union',prepro_pipeline),
    ('stdize',StandardScaler()),
    ('clf',LogisticRegression(solver="liblinear"))
    ])

X = df.drop(["employee_id","is_promoted"],axis=1)
y = df["is_promoted"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1234,stratify=y)

parameters = {'clf__C':[10,30,50,70,100],
              'clf__penalty':["l2","l1"]
    }

cv = GridSearchCV(pl,parameters,scoring="f1",verbose=True)
cv.fit(X_train,y_train)
cv.best_params_
cv.best_score_
cv.best_estimator_.score(X_train,y_train)
cv.best_estimator_.score(X_test,y_test)
y_pred = cv.predict(X_test)
y_prob = cv.predict_proba(X_test)[:,1]
threshold = list(np.arange(0.1,1,0.02))
f1_list = []
for t in threshold:
    y_pred_newt = [1 if i > t else 0 for i in y_prob]
    f1_list.append(f1_score(y_test,y_pred_newt))
plt.plot(threshold,f1_list)    

threshold[np.argmax(f1_list)]

f1_score(y_test,y_pred)


test_prob = cv.best_estimator_.predict_proba(test_df)[:,1]
test_pred = [1 if i > 0.26 else 0 for i in test_prob]
test_sub = pd.DataFrame({"employee_id":test_df["employee_id"],
                        "is_promoted":test_pred})

test_sub.to_csv("logreg_v3.csv",index=False)
