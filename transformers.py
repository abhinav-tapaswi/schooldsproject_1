from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
class _SymtomsCleaner(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform (self,X,y=None):
        lis=[]
        for i in list(X.values.tolist()):
            l=[]
            for j in i:
                if(j!=0):
                    l.append(j)
                else:
                    continue
            lis.append(l)
        return lis

class _MyMlb(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self, X, y=None):
        return MultiLabelBinarizer().fit_transform(X)
class MyPipe(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self, X , y=None):
        pipe= Pipeline([
            ('Cleaner', _SymtomsCleaner()),
            ('MLB', _MyMlb()), 
        ])
        return pipe.fit_transform(X)