from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd

def NaiveBayes(csv_info):
    x = csv_info.iloc[:, 0:csv_info.shape[1]-1].values
    y = csv_info.iloc[:, csv_info.shape[1]-1].values
        
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)
    
    gnb = GaussianNB()
    ypred = gnb.fit(xtrain, ytrain).predict(xtest)
    print("Porcentaje de acierto:", (xtest.shape[0]-(ytest != ypred).sum())/xtest.shape[0] *100)

csv_info = pd.read_csv("SwedenInsurance.csv")
print("Suecos")
#NaiveBayes(csv_info)
csv_info = pd.read_csv("Diabetes.csv")
print("Diabetes")
NaiveBayes(csv_info)
csv_info = pd.read_csv("WineQuality.csv")
print("Vino")
NaiveBayes(csv_info)