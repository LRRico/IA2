from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score 
import pandas as pd

def LogicR(csv_info):
    x = csv_info.iloc[:, 0:csv_info.shape[1]-1].values
    y = csv_info.iloc[:, csv_info.shape[1]-1].values
            
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)
    
    sc_x = StandardScaler() 
    xtrain = sc_x.fit_transform(xtrain) 
    xtest = sc_x.transform(xtest) 
    
    classifier = LogisticRegression(random_state = 0) 
    classifier.fit(xtrain, ytrain)
    
    y_pred = classifier.predict(xtest)
    
    
    print ("Porcentaje de Exito : ", accuracy_score(ytest, y_pred)*100,"%")
    
csv_info = pd.read_csv("SwedenInsurance.csv")
print("Suecos")
#LogicR(csv_info)
csv_info = pd.read_csv("Diabetes.csv")
print("Diabetes")
LogicR(csv_info)
csv_info = pd.read_csv("WineQuality.csv")
print("Vino")
LogicR(csv_info)