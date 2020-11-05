#--------Libraries--------#
import numpy 
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model
import pandas
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
#--------coding-----------#   
# read the file
path = 'E:\\Python\\heart.csv'
readFile = pandas.read_csv(path)
x = readFile[['age','anaemia',
              'creatinine_phosphokinase','diabetes',
              'ejection_fraction','high_blood_pressure','platelets',
              'serum_creatinine','serum_sodium',
              'sex','smoking','time']]
y = readFile['DEATH_EVENT']
##################################################################################
reg = linear_model.LinearRegression()
fitReg = reg.fit(x, y)
##########################################
predictDeath = fitReg.predict([[150,0,168,1,38,1,300000,1.1,137,1,0,11]])
print(predictDeath)
#########################################################################










