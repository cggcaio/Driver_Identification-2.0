from sklearn.cluster import KMeans
import pandas as pd
from scipy.spatial import distance
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

# Local Functions
from functions import normalize
from functions import build_df_final
#from functions import get_maneuvers
from functions import train_model_ocsvm
from functions import test_model_ocsvm
from functions import evaluating_result
from functions import train_model_if
from functions import test_model_if
from functions import split_data
from functions import clusters_of_maneuvers

# Utilizar somente quando quiser normalizar os dados novamente
#print('Normalizing data')
#original_data = pd.read_csv("https://raw.githubusercontent.com/cggcaio/Anomaly-Detection-for-Driver-Identification/master/Data_Bases/KIA_DB/Driving%20Data(KIA%20SOUL)_(150728-160714)_(10%20Drivers_A-J).csv")
#data_normsssalized = (normalize(original_data))

#data_normalized = pd.read_csv("https://raw.githubusercontent.com/cggcaio/Anomaly-Detection-for-Driver-Identification/master/Data_Bases/KIA_DB/data_normalized.csv")
data_normalized = pd.read_csv('data_normalized.csv')

# PARAMETERS
driver = ['A']
impostor = ['B'] 
window_size = [ 5 ]
selected_features = ['Intake_air_pressure','Engine_soacking_time', 'Long_Term_Fuel_Trim_Bank1', 'Torque_of_friction', 'Engine_coolant_temperature', 'Steering_wheel_speed']
method = 'OCSVM'



print("Building DF for Driver", driver, "with Window_Size", window_size)
data_final = build_df_final(data_normalized, driver, window_size, selected_features)

print("Building DF for Impostor", impostor, "with Window_Size", window_size)
data_impostor = build_df_final(data_normalized, impostor, window_size, selected_features )

print('Doing data split')
x_train, x_val = split_data(data_final)

print('Create clusters')
labels_train, centroid_train, x_train_class = clusters_of_maneuvers(x_train)

if(method=='IF'):
  print('Training IF')
  if_list = train_model_if(labels_train, centroid_train, x_train_class, x_val)

  print('Doing predictions IF')
  result = test_model_if(if_list, data_final, data_impostor, centroid_train)


elif(method=='OCSVM'):
  print('Training OCSVM')
  ocsvm_list = train_model_ocsvm(labels_train, centroid_train, x_train_class, x_val)

  print('Doing predictions OCSVM')
  result = test_model_ocsvm(ocsvm_list, data_final, data_impostor, centroid_train)

print('Evaluanting the results')
acc, quantidade_manobras = evaluating_result(result)

print('Acurácia: ', acc, '% MANOBRAS CLASSIFICADAS COMO NORMAIS')
print('Quantidade de manobras necessárias para detectar uma anomalia: ', quantidade_manobras)
