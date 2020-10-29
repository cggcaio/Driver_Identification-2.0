from sklearn import preprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from scipy.spatial import distance

# Receivess the DB and normalizes the data by column
def normalize(original_data):
  data = original_data.drop(columns=['Time(s)', 'Class', 'PathOrder'])
  scaler = preprocessing.MinMaxScaler()
  std_values = scaler.fit_transform(data)
  data_std = pd.DataFrame(data=std_values, columns=data.columns)
  df2 = original_data[['Time(s)','Class', 'PathOrder']].copy()
  data_normalized = data_std.join(df2)
  return data_normalized

# Receives DataFrame normalized, the drivers selected, the window size and the selected features
def build_df_final(data_normalized, drivers, block_sizes, selected_features):

  def create_column_names(block_size,selected_features):
    print(block_size)
    c_names = []
    for t in np.arange(block_size):
      for a in selected_features:
        c_names.append(a + '_s' + str(t))

    return c_names
  
  for driver in drivers:
    for b in (block_sizes):
      # Generate column names
      c_names = create_column_names(b,selected_features)

      # Create a data frame for driver and block_size = b
      data = pd.DataFrame(columns=c_names)

      # Select driver
      driver_df = data_normalized[data_normalized["Class"] == driver] 
      
      # Sweep driver records (1 record per second)
      for i in tqdm(np.arange(len(driver_df)-b)):
        row = []
        time_stamps = []

        for j in np.arange(b):
          # Get register i+j
          df_temp = driver_df.iloc[i+j]
          
          # Build row with selected features
          row = row + [df_temp[a] for a in selected_features]
          time_stamps.append(df_temp['Time(s)'])

        # Check time consistency
        df_temp = driver_df.iloc[i]
        if len(set(np.arange(df_temp['Time(s)'],df_temp['Time(s)']+b)).intersection(time_stamps) ) == len(time_stamps):
          # Add rows to dataframe if times are consistent
          #row = row +
          row_df = pd.DataFrame(np.array([row]),columns=c_names)
          data = pd.concat([data,row_df])

      #data.to_csv('driver_' + driver +'_block_'+ str(b) + 's')
  return data

def split_data(data_std):
  labels = np.ones((data_std.shape[0],1))
  x_train, x_val, y_train, y_val = train_test_split(data_std,  labels, test_size=0.2, random_state=42)
  return x_train, x_val

def clusters_of_maneuvers(x_train):
  x_train = pd.DataFrame(x_train)
  kmeans = KMeans(n_clusters=10, random_state=42)
  kmeans.fit(x_train)
  x_train['K-Class'] = kmeans.labels_
  labels_train = kmeans.labels_
  centroid_train = kmeans.cluster_centers_
  return labels_train, centroid_train, x_train

################## ISOLATION FOREST #####################
def train_model_if( labels_train, centroid_train, x_train_class, x_val):
  if_list = []
  for n in range(0, max(labels_train)+1):
    df_cluster = x_train_class[x_train_class['K-Class'] == n]
    model = IsolationForest(n_estimators=6, random_state=42).fit(x_train_class.drop(columns=['K-Class']))
    if_list.append({'cluster': n, 'driver': 'a', 'model': model, 'centroid': centroid_train[n], 'x_val': x_val })
  return if_list
def test_model_if(if_list, data_final, data_impostor, centroides):
  result = []
  
  for i in range(len(data_impostor)):   # Percorrer o DF de um motorista impostor
    menor = 1000
    for m in range(len(centroides)): # Percorrer entre as 10 manobras 
      dist = distance.euclidean(data_impostor.iloc[i], centroides[m]) # Calcular se uma manabora do motorista impostor se aproxima desse cluster N do motorista principal
      #print('Distancia ', m , 'valor ', dist)
      if (dist<menor):
        menor = dist
        manobra_correspondente = m
    result.append(if_list[manobra_correspondente]['model'].predict([data_impostor.iloc[i].values])[0])
  return result
#########################################################


################## ONE CLASS SVM ########################
def train_model_ocsvm( labels_train, centroid_train, x_train_class, x_val):
  ocsvm_list = []
  for n in range(0, max(labels_train)+1):
    df_cluster = x_train_class[x_train_class['K-Class'] == n]  
    model = OneClassSVM(kernel='sigmoid', nu=0.2).fit(x_train_class.drop(columns=['K-Class']))
    ocsvm_list.append({'cluster': n, 'driver': 'a', 'model': model, 'centroid': centroid_train[n], 'x_val': x_val })
  return ocsvm_list
def test_model_ocsvm(oscvm_list, data_final, data_impostor, centroides):
  result = []
  #data_impostor = x_val
  
  for i in range(len(data_impostor)):   # Percorrer o DF de um motorista impostor
    menor = 1000
    for m in range(len(centroides)): # Percorrer entre as 10 manobras 
      dist = distance.euclidean(data_impostor.iloc[i], centroides[m]) # Calcular se uma manabora do motorista impostor se aproxima desse cluster N do motorista principal
      #print('Distancia ', m , 'valor ', dist)
      if (dist<menor):
        menor = dist
        manobra_correspondente = m
    result.append(oscvm_list[manobra_correspondente]['model'].predict([data_impostor.iloc[i].values])[0])
  return result
#########################################################

def evaluating_result(result):
  score = 0
  quantidade_manobras = 0
  contador_positivos = 0
  for r in range(len(result)):
    if (result[r]==1):
      score = score + 1
      contador_positivos = contador_positivos + 1 
    
    if (result[r]==-1):
      if (contador_positivos>quantidade_manobras):
        quantidade_manobras = contador_positivos
        contador_positivos = 0  


  acc = (score / len(result))*100 # Acurácia - Porcentagem de manobras que foram classificadas como normais

  return acc, quantidade_manobras

if __name__ == '__main__':
  print("")