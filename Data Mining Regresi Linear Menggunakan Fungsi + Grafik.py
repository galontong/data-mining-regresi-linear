import pandas as pd
import sklearn.linear_model as lm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Membaca Data
df = pd.read_excel('data\data_regresi_linear.xlsx','Sheet1')

# memanggil Fungsi
lr = lm.LinearRegression()

x = df.Luas_Tanah.values.reshape(-1,1)
y = df.Harga_dlm_juta.values.reshape(-1,1)

lr.fit(x,y)

# Output
print("[INFO]")
print('Intercept : ', lr.intercept_)
print('Coefisien : ', lr.coef_)
print("\n[PERSAMAAN]")
print('jadi persaman yg terbentuk adalah : Y = ', lr.intercept_ , ' + ', lr.coef_, 'X')
print("\n[PREDIKSI]")
print('prediksi untuk Luas tanah(X) = 1800  maka nilai Harga dlm juta(Y) = ', lr.predict([[1800]]))

print("\n[MANUAL]")
# Evaluasi predict
df['prediksi_harga_dlm_juta'] = lr.predict(x)

# Evaluasi Manual
df['SST']= np.square(df['Harga_dlm_juta'] - df['Harga_dlm_juta'].mean())
df['SSR']= np.square(df['prediksi_harga_dlm_juta'] - df['Harga_dlm_juta'].mean())
print('SSR=', df['SSR'].sum())
print('SST=', df['SST'].sum())
print('perhitungan scr manual R-square = ', df['SSR'].sum() / df['SST'].sum())


# Evaluasi Built in Function
print('R-Squared dengan fungsi : ', r2_score(df.Harga_dlm_juta, df.prediksi_harga_dlm_juta))
print('Mean Absolute Error : ', mean_absolute_error(df.Harga_dlm_juta, df.prediksi_harga_dlm_juta))
print('Root Mean Squared Error : ', np.sqrt(mean_squared_error(df.Harga_dlm_juta, df.prediksi_harga_dlm_juta)))

# Grafik
plt.scatter(x, y, color = 'black')
plt.plot(x, lr.predict(x), color = 'blue', linewidth = 3)
plt.title('Luas Tanah vs Harga dlm juta')
plt.ylabel('Harga_dlm_juta')
plt.xlabel('Luas_Tanah')
plt.show()

