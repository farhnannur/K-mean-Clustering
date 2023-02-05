from flask import Flask, render_template
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# load data
data = pd.read_csv('jumlah_tempat_hiburan1.csv')
data = data.fillna(data.mean())

# Menciptakan objek LabelEncoder
le = LabelEncoder()

# Fitting LabelEncoder pada kolom jenis_tempat_hiburan
data['jenis_tempat_hiburan'] = le.fit_transform(data['jenis_tempat_hiburan'].values)

# encode jenis_tempat_hiburan
data['jenis_tempat_hiburan_encoded'] = data['jenis_tempat_hiburan'].map({'Kafe':0, 'Restoran':1, 'Bar':2, 'Tempat Hiburan Malam':3})
data['jenis_tempat_hiburan_encoded'] = data['jenis_tempat_hiburan_encoded'].astype(float)

# cluster data
model = KMeans(n_clusters=4)
model.fit(data[['jenis_tempat_hiburan', 'tahun']])
prediction = model.predict(data[['jenis_tempat_hiburan', 'tahun']])
data['cluster'] = prediction

app = Flask(__name__)

@app.route("/")
def index():
    # plot data
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='jenis_tempat_hiburan', y='tahun', hue=prediction, palette='viridis', data=data)
    plt.xlabel('Jenis Tempat Hiburan ')
    plt.ylabel('Tahun')
    sns.lineplot(x='jenis_tempat_hiburan', y='tahun', hue='jumlah_tempat_hiburan', color='red', ci=None, data=data)
    plt.savefig('static/plot.png')
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run()
