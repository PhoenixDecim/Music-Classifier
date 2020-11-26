from flask import Flask, request, render_template,jsonify,flash, redirect, url_for, session
from werkzeug.utils import secure_filename
#from flask.ext.session import Session
import os
UPLOAD_FOLDER = "G:\\Music Classifier\\MC\\audio"
app = Flask(__name__)
#sess = Session()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def findgenre(file):
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from joblib import dump, load
    import librosa
    from sklearn.preprocessing import StandardScaler
    import statistics
    rf = load('rfmodel.joblib')
    genrel = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz','metal', 'pop', 'reggae', 'rock']
    y, sr = librosa.load(file,mono=True)
    s = y.shape[0]//661794
    if(s>=1):
        filen = np.array_split(y, s)
    else:
        filen=[y]
    n=[]
    for i in filen:
        chroma_stft = librosa.feature.chroma_stft(y=i, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=i, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=i, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=i, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(i)
        rmse = librosa.feature.rms(y=i)
        mfcc = librosa.feature.mfcc(y=i, sr=sr)
        to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        n.append(to_append.split())
    sn = np.asarray(n,dtype = float)
    genre = genrel[statistics.mode(rf.predict(sn))]
    return genre.upper()
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/join',methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        print(file)
        audiopath = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(file.filename))
        file.save(audiopath)
        out = {"output": findgenre(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))}
        os.remove(audiopath)
        return jsonify(out)
    #out = {"output": findgenre('G:\\Music Classifier\\MC\\audio\\song.mp3')}
    

from flask import send_from_directory
if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.debug = True
    app.run()