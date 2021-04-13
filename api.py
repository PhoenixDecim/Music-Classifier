from __future__ import unicode_literals
from flask import Flask, request, render_template,jsonify,flash, redirect, url_for, session
from werkzeug.utils import secure_filename
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import keras
import youtube_dl
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import librosa
import soundfile as sf
from sklearn.preprocessing import StandardScaler
import statistics
#from flask.ext.session import Session
import os
UPLOAD_FOLDER = os.getcwd()+"\\templates\\audio"
app = Flask(__name__)
#sess = Session()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
gauth = GoogleAuth()
#gauth.LocalWebserverAuth()
gauth.LoadCredentialsFile("mycreds.txt")
drive = GoogleDrive(gauth)
def findgenre(file):
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
def bass(path):
    cnn=keras.models.load_model("bass1d_model.h5",custom_objects={'LeakyReLU': keras.layers.advanced_activations.LeakyReLU})
    x, sr = librosa.load(path)
    specm = librosa.stft(x,n_fft=2048)
    frames=specm.shape[1]
    newspec=specm.reshape(frames,1025,1)
    basswo=cnn.predict(newspec,batch_size=12)
    basswo=basswo.reshape(1025,frames)
    bi=(np.less(basswo,np.percentile(basswo,75)))
    sovox=specm*bi
    svox = librosa.istft(sovox)
    n = len(svox)
    n_fft = 2048
    svox = librosa.util.fix_length(svox, n + n_fft // 2)
    ap='F:/Music-Classifier/templates/audio/basswo.wav'
    sf.write(ap, svox, 22050, subtype='PCM_16')
    file2 = drive.CreateFile({'title': 'basswo.wav','parents': [{'id': '1amw0Ag2Lq2E2skfhDm7NV88Kgir63GpH'}]})  # Create GoogleDriveFile instance with title 'Hello.txt'.
    file2.SetContentFile(ap) # Set content of the file from given string.
    file2.Upload()
    file2.InsertPermission({
                'type': 'anyone',
                'value': 'anyone',
                'role': 'reader'})
    return(file2['id'])
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/join',methods=['GET', 'POST'])
def upload_file():
    file_list = drive.ListFile({'q':"'1amw0Ag2Lq2E2skfhDm7NV88Kgir63GpH'  in parents"}).GetList()
    for x in file_list:
        x.Delete()
    d=UPLOAD_FOLDER
    filesToRemove = [os.path.join(d,f) for f in os.listdir(d)]
    for f in filesToRemove:
        os.remove(f)
    if request.method == 'POST':
        print(request.form)
        if(request.form['category']=='0'):
            file = request.files['file']
            if file.filename == '':
                print('No selected file')
                return redirect(request.url)
            print(file)
            ext=file.filename.rsplit('.', 1)[1].lower()
            fil = 'song.'+ext
            audiopath = os.path.join(app.config['UPLOAD_FOLDER'],fil)
            file.save(audiopath)
            file1 = drive.CreateFile({'title': fil,'parents': [{'id': '1amw0Ag2Lq2E2skfhDm7NV88Kgir63GpH'}]})  # Create GoogleDriveFile instance with title 'Hello.txt'.
            file1.SetContentFile(audiopath) # Set content of the file from given string.
            file1.Upload()
            file1.InsertPermission({
                        'type': 'anyone',
                        'value': 'anyone',
                        'role': 'reader'})
            driveurl=file1['id']
            print(file1['id'])
        elif(request.form['category']=='1'):
            url=request.form['text']
            class MyLogger(object):
                def debug(self, msg):
                    pass
                def warning(self, msg):
                    pass
                def error(self, msg):
                    print(msg)

            def my_hook(d):
                if d['status'] == 'finished':
                    print('Done downloading, now converting ...')

            ydl_opts = {
                'outtmpl': UPLOAD_FOLDER+'/song.%(ext)s',
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'aac',
                    'preferredquality': '256',
                }],
                'logger': MyLogger(),
                'progress_hooks': [my_hook],
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            audiopath = UPLOAD_FOLDER+"/song.aac"
            fil='song.aac'
            file1 = drive.CreateFile({'title': fil,'parents': [{'id': '1amw0Ag2Lq2E2skfhDm7NV88Kgir63GpH'}]})
            file1.SetContentFile(audiopath)
            file1.Upload()
            file1.InsertPermission({
                        'type': 'anyone',
                        'value': 'anyone',
                        'role': 'reader'})
            driveurl=file1['id']
            print(file1['id'])
        basswo=bass(audiopath)
        out = {"output": findgenre(audiopath),"path":driveurl,"bass":basswo}
        return jsonify(out)
    #out = {"output": findgenre('F:\\Music Classifier\\MC\\audio\\song.mp3')}
    
from flask import send_from_directory
if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.debug = True
    app.run()