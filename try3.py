import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from sklearn import metrics
from PIL import Image, ImageTk
import sounddevice as sd
import soundfile as sf
import librosa
import pandas as pd
import imutils

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def VoiceModel(json_file, weights_file):
    with open(json_file, "r") as file:
        model_json = file.read()
    voice_model = model_from_json(model_json)
    voice_model.load_weights(weights_file)
    voice_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return voice_model



class EmotionDetector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('800x600')
        self.root.title('Emotion Detector')
        self.root.configure(background='#CDCDCD')

        self.label1 = Label(self.root, background='#CDCDCD', font=('arial', 15, 'bold'))
        self.sign_image = Label(self.root)
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = FacialExpressionModel("model_a1.json", "model_weights1.h5")
        self.EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

        self.cap = cv2.VideoCapture(0)  

        self.upload_button = Button(self.root, text="Start Emotion Detection", command=self.start_detection, padx=10, pady=5)
        self.upload_button.configure(background="#364156", foreground="white", font=("arial", 20, "bold"))
        self.upload_button.pack(side='bottom', pady=50)

        self.sign_image.pack(side='bottom', expand='True')
        self.label1.pack(side='bottom', expand='True')
        self.heading = Label(self.root, text="Emotion Detector", pady=20, font=("arial", 25, "bold"))
        self.heading.configure(background="#CDCDCD", foreground="#364156")
        self.heading.pack()

    def record_audio(self, duration):
        fs = 44100  
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()

        temp_audio_file = "temp_audio.wav"
        sf.write(temp_audio_file, recording, fs)

        return temp_audio_file
    
    def predict_audio_emotion(self, audio_file):
        voice_model = VoiceModel("model_voice.json", "Emotion_Voice_Model.h5")

        audio_data, sample_rate = librosa.load(audio_file,res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13),axis=0)

        if len(mfccs) < 216:
          mfccs = np.pad(mfccs, (0, 216 - len(mfccs)), mode='constant')
        elif len(mfccs) > 216:
          mfccs = mfccs[:216]


        featurelive = mfccs
        livedf2 = featurelive
        livedf2= pd.DataFrame(data=livedf2)
        livedf2 = livedf2.stack().to_frame().T
        twodim= np.expand_dims(livedf2, axis=2)
        livepreds = voice_model.predict(twodim,batch_size=32,verbose=1)
        livepreds1=livepreds.argmax(axis=1)

        EMOTIONS_LIST_VOICE = ["Calm","Calm","Happy","Happy","Sad","Sad","Angry","Angry","Fearful","Fearful"]

        predicted_emotion = [EMOTIONS_LIST_VOICE[i] for i in livepreds1]

        return predicted_emotion


    def start_detection(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = imutils.resize(frame, width=450)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)

            for (x, y, w, h) in faces:
                roi = gray_frame[y:y+h, x:x+w]
                roi = cv2.resize(roi, (48, 48))
                pred_visual = self.EMOTIONS_LIST[np.argmax(self.model.predict(roi[np.newaxis, :, :, np.newaxis]))]

            
                temp_audio_file = self.record_audio(2.5)
                
                pred_audio = self.predict_audio_emotion(temp_audio_file)

                combined_pred = "Visual: {}, Audio: {}".format(pred_visual, pred_audio)

                cv2.putText(frame, combined_pred, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, combined_pred, (10,325),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


                print("Predicted Emotion is {}".format(combined_pred))


            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = EmotionDetector()
    app.run()
