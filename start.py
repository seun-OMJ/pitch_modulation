import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from PyQt5 import QtWidgets
import sys
from PyQt5 import QtWidgets, QtCore, QtGui

def detect_pitch(audio, sr):
    pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
    pitch = []
    for t in range(pitches.shape[1]):
        index =magnitudes[: , t].argmax()
        pitch.append(int(round(pitches[index, t])))
    return np.array(pitch)

def correct_pitch(pitches, target_pitch):
    non_zero_pitches = pitches[pitches > 0]
    if len(non_zero_pitches) == 0:
        return 0
    average_pitch = np.mean(non_zero_pitches)
    n_steps = np.log2(target_pitch / average_pitch) * 12
    return n_steps

def convert_to_wav(input_audio_path):
    audio = AudioSegment.from_file(input_audio_path)
    wav_path = input_audio_path.rsplit('.', 1)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

def PitchShift_audio(input_audio_path, output_audio_path, target_pitch):
    if input_audio_path.endswith('.mp3'):
        input_audio_path = convert_to_wav(input_audio_path)
    
    audio, sr = librosa.load(input_audio_path)
    pitches = detect_pitch(audio, sr)
    n_steps = correct_pitch(pitches, target_pitch)
    if n_steps != 0:
        corrected_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    else:
        corrected_audio = audio
    sf.write(output_audio_path, corrected_audio, sr)

class PitchShiftApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('PitchShift App')
        self.layout = QtWidgets.QVBoxLayout()
        
        self.introLabel = QtWidgets.QLabel("-12 (220 Hz): 220.00 Hz \n-11: 233.08 Hz \n-10: 246.94 Hz \n-9: 261.63 Hz \n-8: 277.18 Hz \n-7: 293.66 Hz \n-6: 311.13 Hz \n-5: 329.63 Hz \n-4: 349.23 Hz \n-3: 369.99 Hz \n-2: 391.99 Hz \n-1: 415.30 Hz \n0 (440 Hz): 440.00 Hz \n+1: 466.16 Hz \n+2: 493.88 Hz \n+3: 523.25 Hz \n+4: 554.37 Hz \n+5: 587.33 Hz \n+6: 622.25 Hz \n+7: 659.26 Hz \n+8: 698.46 Hz \n+9: 739.99 Hz \n+10: 783.99 Hz \n+11: 830.61 Hz \n+12 (880 Hz): 880.00 Hz")
        self.layout.addWidget(self.introLabel)
        
        self.openButton = QtWidgets.QPushButton('Open Audio File')
        self.openButton.clicked.connect(self.openFile)
        self.layout.addWidget(self.openButton)
        
        self.pitchSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pitchSlider.setMinimum(220)  # Setting minimum pitch value
        self.pitchSlider.setMaximum(880)  # Setting maximum pitch value
        self.pitchSlider.setValue(440)    # Default pitch value
        self.pitchSlider.valueChanged.connect(self.updatePitchLabel)
        self.layout.addWidget(self.pitchSlider)
        
        self.pitchLabel = QtWidgets.QLabel(f'Target Pitch: {self.pitchSlider.value()} Hz')
        self.layout.addWidget(self.pitchLabel)
        
        self.saveButton = QtWidgets.QPushButton('Save Pitch-Shift Audio')
        self.saveButton.clicked.connect(self.saveFile)
        self.layout.addWidget(self.saveButton)
        
        self.setLayout(self.layout)

    def openFile(self):
        options = QtWidgets.QFileDialog.Options()
        self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.mp3; *.wav);;All Files (*)", options=options)
        if self.fileName:
            print(f"Loaded file: {self.fileName}")

    def saveFile(self):
        if not hasattr(self, 'fileName') or not self.fileName:
            return
        
        options = QtWidgets.QFileDialog.Options()
        saveFileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Pitch-Shift Audio", "", "Audio Files (*.wav);;All Files (*)", options=options)
        if saveFileName:
            PitchShift_audio(self.fileName, saveFileName, self.pitchSlider.value())
            print(f"Saved Pitch-Shift file: {saveFileName}")
            
    def updatePitchLabel(self):
        self.pitchLabel.setText(f'Target Pitch: {self.pitchSlider.value()} Hz')

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = PitchShiftApp()
    ex.show()
    sys.exit(app.exec_())
