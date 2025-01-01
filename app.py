from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment
from io import BytesIO

app = Flask(__name__)

# Define path to save plots
PLOT_PATH = 'static/plots/'

# Allowed audio file formats
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to convert mp3 to wav
def mp3_to_wav(mp3_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    wav_file_path = mp3_file_path.replace('.mp3', '.wav')
    audio.export(wav_file_path, format='wav')
    return wav_file_path

# Function to generate the waveform and FFT plots
def generate_plots(file_path):
    # Convert mp3 to wav if necessary
    if file_path.endswith('.mp3'):
        wav_path = mp3_to_wav(file_path)
    elif file_path.endswith('.wav'):
        wav_path = file_path

    # Read the audio file
    sample_rate, data = wavfile.read(wav_path)

    # Normalize data if stereo (convert to mono)
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Time axis for waveform
    time = np.linspace(0, len(data) / sample_rate, num=len(data))

    # Plot the waveform (time-domain)
    plt.figure(figsize=(10, 6))
    plt.plot(time, data, color='blue')
    plt.title('Waveform (Time-domain)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Save the waveform plot
    waveform_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_waveform.png"
    waveform_path = os.path.join(PLOT_PATH, waveform_filename)
    plt.savefig(waveform_path)
    plt.close()

    # Perform FFT analysis (frequency-domain)
    N = len(data)
    fft_data = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(N, 1/sample_rate)

    # Only keep the positive frequencies
    fft_data = np.abs(fft_data[:N//2])
    fft_freq = fft_freq[:N//2]

    # Plot the FFT result
    plt.figure(figsize=(10, 6))
    plt.plot(fft_freq, fft_data, color='red')
    plt.title('FFT Analysis (Frequency-domain)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

    # Save the FFT plot
    fft_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_fft.png"
    fft_path = os.path.join(PLOT_PATH, fft_filename)
    plt.savefig(fft_path)
    plt.close()

    return waveform_filename, fft_filename

# Route for the home page
@app.route('/')
def index():
    return render_template('result.html')

# Route for file upload and analysis
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file and allowed_file(file.filename):
        # Save the uploaded file
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Generate both waveform and FFT plots
        waveform_filename, fft_filename = generate_plots(file_path)

        # Return the filenames to the webpage for displaying the plots
        return render_template('result.html', waveform_filename=waveform_filename, fft_filename=fft_filename)

    return 'File format not allowed, please upload a .wav or .mp3 file'

# Route to serve the plot images
@app.route('/static/plots/<filename>')
def send_plot(filename):
    return send_from_directory(PLOT_PATH, filename)

if __name__ == '__main__':
    # Make sure the directories exist
    os.makedirs(PLOT_PATH, exist_ok=True)
    os.makedirs('uploads', exist_ok=True)

    app.run(debug=True)
