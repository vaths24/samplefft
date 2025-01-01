import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

app = Flask(__name__)

# Create the uploads folder if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio_file' not in request.files:
        return "No file part"

    file = request.files['audio_file']

    if file.filename == '':
        return "No selected file"

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Load audio file using librosa
    y, sr = librosa.load(file_path, sr=None)

    # Generate time vector for plotting the sine wave
    duration = len(y) / sr
    time = np.linspace(0, duration, len(y))

    # Plot the time-domain waveform (sine wave)
    plt.figure(figsize=(10, 6))
    plt.plot(time, y)
    plt.title('Time-Domain Audio Signal (Sine Wave)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')

    # Save plot as an image
    plot_path = os.path.join('static', 'plot.png')
    plt.savefig(plot_path)
    plt.close()

    # Return the plot to display
    return render_template('index.html', plot_url='static/plot.png')


if __name__ == '__main__':
    app.run(debug=True)
