import os

# Write to Excel, Graphs, Numpy
import xlsxwriter as xls
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pathlib
import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# PyDub for cutting audio into chunks
from pydub import AudioSegment
from pydub.utils import make_chunks
# Transform of the wav
import scipy
from scipy.io import wavfile
#Unfreeze
from PyQt5.QtWidgets import QApplication

# Characteristics
# Frequency +
# Amplitude +
# Phase +
# Pitch +
# Harmonics\Tembre +
# Bandwidth +

# Ignore
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (9, 7)

# Cut audio to chunks -----------------------------------------------------------------------------------------------

# Get start and end len vals from audio file in sec
def getFileInfo(file_name):
    audio = AudioSegment.from_file(file_name, "wav")

    return audio[0], audio[-1]

# Cutting intial audio if required
# (path to file, save path, chunk start time, chunk end time, link to console output)
def cutAudioFile(file_name, savePath, startTime, endTime, consoleOutput):

    before, sep, after = file_name.rpartition("/")

    audio = AudioSegment.from_file(file_name, "wav")
    cut_audio = audio[startTime:endTime]

    chunk_name = after.strip('.wav') + "_Cut.wav"
    file = savePath + "/" + chunk_name

    cut_audio.export(file, format="wav")

    consoleOutput.insertPlainText("Збереження результату: %s \n" % file)

# Splitting initial audio
# (path to file, chunk length, save path, link to progress bar, link to console output)
def splitAudioFile(file_name, chunkLength, savePath, pBar, consoleOutput):

    myaudio = AudioSegment.from_file(file_name, "wav")
    chunks = make_chunks(myaudio, chunkLength) # pydub calculates in millisec (1000 = 1 sec)

    before, sep, after = file_name.rpartition("/")

    pBar.setMaximum(len(chunks))
    c = 1

    for i, chunk in enumerate(chunks):
        if len(chunk) == chunkLength:
            chunk_name = after.strip('.wav') + "_{0}.wav".format(i)
            file = savePath + "/" + chunk_name
            # Export
            chunk.export(file, format="wav")
            # Visual output
            consoleOutput.insertPlainText("Exporting: %s \n" % file)
            # Auto scroll
            consoleOutput.verticalScrollBar().setValue(consoleOutput.verticalScrollBar().maximum())
            QApplication.processEvents()
            pBar.setValue(c+1)
            c += 1
        else:
            break

# Processing audio file into numerical data
# (Path to file, frequency range start, end, column position pointer, ally pointer,
# worksheets for saving data, link to console output)
def processAudio(file_name, rangeStart, rangeEnd, pos, allyPointer, worksheetAmp, worksheetPh, consoleOutput):

    signal, sampFreq = librosa.load(
        file_name,
        sr=None,
        mono=True
    )
    consoleOutput.insertPlainText(f"Файл: {file_name}\n")

    # Signal lenght
    length_in_s = signal.shape[0] / sampFreq
    print("Sighnal lenght: ", length_in_s, "s")

    # FFT using librosa
    fft_spectrum = librosa.stft(signal, n_fft=len(signal), hop_length=len(signal) + 1)
    # Convert 2D STFT result to 1D
    fft_spectrum = fft_spectrum[:, 0]
    # Frequencies
    freq = librosa.fft_frequencies(sr=sampFreq, n_fft=len(signal))

    # Phase and magnitude
    phase = np.angle(fft_spectrum)
    magnitude = np.abs(fft_spectrum)
    fft_spectrum_abs = magnitude

    # Empty lists and dict for saving results
    frequency = []
    amplitude = []

    dataDict = {}
    phaseDict = {}
    magnitudeDict = {}

    for k in range(rangeStart, rangeEnd):
        dataDict[k] = ''

    for k in range(rangeStart, rangeEnd):
        phaseDict[k] = ''

    # Procession
    for i, f in enumerate(fft_spectrum_abs):
        if i < rangeEnd:  # looking for frequency below 350 (rangeEnd)

            frequency.append(format(np.round(freq[i], 1))) # Частота
            amplitude.append(format(f))          # Амплітуда

            for key in dataDict.keys():
                if key == int(float((format(np.round(freq[i], 1))))):
                    dataDict[key] = format(f) # Амплітуда
                    phaseDict[key] = format(phase[i])  # Фаза

            # consoleOutput.insertPlainText(
            #     "frequency = {} Hz with amplitude {} \n".format(np.round(freq[i], 1), format(f)))
            # consoleOutput.insertPlainText(
            #     "frequency = {} Hz with phase {} \n".format(np.round(freq[i], 1), phase[i]))

            QApplication.processEvents()
            # Auto scroll
            consoleOutput.verticalScrollBar().setValue(consoleOutput.verticalScrollBar().maximum())

            # print('frequency = {} Hz with amplitude {} '.format(np.round(freq[i], 1), format(f)))
            # print('frequency = {} Hz with phase {} '.format(np.round(freq[i], 1), phase[i]))

    # Pitch
    peakIdx = np.argmax(amplitude)
    pitchFreq = frequency[peakIdx]

    print(f"Estimated pitch: {pitchFreq} Hz")

    note, octave, midi = freqToNoteName(pitchFreq)
    print(f"{pitchFreq} Hz -> {note}{octave} (MIDI {midi})")

    # Magnitude
    print(magnitude)
    print(f"Absolute magnitude: {abs(magnitude)}")

    frequency = np.array([float(x) for x in frequency])
    amplitude = np.array([float(x) for x in amplitude])

    # Get timbre descriptors
    descriptors = timbreDescriptors(frequency, amplitude)
    print("\nTimbre Descriptors:")
    for k, v in descriptors.items():
        print(f"{k}: {v:.2f}")

    exportData(file_name, dataDict, phaseDict, pitchFreq, descriptors, pos, allyPointer, worksheetAmp, worksheetPh)

def processAudioNpy(file_name, rangeStart, rangeEnd, pos, allyPointer, consoleOutput):

    signal, sr = librosa.load(
        file_name,
        sr=None,
        mono=True
    )

    fft_spectrum = librosa.stft(
        signal,
        n_fft=len(signal),
        hop_length=len(signal) + 1
    )

    fft_spectrum = fft_spectrum[:, 0]

    freq = librosa.fft_frequencies(
        sr=sr,
        n_fft=len(signal)
    )

    phase = np.angle(fft_spectrum)
    magnitude = np.abs(fft_spectrum)

    amp_features = []
    # phase_features = []

    print(len(magnitude))
    print(rangeEnd)

    for i in range(min(rangeEnd, len(magnitude))):
        amp_features.append(magnitude[i])
        # phase_features.append(phase[i])

    frequency = freq[:rangeEnd]
    amplitude = magnitude[:rangeEnd]

    descriptors = timbreDescriptors(
        frequency,
        amplitude
    )

    peakIdx = np.argmax(amplitude)
    pitchFreq = frequency[peakIdx]

    descriptor_values = list(descriptors.values())

    feature_vector = np.concatenate([
        amp_features,
        # phase_features,
        [pitchFreq],
        descriptor_values
    ])

    return feature_vector, allyPointer


# Output into Excel ------------------------------------------------------------------------------------------------

# Exporting numerical data into Excel file
# (Path to file, ampData dictionary, phaseData dictionary, column position pointer, ally pointer,
# worksheets for saving data)
def exportData (fileName, dataDict, phaseDict, pitch, timbre, position, allyPointer, worksheetAmp, worksheetPh):

    pos = position

    col = 1

    # Partition must return 3 arguments
    before, sep, after = fileName.rpartition('/')
    fileName = after

    worksheetAmp.write(0, 0, 'Файл')
    worksheetAmp.write(0, 1, 'Свій/Чужий')
    worksheetPh.write(0, 0, 'Файл')
    worksheetPh.write(0, 1, 'Свій/Чужий')

    worksheetAmp.write(pos , 0 , fileName )
    worksheetPh.write(pos , 0 , fileName )

    worksheetAmp.write(pos , 1, allyPointer)
    worksheetPh.write(pos , 1, allyPointer)

    # Getting rid of empty columns

    formattedAmpDict = {}
    formattedPhaseDict = {}

    for key, value in dataDict.items():
        if value != '':
            formattedAmpDict[key] = value

    for key, value in phaseDict.items():
        if value != '':
            formattedPhaseDict[key] = value

    # Add amp page

    # Top row with chars names
    for key in formattedAmpDict.keys():
        worksheetAmp.write(0 , col + 1 , '%d Гц Амп' %key)
        col +=1

    worksheetAmp.write(0, col + 1, 'Висота')
    col += 1
    worksheetAmp.write(0, col + 1, 'Висота (Ноти)')
    col += 1

    for key in timbre.keys():
        worksheetAmp.write(0, col + 1, '%s' %key)
        col += 1

    col = 1

    # Data row with chars names
    for key, value in formattedAmpDict.items():
        worksheetAmp.write(pos, col + 1, value)
        col += 1

    worksheetAmp.write(pos, col + 1, pitch)
    col += 1
    note, octave, midi = freqToNoteName(pitch)
    worksheetAmp.write(pos, col + 1, f'{note}{octave} (MIDI {midi})')
    col += 1

    for key, value in timbre.items():
        worksheetAmp.write(pos, col + 1, '%f' % value)
        col += 1

    col = 1

    # Add phase page
    # Top row with chars names
    for key in formattedPhaseDict.keys():
        worksheetPh.write(0 , col + 1 , '%d Гц Фаза' %key )
        col +=1

    worksheetPh.write(0, col + 1, 'Висота')
    col += 1
    worksheetPh.write(0, col + 1, 'Висота (Ноти)')
    col += 1

    for key in timbre.keys():
        worksheetPh.write(0, col + 1, '%s' % key)
        col += 1

    col = 1

    # Data row with chars names
    for key, value in formattedPhaseDict.items():
        worksheetPh.write(pos, col + 1, value)
        col += 1

    worksheetPh.write(pos, col + 1, pitch)
    col += 1
    note, octave, midi = freqToNoteName(pitch)
    worksheetPh.write(pos, col + 1, f'{note}{octave} (MIDI {midi})')
    col += 1

    for key, value in timbre.items():
        worksheetPh.write(pos, col + 1, '%f' % value)
        col += 1

# Main processing ----------------------------------------------------------------------------------------------------

# Main procession method
# (Path to files, filesList, ally pointer, frequency range start, end,
# data save path, link to progress bar, link to console output)
def processFiles(filesFolder, filesList, allyPointer, rangeStart, rangeEnd, savePath, pBar, consoleOutput):

    print(savePath)

    workbook = xls.Workbook(savePath + filesList[0].strip("_0.vaw") + ".xlsx", {'nan_inf_to_errors': True})
    worksheetAmp = workbook.add_worksheet('Амп')
    worksheetPh = workbook.add_worksheet('Фаза')

    pBar.setMaximum(len(filesList))

    for i in range(len(filesList)):
        fileName = filesFolder + filesList[i]
        print(fileName)
        if fileName.endswith(".wav"):
            processAudio(fileName, rangeStart, rangeEnd, i + 1, allyPointer, worksheetAmp, worksheetPh, consoleOutput)
            pBar.setValue(i+1)
        else:
            pass

    workbook.close()

def processFilesNpy(filesFolder, filesList, allyPointer, rangeStart, rangeEnd, savePath, pBar, consoleOutput):

    pBar.setMaximum(len(filesList))

    X = []
    Y = []

    for i in range(len(filesList)):
        fileName = filesFolder + filesList[i]
        print(fileName)
        if fileName.endswith(".wav"):
            consoleOutput.insertPlainText(f"{fileName}\n")
            x, y = processAudioNpy(fileName, rangeStart, rangeEnd, i + 1, allyPointer, consoleOutput)
            X.append(x)
            Y.append(y)
            pBar.setValue(i + 1)
            QApplication.processEvents()
        else:
            pass

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y)

    print(X)
    print(Y)
    print("Npy processed")

    cleanPath = filesFolder.rstrip("/")
    name = cleanPath.rsplit("/", 1)[-1]

    np.save(f"{savePath}/X_{name}.npy", X)
    np.save(f"{savePath}/y_{name}.npy", Y)

def processFilesNpyComplex(filesFolder, filesList, allyPointer, rangeStart, rangeEnd, savePath, pBar, consoleOutput):

    # 22050
    SAMPLE_RATE = rangeEnd
    CHUNK_DURATION = 3.0  # seconds
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
    # 20 is default
    MAX_FILES_PER_CLASS = 50
    TARGET_CHUNKS = 150

    DATASET_PATH = filesFolder

    features = ["RMS", "ZCR", "Pitch", "Pitch Delta", "Centroid", "Bandwidth", "Rolloff", "Flatness", "mfcc",
                "delta_mfcc", "delta2_mfcc"]

    X_train, y_train, X_val, y_val, X_test, y_test, test_file_ids = process_dataset_split(DATASET_PATH, MAX_FILES_PER_CLASS,
                                                                                          SAMPLE_RATE, CHUNK_SIZE, pBar)
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)

    # Encode labels
    le = LabelEncoder()
    # y_encoded = le.fit_transform(y)

    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)

    print(len(np.unique(y_train)))

    np.save(f"{savePath}X_train.npy", X_train)
    np.save(f"{savePath}X_val.npy", X_val)
    np.save(f"{savePath}X_test.npy", X_test)
    np.save(f"{savePath}y_train.npy", y_train)
    np.save(f"{savePath}y_test.npy", y_test)
    np.save(f"{savePath}y_val.npy", y_val)
    np.save(f"{savePath}test_file_ids.npy", test_file_ids)



def chunks_to_dataframe(X, y, max_points=200):
    rows = []

    for i in range(len(X)):
        chunk = X[i].squeeze()

        # optionally downsample for readability
        if len(chunk) > max_points:
            step = len(chunk) // max_points
            chunk = chunk[::step]

        row = {"label": y[i]}

        for j, val in enumerate(chunk):
            row[f"x{j}"] = val

        rows.append(row)

    return pd.DataFrame(rows)

"""
Util for complex data
"""
def rms_normalize(y, target_rms=0.1):
    rms = np.sqrt(np.mean(y**2))
    if rms == 0:
        return y
    return y * (target_rms / rms)

def fix_length(chunk, size):
    if len(chunk) < size:
        # pad with zeros
        return np.pad(chunk, (0, size - len(chunk)))
    else:
        return chunk[:size]

def split_into_chunks(y, chunk_size):
    chunks = []
    for i in range(0, len(y), chunk_size):
        chunk = y[i:i + chunk_size]
        if len(chunk) < chunk_size:
            continue  # skip small tail (or pad if you prefer)
        chunks.append(chunk)
    return chunks

def fix_spectrogram_size(mel, target_width=130):
    current_width = mel.shape[1]

    if current_width < target_width:
        # pad with zeros
        pad_width = target_width - current_width
        mel = np.pad(mel, ((0, 0), (0, pad_width)))
    else:
        # crop
        mel = mel[:, :target_width]

    return mel

def process_file_multifeature(file_path, SAMPLE_RATE, CHUNK_SIZE):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Normalize
    y = rms_normalize(y)

    # Split into chunks
    chunks = split_into_chunks(y, CHUNK_SIZE)
    # chunks = split_into_chunks(y, CHUNK_SIZE, 1)

    """
    Resize to match chunk length
    """
    def expand_feature(feature, target_length):
        return np.interp(
            np.linspace(0, len(feature), num=target_length),
            np.arange(len(feature)),
            feature
        )

    processed_chunks = []
    """
    Noise detection by RMS percentile
    """
    chunk_rms_values = []

    for chunk in chunks:
        chunk = fix_length(chunk, CHUNK_SIZE)

        rms = librosa.feature.rms(y=chunk)[0]
        chunk_rms_values.append(np.mean(rms))

    activity_threshold = np.percentile(
        chunk_rms_values,
        40
    )

    for chunk in chunks:
        chunk = fix_length(chunk, CHUNK_SIZE)

        """
        RMS and ZCR
        """
        rms = librosa.feature.rms(y=chunk)[0]

        """"
        Noise filter (if rms < treshhold) = processing
        """
        if np.mean(rms) < activity_threshold:
            continue

        zcr = librosa.feature.zero_crossing_rate(chunk)[0]

        rms_expanded = expand_feature(rms, len(chunk))
        zcr_expanded = expand_feature(zcr, len(chunk))

        """
        Pitch
        """

        f0, voiced_flag, voiced_probs = librosa.pyin(
            chunk,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )

        f0 = np.nan_to_num(f0)
        pitch_expanded = expand_feature(f0, len(chunk))

        """
        Harmonic energy
        """

        harmonic, percussive = librosa.effects.hpss(chunk)

        harmonic_rms = librosa.feature.rms(y=harmonic)[0]
        harmonic_expanded = expand_feature(
            harmonic_rms,
            len(chunk)
        )

        """
        Frequency Modulation
        """

        pitch_delta = np.gradient(f0)

        pitch_delta_expanded = expand_feature(
            pitch_delta,
            len(chunk)
        )

        """
        Spectral Centroid
        """

        centroid = librosa.feature.spectral_centroid(
            y=chunk,
            sr=sr
        )[0]

        centroid_expanded = expand_feature(
            centroid,
            len(chunk)
        )

        """
        Spectral Bandwidth
        """

        bandwidth = librosa.feature.spectral_bandwidth(
            y=chunk,
            sr=sr
        )[0]

        bandwidth_expanded = expand_feature(
            bandwidth,
            len(chunk)
        )

        """
        Spectral rolloff
        """

        rolloff = librosa.feature.spectral_rolloff(
            y=chunk,
            sr=sr
        )[0]

        rolloff_expanded = expand_feature(
            rolloff,
            len(chunk)
        )

        """
        Spectral flatness
        """

        flatness = librosa.feature.spectral_flatness(
            y=chunk
        )[0]

        flatness_expanded = expand_feature(
            flatness,
            len(chunk)
        )

        """
        Syllable Structure
        """

        energy = librosa.feature.rms(y=chunk)[0]
        threshold = np.mean(energy)
        syllables = energy > threshold
        syllable_count = np.sum(
            np.diff(syllables.astype(int)) == 1
        )

        syllable_feature = np.full(
            len(chunk),
            syllable_count
        )

        """
        MFCC
        """

        mfcc = librosa.feature.mfcc(
            y=chunk,
            sr=sr,
            n_mfcc=13
        )

        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        delta_mfcc = delta_mfcc.T
        delta2_mfcc = delta2_mfcc.T

        # =========================
        # Stack channels
        # =========================

        rms_rehsaped = rms.reshape(-1, 1)
        zcr_rehsaped = zcr.reshape(-1, 1)
        f0_rehsaped = f0.reshape(-1, 1)
        pitch_delta_rehsaped = pitch_delta.reshape(-1, 1)
        centroid_rehsaped = centroid.reshape(-1, 1)
        bandwidth_rehsaped = bandwidth.reshape(-1, 1)
        rolloff_rehsaped = rolloff.reshape(-1, 1)
        flatness_rehsaped = flatness.reshape(-1, 1)

        mfcc = mfcc.T

        # print(rms.shape)
        # print(mfcc.shape)

        combined = np.concatenate([
            rms_rehsaped,
            zcr_rehsaped,
            f0_rehsaped,
            pitch_delta_rehsaped,
            centroid_rehsaped,
            bandwidth_rehsaped,
            rolloff_rehsaped,
            flatness_rehsaped,
            mfcc,
            delta_mfcc,
            delta2_mfcc
        ], axis=-1)

        # combinedOriginal = np.stack([
        #     normalize_feature(chunk),
        #
        #     rms_expanded,
        #     zcr_expanded,
        #     pitch_expanded,
        #     harmonic_expanded,
        #     pitch_delta_expanded,
        #     centroid_expanded,
        #     bandwidth_expanded,
        #     rolloff_expanded,
        #     flatness_expanded,
        #     # mfcc
        #     # mfcc,
        #     # syllable_feature
        # ], axis=-1)

        # print(combined.shape)
        processed_chunks.append(combined)

    return processed_chunks

def normalize_feature(f):
    return (f - np.mean(f)) / (np.std(f) + 1e-8)

def downsample_sequence(data, factor=10):
    # data shape: (samples, time, features)
    samples, time_steps, feat = data.shape
    new_length = time_steps // factor
    # Trim to a multiple of factor
    trimmed = data[:, :new_length * factor, :]
    # Reshape and mean over the factor dimension
    return trimmed.reshape(samples, new_length, factor, feat).mean(axis=2)

"""
Main Procession
Data type (Sound, Spectrogram etc) is chosen after try: cycle starts
"""
def process_dataset_split(dataset_path, MAX_FILES_PER_CLASS, SAMPLE_RATE, CHUNK_SIZE, pBar, test_size=0.2):
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    test_file_ids = []

    i = 0
    pBar.setMaximum(sum(1 for item in dataset_path.iterdir() if item.is_dir()))

    for label in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, label)

        if not os.path.isdir(folder_path):
            continue

        files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

        if MAX_FILES_PER_CLASS:
            files = files[:MAX_FILES_PER_CLASS]

        train_files, temp_files = train_test_split(files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

        print(f"\nClass: {label}")
        print(f"Train files: {len(train_files)}, Val files: {len(val_files)}, Test files: {len(test_files)}")

        """check for simmilar files in several sets"""
        checkIntersections(train_files, val_files, test_files)

        for file in tqdm(train_files, desc=f"{label} train"):
            full_path = os.path.join(folder_path, file)

            try:
                # chunks = process_file(full_path)
                chunks = process_file_multifeature(full_path, SAMPLE_RATE, CHUNK_SIZE)
                # chunks = process_file_spectrogram(full_path, augment=True)

                for chunk in chunks:
                    X_train.append(chunk)
                    y_train.append(label)

            except Exception as e:
                print(f"Error processing {file}: {e}")

        for file in tqdm(val_files, desc=f"{label} val"):
            full_path = os.path.join(folder_path, file)

            try:
                # chunks = process_file(full_path)
                chunks = process_file_multifeature(full_path, SAMPLE_RATE, CHUNK_SIZE)
                # chunks = process_file_spectrogram(full_path, augment=False)

                for chunk in chunks:
                    X_val.append(chunk)
                    y_val.append(label)

            except Exception as e:
                print(f"Error processing {file}: {e}")

        for file in tqdm(test_files, desc=f"{label} test"):
            full_path = os.path.join(folder_path, file)

            try:
                # chunks = process_file(full_path)
                chunks = process_file_multifeature(full_path, SAMPLE_RATE, CHUNK_SIZE)
                # chunks = process_file_spectrogram(full_path, augment=False)

                for chunk in chunks:
                    X_test.append(chunk)
                    y_test.append(label)

                    test_file_ids.append(file)

            except Exception as e:
                print(f"Error processing {file}: {e}")

        pBar.setValue(i + 1)

    return (
        np.array(X_train), np.array(y_train),
        np.array(X_val), np.array(y_val),
        np.array(X_test), np.array(y_test),
        np.array(test_file_ids)
    )

def checkIntersections(train_files, val_files, test_files):
    common_files = list(set(train_files).intersection(val_files, test_files))

    print("Files in all 3 lists:", common_files)

    print(
        "Train-Val:",
        len(set(train_files) & set(val_files))
    )

    print(
        "Train-Test:",
        len(set(train_files) & set(test_files))
    )

    print(
        "Val-Test:",
        len(set(val_files) & set(test_files))
    )

# Util --------------------------------------------------------------------------------------------------------------

# Get note names from frequency data
NOTES = ["C", "C#", "D", "D#", "E", "F",
         "F#", "G", "G#", "A", "A#", "B"]
def freqToNoteName(freq, a4Freq=440.0):

    freq = float(freq.rstrip("Hz"))
    if freq <= 0:
        return None, None, None  # invalid frequency

    # Calculate MIDI note number
    midi = int(round(69 + 12 * np.log2(freq / a4Freq)))

    # Get note name and octave
    noteName = NOTES [midi % 12]
    octave = (midi // 12) - 1  # MIDI octave formula

    return noteName, octave, midi

# Get Harmonics from amp and frequencies
def extractHarmonics(frequencies, amplitudes, maxHarmonics=10, prominence=0.05):
    # Find all peaks in amplitude spectrum
    peakIndices, _ = scipy.signal.find_peaks(amplitudes, prominence=prominence * np.max(amplitudes))
    # Sort peaks by amplitude (strongest first)
    sortedPeaks = sorted(peakIndices, key=lambda i: amplitudes[i], reverse=True)

    # Assume strongest peak is fundamental frequency
    f0 = frequencies[sortedPeaks[0]]
    harmonics = [(1, f0, amplitudes[sortedPeaks[0]])]  # (harmonic_number, frequency, amplitude)

    # Look for integer multiples of f0
    for n in range(2, maxHarmonics + 1):
        targetFreq = f0 * n
        idx = np.argmin(np.abs(frequencies - targetFreq))
        harmonics.append((n, frequencies[idx], amplitudes[idx]))

    return f0, harmonics

# Timbre descriptors from amp and frequencies
def timbreDescriptors(frequencies, amplitudes):
    amplitudes = np.array(amplitudes)
    frequencies = np.array(frequencies)

    # Normalize amplitudes for descriptor calculations
    ampSum = np.sum(amplitudes)
    if ampSum == 0:
        return None

    # Spectral Centroid (brightness)
    centroid = np.sum(frequencies * amplitudes) / ampSum
    # Spectral Spread (variance)
    spread = np.sqrt(np.sum(((frequencies - centroid) ** 2) * amplitudes) / ampSum)

    # Spectral Flatness
    geometricMean = np.exp(np.mean(np.log(amplitudes + 1e-12)))
    flatness = geometricMean / (np.mean(amplitudes) + 1e-12)

    # Spectral Roll-off (85% energy)
    cumulativeEnergy = np.cumsum(amplitudes)
    rolloffFreq = frequencies[np.where(cumulativeEnergy >= 0.85 * ampSum)[0][0]]

    return {
        "Тембр: Центроїда": centroid,
        "Тембр: Поширення": spread,
        "Тембр: Плоскість": flatness,
        "Тембр: Спад": rolloffFreq
    }


def saveAsExcelAlt(fileName):

    before, sep, after = fileName.rpartition('/')
    fileName = after

    # Load existing master file (if exists)
    excelFile = pathlib.Path(fileName)
    if excelFile.exists():
        dataFrame = pd.read_excel(excelFile)
    else:
        dataFrame = pd.DataFrame()

    # Example: new recording data
    rec_id = "Rec003"
    frames = [
        {"Recording ID": rec_id, "Time (s)": 0.000, "Frequency (Hz)": 16.0, "Amplitude": 0.50, "Phase (rad)": 0.05,
         "Pitch": "C0", "Bandwidth (Hz)": 85, "Harmonic 1": 0.20, "Harmonic 2": 0.10, "Harmonic 3": 0.05},
        {"Recording ID": rec_id, "Time (s)": 0.010, "Frequency (Hz)": 17.0, "Amplitude": 0.50, "Phase (rad)": 0.05,
         "Pitch": "C0", "Bandwidth (Hz)": 84, "Harmonic 1": 0.21, "Harmonic 2": 0.11, "Harmonic 3": 0.04}
    ]

    # Append new data
    newDataFrame = pd.DataFrame(frames)
    master_df = pd.concat([dataFrame, newDataFrame], ignore_index=True)

    # Save back to Excel
    master_df.to_excel(excelFile, index=False)