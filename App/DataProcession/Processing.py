# Write to Excel, Graphs, Numpy
import xlsxwriter as xls
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pathlib
import pandas as pd
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

    # reading file
    # sampFreq, sound = wavfile.read(file_name)
    # # convert our sound (numpy) array to floating point values ranging from -1 to 1
    # sound = sound / 2.0 ** 15
    #
    # # Check for 2 or 1 channels
    # temp = sound[0]
    # print(type(temp))
    # if isinstance(temp, np.ndarray):
    #     signal = sound[:, 0]
    # else:
    #     signal = sound

    signal, sampFreq = librosa.load(
        file_name,
        sr=None,
        mono=True
    )
    consoleOutput.insertPlainText(f"Файл: {file_name}\n")

    # Signal lenght
    length_in_s = signal.shape[0] / sampFreq
    print("Sighnal lenght: ", length_in_s, "s")

    # Obtaining the proper frequency units
    # fft_spectrum = np.fft.rfft(signal)
    #
    # freq = np.fft.rfftfreq(signal.size, d=1. / sampFreq)
    # phase = np.angle(fft_spectrum)
    # magnitude = np.abs(fft_spectrum)  # magnitude spectrum
    # fft_spectrum_abs = np.abs(fft_spectrum)

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

    """Add additional sound data"""

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