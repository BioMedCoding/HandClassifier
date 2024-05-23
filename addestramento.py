import numpy as np
import pandas as pd
import scipy.signal as sp_signal
import h5py
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import signal
from contextlib import contextmanager

# General parameters
mostra_grafici_segnali = True
mostra_segnale_per_canale = False

# Define relative paths based on the current working directory
base_dir = os.path.dirname(os.path.abspath(__file__))
percorso_dati_aperture = os.path.join(base_dir, "Original_data/aperture.txt")
percorso_dati_chiusure = os.path.join(base_dir, "Original_data/chiusure.txt")
percorso_label_training = os.path.join(base_dir, "Prepared_data/label_dataset_completo.mat")

valore_apertura = 1
valore_chiusura = 2
classi = ['Rilassata', 'Apertura', 'Chiusura']

applica_data_augmentation = True
applica_data_augmentation_rumore_gaussiano = False
livello_rumore_gaussiano = 0.01
applica_data_augmentation_ampiezza_dinamica = True
amp_range = [0.7, 1.3]
change_rate = 5

allena_svm = False
allena_lda = False
allena_rete_neurale = True

rapporto_training_validation = 0.7
numero_worker = 14

salva_modelli = True
salvataggio_train_val = True
salvataggio_dataset_completo = False

percorso_salvataggio_modelli = os.path.join(base_dir, "Modelli_allenati_addestramento_gaussiano")
percorso_salvataggio_train_val = os.path.join(base_dir, "Prepared_data_gaussiano")
percorso_salvataggio_dataset_completo = os.path.join(base_dir, "Prepared_data")

# Timeout settings
svm_hyperparameter_tuning_timeout = 3600  # 1 hour

# Ensure directories exist
os.makedirs(percorso_salvataggio_modelli, exist_ok=True)
os.makedirs(percorso_salvataggio_train_val, exist_ok=True)
os.makedirs(percorso_salvataggio_dataset_completo, exist_ok=True)

# Filter parameters
tipo_filtro = "cheby2"
f_sample = 2000
f_taglio_basso = 20
f_taglio_alta = 400
f_notch = 50
f_envelope = 4
percH = 1.3
visualisation = "no"

# Training parameters for LDA
discrimType = 'quadratic'

# Training parameters for Neural Network
max_epoche = 500
val_metrica_obiettivo = 0.00005
lr = 0.02
momentum = 0.9

train_function = 'adam'
neuron_function = 'logsig'
num_layer = 3
layer1, layer2, layer3 = 10, 5, 3  # Adjusted for 3 layers

# Functions to be used in data processing
def add_gaussian_noise(signal, noise_level):
    noise = noise_level * np.random.randn(*signal.shape)
    return signal + noise

def vary_amplitude(signal, amp_range, change_rate):
    time_vector = np.arange(len(signal))
    scaling_factors = amp_range[0] + (amp_range[1] - amp_range[0]) * 0.5 * (1 + np.sin(2 * np.pi * change_rate * time_vector / len(signal)))
    return signal * scaling_factors

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Load signals with error handling
try:
    sig_aperture = pd.read_csv(percorso_dati_aperture, delimiter='\t', header=4, usecols=range(1, 6))
    sig_chiusura = pd.read_csv(percorso_dati_chiusure, delimiter='\t', header=4, usecols=range(1, 6))
except pd.errors.ParserError as e:
    print(f"Error reading file: {e}")
    # Optionally, inspect the file content here
    with open(percorso_dati_aperture, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines[:10]):  # Display the first 10 lines
            print(f"Line {i+1}: {line.strip()}")
    raise

# Convert to numpy array and remove the first column
sig_aperture = sig_aperture.values
sig_chiusura = sig_chiusura.values

sig = np.concatenate((sig_aperture, sig_chiusura), axis=0)

# Filter signals
n_channel = sig.shape[1]
sig_filt = np.zeros_like(sig)

for i in range(n_channel):
    sos = sp_signal.cheby2(4, 40, [f_taglio_basso, f_taglio_alta], btype='bandpass', fs=f_sample, output='sos')
    sig_filt[:, i] = sp_signal.sosfilt(sos, sig[:, i])
    b_notch, a_notch = sp_signal.iirnotch(f_notch, 30, f_sample)
    sig_filt[:, i] = sp_signal.filtfilt(b_notch, a_notch, sig_filt[:, i])

envelope = np.zeros_like(sig_filt)

for i in range(n_channel):
    sos = sp_signal.cheby2(4, 40, f_envelope, btype='low', fs=f_sample, output='sos')
    envelope[:, i] = sp_signal.sosfilt(sos, np.abs(sig_filt[:, i]))

# Standardize envelope signals
scaler = StandardScaler()
envelope_std = scaler.fit_transform(envelope)

# Load labels using h5py for MATLAB v7.3 files
def loadmat_h5py(filename):
    with h5py.File(filename, 'r') as f:
        # Assuming the data you need is stored in the first dataset
        return {k: np.array(v) for k, v in f.items()}

label_data = loadmat_h5py(percorso_label_training)
label_dataset_completo = label_data[list(label_data.keys())[0]].ravel()

# Cut signal to match label length
envelope_std = envelope_std[:len(label_dataset_completo), :]

# Apply data augmentation
if applica_data_augmentation:
    envelope_std_originale = envelope_std.copy()
    augmentedData = envelope_std
    augmentedLabels = label_dataset_completo

    if applica_data_augmentation_rumore_gaussiano:
        envelope_std_gauss = np.zeros_like(envelope_std)
        for i in range(envelope_std_gauss.shape[0]):
            envelope_std_gauss[i, :] = add_gaussian_noise(envelope_std[i, :], livello_rumore_gaussiano)
        augmentedData = np.vstack((augmentedData, envelope_std_gauss))
        augmentedLabels = np.concatenate((augmentedLabels, label_dataset_completo))

    if applica_data_augmentation_ampiezza_dinamica:
        varied_amplitude_signal = np.zeros_like(envelope_std)
        for i in range(envelope_std.shape[0]):
            varied_amplitude_signal[i, :] = vary_amplitude(envelope_std[i, :], amp_range, change_rate)
        augmentedData = np.vstack((augmentedData, varied_amplitude_signal))
        augmentedLabels = np.concatenate((augmentedLabels, label_dataset_completo))

    envelope_std = augmentedData
    label_dataset_completo = augmentedLabels

# Split dataset into training and validation sets
sig_train, sig_val, label_train, label_val = train_test_split(envelope_std, label_dataset_completo, train_size=rapporto_training_validation, stratify=label_dataset_completo)

print("Dimensione train set: ")
print(sig_train.shape)

# Save training and validation sets to CSV
if salvataggio_train_val:
    np.savetxt(f"{percorso_salvataggio_train_val}/training_set.csv", sig_train, delimiter=",")
    np.savetxt(f"{percorso_salvataggio_train_val}/validation_set.csv", sig_val, delimiter=",")
    np.savetxt(f"{percorso_salvataggio_train_val}/label_train.csv", label_train, delimiter=",")
    np.savetxt(f"{percorso_salvataggio_train_val}/label_val.csv", label_val, delimiter=",")

# Train LDA
if allena_lda:
    lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', priors=None, n_components=None, store_covariance=True, tol=0.0001)
    lda.fit(sig_train, label_train)
    print("Allenamento modello LDA terminato")
    if salva_modelli:
        lda_model_path = f"{percorso_salvataggio_modelli}/lda_model.json"
        with open(lda_model_path, 'w') as f:
            json.dump({'coef_': lda.coef_.tolist(), 'intercept_': lda.intercept_.tolist(), 'classes_': lda.classes_.tolist(), 'priors_': lda.priors_.tolist(), 'covariance_': lda.covariance_.tolist()}, f)

# Hyperparameter tuning for SVM with time limit and parallel processing
if allena_svm:
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, n_jobs=-1)  # Use all available cores
    try:
        with time_limit(svm_hyperparameter_tuning_timeout):
            grid.fit(sig_train, label_train)
            best_svm = grid.best_estimator_
            print("Allenamento di un modello SVM terminato")
            if salva_modelli:
                svm_model_path = f"{percorso_salvataggio_modelli}/svm_model.json"
                with open(svm_model_path, 'w') as f:
                    json.dump({
                        'support_vectors_': best_svm.support_vectors_.tolist(),
                        'dual_coef_': best_svm.dual_coef_.tolist(),
                        'intercept_': best_svm.intercept_.tolist(),
                        'classes_': best_svm.classes_.tolist(),
                        'best_params_': grid.best_params_
                    }, f)
    except TimeoutError:
        print(f"Hyperparameter tuning for SVM exceeded {svm_hyperparameter_tuning_timeout} seconds and was stopped.")
    print("Allenamento finale SVM terminato")

# Train Neural Network using PyTorch
if allena_rete_neurale:
    class SimpleNN(nn.Module):
        def __init__(self, input_size, hidden_sizes, output_size):
            super(SimpleNN, self).__init__()
            layers = []
            in_size = input_size
            for h_size in hidden_sizes:
                layers.append(nn.Linear(in_size, h_size))
                layers.append(nn.LogSigmoid() if neuron_function == 'logsig' else nn.Tanh())
                in_size = h_size
            layers.append(nn.Linear(in_size, output_size))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    input_size = sig_train.shape[1]
    hidden_sizes = [layer1, layer2, layer3][:num_layer]
    output_size = len(classi)

    model = SimpleNN(input_size, hidden_sizes, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    sig_train_tensor = torch.tensor(sig_train, dtype=torch.float32)
    label_train_tensor = torch.tensor(label_train, dtype=torch.long)

    for epoch in range(max_epoche):
        model.train()
        optimizer.zero_grad()
        outputs = model(sig_train_tensor)
        loss = criterion(outputs, label_train_tensor)
        loss.backward()
        optimizer.step()
        if loss.item() < val_metrica_obiettivo:
            break

    print("Allenamento modello NN terminato")

    if salva_modelli:
        torch.save(model.state_dict(), f"{percorso_salvataggio_modelli}/nn_model.pth")
