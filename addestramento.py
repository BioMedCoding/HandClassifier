import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import torch
print(torch.__version__)
print(torch.cuda.is_available())
import torch.nn as nn
import torch.optim as optim

# General parameters
mostra_grafici_segnali = True
mostra_segnale_per_canale = False

percorso_dati_aperture = "Original_data/aperture.txt"
percorso_dati_chiusure = "Original_data/chiusure.txt"
percorso_label_training = "Prepared_data/label_dataset_completo"

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
allena_lda = True
allena_rete_neurale = False

rapporto_training_validation = 0.7
numero_worker = 14

salva_modelli = True
salvataggio_train_val = True
salvataggio_dataset_completo = False

percorso_salvataggio_modelli = "C:/Users/matte/Documents/GitHub/HandClassifier/Modelli_allenati_addestramento_gaussiano"
percorso_salvataggio_train_val = "C:/Users/matte/Documents/GitHub/HandClassifier/Prepared_data_gaussiano"
percorso_salvataggio_dataset_completo = "C:/Users/matte/Documents/GitHub/HandClassifier/Prepared_data"

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

def balance_dataset(input_data, target_labels, method='smote'):
    from imblearn.over_sampling import SMOTE
    if method == 'smote':
        smote = SMOTE()
        balanced_data, balanced_labels = smote.fit_resample(input_data, target_labels)
    else:
        raise ValueError('Metodo non riconosciuto. Usa "smote".')
    return balanced_data, balanced_labels

# Load signals
sig_aperture = pd.read_csv(percorso_dati_aperture, delimiter='\t', header=None).values[:, 1:]
sig_chiusura = pd.read_csv(percorso_dati_chiusure, delimiter='\t', header=None).values[:, 1:]
sig = np.concatenate((sig_aperture, sig_chiusura), axis=0)

# Filter signals
n_channel = sig.shape[1]
sig_filt = np.zeros_like(sig)

for i in range(n_channel):
    sos = signal.cheby2(4, 40, [f_taglio_basso, f_taglio_alta], btype='bandpass', fs=f_sample, output='sos')
    sig_filt[:, i] = signal.sosfilt(sos, sig[:, i])
    b_notch, a_notch = signal.iirnotch(f_notch, 30, f_sample)
    sig_filt[:, i] = signal.filtfilt(b_notch, a_notch, sig_filt[:, i])

envelope = np.zeros_like(sig_filt)

for i in range(n_channel):
    sos = signal.cheby2(4, 40, f_envelope, btype='low', fs=f_sample, output='sos')
    envelope[:, i] = signal.sosfilt(sos, np.abs(sig_filt[:, i]))

# Standardize envelope signals
scaler = StandardScaler()
envelope_std = scaler.fit_transform(envelope)

# Load labels
label_data = loadmat(percorso_label_training)
label_dataset_completo = label_data[list(label_data.keys())[-1]].ravel()

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

# Balance dataset
envelope_std, label_dataset_completo = balance_dataset(envelope_std, label_dataset_completo)

# Split dataset into training and validation sets
sig_train, sig_val, label_train, label_val = train_test_split(envelope_std, label_dataset_completo, train_size=rapporto_training_validation, stratify=label_dataset_completo)

# Save training and validation sets
if salvataggio_train_val:
    savemat(f"{percorso_salvataggio_train_val}/training_set.mat", {"sig_train": sig_train})
    savemat(f"{percorso_salvataggio_train_val}/validation_set.mat", {"sig_val": sig_val})
    savemat(f"{percorso_salvataggio_train_val}/label_train.mat", {"label_train": label_train})
    savemat(f"{percorso_salvataggio_train_val}/label_val.mat", {"label_val": label_val})

# Train LDA
if allena_lda:
    lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', priors=None, n_components=None, store_covariance=True, tol=0.0001)
    lda.fit(sig_train, label_train)
    if salva_modelli:
        savemat(f"{percorso_salvataggio_modelli}/lda_model.mat", {"lda_model": lda})

# Train SVM (Note: Hyperparameter tuning is omitted for simplicity)
if allena_svm:
    svm = SVC(kernel='rbf', C=21.344, gamma=0.55962)
    svm.fit(sig_train, label_train)
    if salva_modelli:
        savemat(f"{percorso_salvataggio_modelli}/svm_model.mat", {"svm_model": svm})

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

    if salva_modelli:
        torch.save(model.state_dict(), f"{percorso_salvataggio_modelli}/nn_model.pth")