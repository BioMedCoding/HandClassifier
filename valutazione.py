import numpy as np
import pandas as pd
import scipy.signal as sp_signal
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import h5py
import json

# General parameters
mostra_grafici_segnali = False
mostra_segnale_per_canale = False
mostra_cm = False
mostra_risultati_singoli = True
mostra_risultati_complessivi = True

classi = ['Rilassata', 'Apertura', 'Chiusura']

percorso_segnale = "Prepared_data/test_set_processato.mat"
percorso_label = "Prepared_data/label_test.mat"
nome_grafici = "Test set"

preprocessa_segnale = False

# Evaluation parameters
valuta_svm = False
valuta_lda = True
valuta_nn = True

percorso_salvataggio_svm = "Modelli_allenati_addestramento_gaussiano/svm_model.json"
percorso_salvataggio_lda = "Modelli_allenati_addestramento_gaussiano/lda_model.mat"
percorso_salvataggio_nn = "Modelli_allenati_addestramento_gaussiano/nn_model.pth"

prediction_parallel = False

# Filter parameters
tipo_filtro = "cheby2"
f_sample = 2000
f_taglio_basso = 20
f_taglio_alta = 400
f_notch = 50
f_envelope = 4
percH = 1.3
visualisation = "no"

# Post-process parameters
applica_postprocess_multiplo = False
applica_postprocess_singolo = True
segnale_da_elaborare = 'prediction_lda_test'

lunghezza_buffer_precedenti = 400
lunghezza_buffer_successivi = 400

# Load test signals and labels using h5py for MATLAB v7.3 files
def load_mat_file_h5py(filepath):
    with h5py.File(filepath, 'r') as f:
        var_names = list(f.keys())
        if var_names:
            return np.array(f[var_names[0]])
        else:
            raise ValueError('No variable found in the file.')

test_signal = load_mat_file_h5py(percorso_segnale)
print("Dimensione test set: ", test_signal.shape)

test_signal = test_signal.T

label_test = load_mat_file_h5py(percorso_label)
print("Dimensione originale label test set: ", label_test.shape)

# Converti label_test in un array monodimensionale se necessario
if label_test.ndim > 1:
    label_test = label_test.ravel()

print("Dimensione label test set dopo la conversione: ", label_test.shape)

# Assicurati che label_test e test_signal abbiano lo stesso numero di campioni
assert label_test.shape[0] == test_signal.shape[0], "Il numero di campioni in label_test e test_signal non corrisponde!"

# Preprocess signal
def preprocess_signal(signal, f_sample, f_taglio_basso, f_taglio_alta, f_notch, f_envelope, percH):
    n_channel = signal.shape[1]
    sig_filt = np.zeros_like(signal)

    for i in range(n_channel):
        sos = sp_signal.cheby2(4, 40, [f_taglio_basso, f_taglio_alta], btype='bandpass', fs=f_sample, output='sos')
        sig_filt[:, i] = sp_signal.sosfilt(sos, signal[:, i])
        b_notch, a_notch = sp_signal.iirnotch(f_notch, 30, f_sample)
        sig_filt[:, i] = sp_signal.filtfilt(b_notch, a_notch, sig_filt[:, i])

    envelope = np.zeros_like(sig_filt)
    for i in range(n_channel):
        sos = sp_signal.cheby2(4, 40, f_envelope, btype='low', fs=f_sample, output='sos')
        envelope[:, i] = sp_signal.sosfilt(sos, np.abs(sig_filt[:, i]))

    envelope_std = (envelope - np.mean(envelope)) / np.std(envelope)
    return envelope_std

if preprocessa_segnale:
    test_signal = preprocess_signal(test_signal, f_sample, f_taglio_basso, f_taglio_alta, f_notch, f_envelope, percH)

# Evaluation function
def evalua_classificatore(label_val, prediction, mostra_cm, classi, nome_metodo, nome_set):
    cm = confusion_matrix(label_val, prediction)
    if mostra_cm:
        print(confusion_matrix(label_val, prediction))
    
    report = classification_report(label_val, prediction, target_names=classi, output_dict=True)
    
    acc = report['accuracy']
    precision = [report[class_]['precision'] for class_ in classi]
    recall = [report[class_]['recall'] for class_ in classi]
    f1 = [report[class_]['f1-score'] for class_ in classi]
    specificity = [cm[i, i] / (cm[i, i] + cm[:, i].sum() - cm[i, i]) for i in range(len(classi))]
    
    print(f'\n---------------------------\nAccuratezza complessiva {nome_metodo} - {nome_set}: {acc*100:.2f}%')
    print(f'{"Classe":<10} {"Precisione":<12} {"Specificità":<12} {"Sensibilità":<12} {"F1 Score":<12}')
    for i, class_ in enumerate(classi):
        print(f'{class_:<10} {precision[i]*100:<12.2f} {specificity[i]*100:<12.2f} {recall[i]*100:<12.2f} {f1[i]*100:<12.2f}')
    print('\n---------------------------\n')
    return cm, acc, precision, specificity, recall, f1

# Load SVM model from JSON
def load_svm_model_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    svm_model = SVC(kernel='rbf')  # Default kernel; will be overwritten
    svm_model.support_vectors_ = np.array(data['support_vectors_'])
    svm_model.dual_coef_ = np.array(data['dual_coef_'])
    svm_model.intercept_ = np.array(data['intercept_'])
    svm_model._classes = np.array(data['classes_'])
    svm_model.C = data['best_params_']['C']
    svm_model.gamma = data['best_params_']['gamma']
    svm_model.kernel = data['best_params_']['kernel']
    return svm_model

# Load LDA model from JSON
def load_lda_model_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    lda_model = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    lda_model.coef_ = np.array(data['coef_'])
    lda_model.intercept_ = np.array(data['intercept_'])
    lda_model.classes_ = np.array(data['classes_'])
    lda_model.priors_ = np.array(data['priors_'])
    lda_model.covariance_ = np.array(data['covariance_'])
    return lda_model

# Evaluation SVM
if valuta_svm:
    svm_model = load_svm_model_from_json(percorso_salvataggio_svm)
    prediction_svm_test = svm_model.predict(test_signal)
    evalua_classificatore(label_test, prediction_svm_test, mostra_cm, classi, "SVM", nome_grafici)
    if mostra_risultati_singoli:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(prediction_svm_test)
        plt.title('Predizioni SVM - test')
        plt.xlabel('Campioni')
        plt.ylabel('[a.u.]')
        plt.subplot(2, 1, 2)
        plt.plot(label_test)
        plt.title('Ground truth')
        plt.xlabel('Campioni')
        plt.ylabel('[a.u.]')
        plt.show()

# Evaluation LDA
if valuta_lda:
    lda_model = load_lda_model_from_json(percorso_salvataggio_lda)
    prediction_lda_test = lda_model.predict(test_signal)
    evalua_classificatore(label_test, prediction_lda_test, mostra_cm, classi, "LDA", nome_grafici)
    if mostra_risultati_singoli:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(prediction_lda_test)
        plt.title('Predizioni LDA - test')
        plt.xlabel('Campioni')
        plt.ylabel('[a.u.]')
        plt.subplot(2, 1, 2)
        plt.plot(label_test)
        plt.title('Ground truth')
        plt.xlabel('Campioni')
        plt.ylabel('[a.u.]')
        plt.show()

# Evaluation NN
if valuta_nn:
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

    input_size = test_signal.shape[1]
    hidden_sizes = [layer1, layer2, layer3][:num_layer]
    output_size = len(classi)

    model = SimpleNN(input_size, hidden_sizes, output_size)
    model.load_state_dict(torch.load(percorso_salvataggio_nn))
    model.eval()

    test_signal_tensor = torch.tensor(test_signal, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(test_signal_tensor)
        _, prediction_nn_test = torch.max(outputs, 1)
    
    prediction_nn_test = prediction_nn_test.numpy()
    evalua_classificatore(label_test, prediction_nn_test, mostra_cm, classi, "NN", nome_grafici)
    if mostra_risultati_singoli:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(prediction_nn_test)
        plt.title('Predizioni NN - test')
        plt.xlabel('Campioni')
        plt.ylabel('[a.u.]')
        plt.subplot(2, 1, 2)
        plt.plot(label_test)
        plt.title('Ground truth')
        plt.xlabel('Campioni')
        plt.ylabel('[a.u.]')
        plt.show()

# Post-process function
def live_process(buffer_precedenti, buffer_futuri, nuovo_campione, cambiamento_stato, coda, futuri_massimi):
    if all(nuovo_campione != buffer_precedenti):
        cambiamento_stato = 1

    if cambiamento_stato:
        if coda < futuri_massimi + 1:
            buffer_futuri.append(nuovo_campione)
            correct_value = -1
            coda += 1
        else:
            if all([x == buffer_futuri[0] for x in buffer_futuri]):
                correct_value = buffer_futuri[0]
                cambiamento_stato = 2
                buffer_precedenti = buffer_futuri[-len(buffer_precedenti):]
                buffer_futuri = []
                coda = 1
            else:
                correct_value = buffer_precedenti[-1]
                cambiamento_stato = -1
                buffer_futuri = []
                coda = 1
    else:
        correct_value = nuovo_campione
    return correct_value, cambiamento_stato, buffer_precedenti, buffer_futuri, coda

def live_process2(buffer_precedenti, buffer_futuri, nuovo_campione, cambiamento_stato, coda, futuri_massimi):
    if all(nuovo_campione != buffer_precedenti):
        cambiamento_stato = 1

    if cambiamento_stato:
        if coda < futuri_massimi + 1:
            buffer_futuri.append(nuovo_campione)
            correct_value = -1
            coda += 1
        else:
            if all([x == buffer_futuri[0] for x in buffer_futuri]):
                correct_value = mode(buffer_futuri)
                cambiamento_stato = 2
                buffer_precedenti = buffer_futuri[-len(buffer_precedenti):]
                buffer_futuri = []
                coda = 1
            else:
                correct_value = mode(buffer_futuri)
                cambiamento_stato = -1
                buffer_futuri = []
                coda = 1
    else:
        correct_value = nuovo_campione
    return correct_value, cambiamento_stato, buffer_precedenti, buffer_futuri, coda

if applica_postprocess_singolo:
    predizione_originale = locals()[segnale_da_elaborare]

    # Post-process 1
    buffer_precedenti = []
    buffer_futuri = []
    cambiamento_stato = 0
    coda = 1

    predizione_processata = np.zeros(len(predizione_originale))
    for index in range(len(predizione_originale)):
        if index < lunghezza_buffer_precedenti + 1:
            predizione_processata[index] = predizione_originale[index]
            buffer_precedenti.append(predizione_originale[index])
        elif index > len(predizione_originale) - lunghezza_buffer_successivi:
            predizione_processata[index] = predizione_originale[index]
        else:
            nuovo_campione = predizione_originale[index]
            valore_corretto, cambiamento_stato, buffer_precedenti, buffer_futuri, coda = live_process(buffer_precedenti, buffer_futuri, nuovo_campione, cambiamento_stato, coda, lunghezza_buffer_successivi)
            if cambiamento_stato != 2 and cambiamento_stato != -1 and valore_corretto != -1:
                predizione_processata[index] = valore_corretto
            elif valore_corretto == -1 and cambiamento_stato == 1:
                pass
            elif cambiamento_stato == 2 or cambiamento_stato == -1:
                predizione_processata[index-lunghezza_buffer_successivi:index] = valore_corretto
                cambiamento_stato = 0

    # Post-process 2
    buffer_precedenti = []
    buffer_futuri = []
    cambiamento_stato = 0
    coda = 1

    predizione_processata2 = np.zeros(len(predizione_originale))
    for index in range(len(predizione_originale)):
        if index < lunghezza_buffer_precedenti + 1:
            predizione_processata2[index] = predizione_originale[index]
            buffer_precedenti.append(predizione_originale[index])
        elif index > len(predizione_originale) - lunghezza_buffer_successivi:
            predizione_processata2[index] = predizione_originale[index]
        else:
            nuovo_campione = predizione_originale[index]
            valore_corretto, cambiamento_stato, buffer_precedenti, buffer_futuri, coda = live_process2(buffer_precedenti, buffer_futuri, nuovo_campione, cambiamento_stato, coda, lunghezza_buffer_successivi)
            if cambiamento_stato != 2 and cambiamento_stato != -1 and valore_corretto != -1:
                predizione_processata2[index] = valore_corretto
            elif valore_corretto == -1 and cambiamento_stato == 1:
                pass
            elif cambiamento_stato == 2 or cambiamento_stato == -1:
                predizione_processata2[index-lunghezza_buffer_successivi:index] = valore_corretto
                cambiamento_stato = 0

    if mostra_risultati_singoli:
        plt.figure()
        plt.plot(predizione_originale)
        plt.plot(predizione_processata)
        plt.plot(predizione_processata2)
        plt.legend(["Predizione originale", "Predizione processata", "Predizione processata 2"])
        plt.show()

    evalua_classificatore(label_test, predizione_processata, mostra_cm, classi, "Prediction processate", "Test")
    evalua_classificatore(label_test, predizione_processata2, mostra_cm, classi, "Prediction processate 2", "Test")

    if mostra_risultati_complessivi:
        plt.figure()
        plt.subplot(4, 1, 1)
        plt.plot(predizione_originale)
        plt.title('Predizioni originali')
        plt.xlabel('Campioni')
        plt.ylabel('[a.u.]')
        plt.subplot(4, 1, 2)
        plt.plot(predizione_processata)
        plt.title('Predizioni processate')
        plt.xlabel('Campioni')
        plt.ylabel('[a.u.]')
        plt.subplot(4, 1, 3)
        plt.plot(predizione_processata2)
        plt.title('Predizioni processate 2')
        plt.xlabel('Campioni')
        plt.ylabel('[a.u.]')
        plt.subplot(4, 1, 4)
        plt.plot(label_test)
        plt.title('Ground truth')
        plt.xlabel('Campioni')
        plt.ylabel('[a.u.]')
        plt.show()
