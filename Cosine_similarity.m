%% Inizializzazione
clear 
close all
clc

warning('off', 'MATLAB:table:ModifiedAndSavedVarnames'); % Disabilita il warning relativo agli header

%% Carica i dati delle contrazioni massimali
% Carica i dati delle contrazioni massimali utilizzando readtable
apertura_max = readtable('Data\massimale_apertura.txt', "Delimiter", '\t');
apertura_max(:,1) = [];
chiusura_max = readtable('Data\massimale_chiusura.txt', "Delimiter", '\t');
chiusura_max(:,1) = [];

% Converti i dati della tabella in matrici
apertura_max = table2array(apertura_max);
segnale_nullo = vertcat(apertura_max(1:9000, :), apertura_max(22000:end, :));
apertura_max = apertura_max(9544:21182, :);
chiusura_max = table2array(chiusura_max);
segnale_nullo = vertcat(chiusura_max(1:8500, :), segnale_nullo, chiusura_max(22000:end, :));
chiusura_max = chiusura_max(9126:21182, :);

%% Parametri per la finestratura
window_size = 250; % 250 ms
overlap = 0.5;     % 50% overlap
step = window_size * (1 - overlap);

%% Calcola i valori di ARV per ciascun canale per i dati di apertura
num_channels = size(apertura_max, 2);
num_samples_apertura = size(apertura_max, 1);
num_windows_apertura = floor((num_samples_apertura - window_size) / step) + 1;

ARV_values_apertura = zeros(num_windows_apertura, num_channels);

for ch = 1:num_channels
    for w = 1:num_windows_apertura
        start_idx = (w-1) * step + 1;
        end_idx = start_idx + window_size - 1;
        window_data = apertura_max(start_idx:end_idx, ch);
        ARV_values_apertura(w, ch) = mean(abs(window_data));
    end
end

%% Calcola i valori di ARV per ciascun canale per i dati di chiusura
num_samples_chiusura = size(chiusura_max, 1);
num_windows_chiusura = floor((num_samples_chiusura - window_size) / step) + 1;

ARV_values_chiusura = zeros(num_windows_chiusura, num_channels);

for ch = 1:num_channels
    for w = 1:num_windows_chiusura
        start_idx = (w-1) * step + 1;
        end_idx = start_idx + window_size - 1;
        window_data = chiusura_max(start_idx:end:end_idx, ch);
        ARV_values_chiusura(w, ch) = mean(abs(window_data));
    end
end

%% Calcola i valori di ARV per ciascun canale per i dati non attivi

% Prima si vanno a spostare randomicamente i valori all'interno delle
% colonne

% Inizializziamo una matrice B della stessa dimensione di A
[nRows, nCols] = size(segnale_nullo);
segnale_nullo_random = zeros(nRows, nCols);

% Per ogni colonna di A, mescoliamo i valori e li mettiamo in B
for col = 1:nCols
    % Otteniamo una permutazione casuale degli indici delle righe
    randIdx = randperm(nRows);
    % Riorganizziamo gli elementi della colonna corrente
    segnale_nullo_random(:, col) = segnale_nullo(randIdx, col);
end

segnale_nullo = segnale_nullo_random;

num_samples_nullo = size(segnale_nullo, 1);
num_windows_nullo = floor((num_samples_nullo - window_size) / step) + 1;

ARV_values_nullo = zeros(num_windows_nullo, num_channels);

for ch = 1:num_channels
    for w = 1:num_windows_nullo 
        start_idx = (w-1) * step + 1;
        end_idx = start_idx + window_size - 1;
        window_data = segnale_nullo(start_idx:end:end_idx, ch);
        ARV_values_nullo(w, ch) = mean(abs(window_data));
    end
end

%% Calcola i prototipi mediando i valori di ARV per ciascun canale
prototype_apertura = mean(ARV_values_apertura, 1);
prototype_chiusura = mean(ARV_values_chiusura, 1);
prototype_nullo = mean(ARV_values_nullo, 1);

%% Classificatore che usa la cosine similarity
% Carica i dati di test utilizzando load
ground_truth = load("label_test.mat");
ground_truth = ground_truth.label_test;

test_data = load("test_set.mat");
test_data = test_data.envelope_std_test;

% Calcola i valori di ARV per ciascun canale per i dati di test
num_samples_test = size(test_data, 1);
num_windows_test = floor((num_samples_test - window_size) / step) + 1;

ARV_values_test = zeros(num_windows_test, num_channels);

for ch = 1:num_channels
    for w = 1:num_windows_test
        start_idx = (w-1) * step + 1;
        end_idx = start_idx + window_size - 1;
        window_data = test_data(start_idx:end_idx, ch);
        ARV_values_test(w, ch) = mean(abs(window_data));
    end
end

% Classifica ciascun segmento di test utilizzando la cosine similarity
predicted_classes = zeros(num_windows_test, 1);

for w = 1:num_windows_test
    test_ARV = ARV_values_test(w, :);
    similarity_apertura = cosine_similarity(test_ARV, prototype_apertura);
    similarity_chiusura = cosine_similarity(test_ARV, prototype_chiusura);
    similarity_nullo = cosine_similarity(test_ARV, prototype_nullo);
    
    % Assegna la classe in base alla massima similarità
    [~, max_index] = max([similarity_nullo, similarity_apertura, similarity_chiusura]);
    predicted_classes(w) = max_index - 1; % 0 per nullo, 1 per apertura, 2 per chiusura
end

%% Mappa il ground truth alle finestre
ground_truth_windows = zeros(num_windows_test, 1);

for w = 1:num_windows_test
    start_idx = (w-1) * step + 1;
    end_idx = start_idx + window_size - 1;
    ground_truth_windows(w) = mode(ground_truth(start_idx:end_idx));
end

%% Plot dei risultati
figure;

% Plot delle predizioni
subplot(2, 1, 1);
plot(predicted_classes);
title('Predizioni Cosine - test');
xlabel('Campioni');
ylabel('[a.u.]');

% Plot del ground truth
subplot(2, 1, 2);
plot(ground_truth_windows);
title('Ground truth');
xlabel('Campioni');
ylabel('[a.u.]');

linkaxes([subplot(2,1,1), subplot(2,1,2)]);

% Definizione delle funzioni
function sim = cosine_similarity(vec1, vec2)
    dot_product = dot(vec1, vec2);
    norm_vec1 = norm(vec1);
    norm_vec2 = norm(vec2);
    sim = dot_product / (norm_vec1 * norm_vec2);
end
