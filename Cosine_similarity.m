
%% Inizializzazione
clear 
close all
clc

warning('off', 'MATLAB:table:ModifiedAndSavedVarnames'); % Disabilita il warning relativo agli header

tipo_filtro = "cheby2";
f_sample = 2000;                                    % Frequenza campionamento
f_taglio_basso = 20;                                % Frequenza minima del passabanda
f_taglio_alta = 400;                                % Frequenza massima del passabanda
f_notch = 50;                                       % Frequenza del notch
f_envelope = 4;
percH = 1.3;                                        % Percentuale frequenza alta
visualisation = "no";                               % Mostra grafici filtraggio

% Parametri postprocess
lunghezza_buffer_precedenti = 2;                    % 400 ha dato valori migliori
lunghezza_buffer_successivi = 2;


%% Carica i dati delle contrazioni massimali
% Carica i dati delle contrazioni massimali utilizzando readtable
apertura_max = readtable('Original_data\massimale_apertura.txt', "Delimiter", '\t');
apertura_max(:,1) = [];
chiusura_max = readtable('Original_data\massimale_chiusura.txt', "Delimiter", '\t');
chiusura_max(:,1) = [];

% Converti i dati della tabella in matrici
apertura_max = table2array(apertura_max);
segnale_nullo = apertura_max(1:9000, :) ; %vertcat(apertura_max(1:9000, :), apertura_max(22000:end, :));
apertura_max = apertura_max(9544:21182, :);
chiusura_max = table2array(chiusura_max);
% segnale_nullo = vertcat(chiusura_max(1:8500, :), segnale_nullo, chiusura_max(22000:end, :));
chiusura_max = chiusura_max(9126:21182, :);

n_channel = length(apertura_max(1,:));

% Filtraggio banda
for i=1:n_channel
        apertura_max(:,i) = filter_general(apertura_max(:,i),tipo_filtro,f_sample,"fL",f_taglio_basso,"fH",f_taglio_alta,"fN",[f_notch f_notch*3 f_notch*5],"percH",percH,"visualisation",visualisation);
        chiusura_max(:,i) = filter_general(chiusura_max(:,i),tipo_filtro,f_sample,"fL",f_taglio_basso,"fH",f_taglio_alta,"fN",[f_notch f_notch*3 f_notch*5],"percH",percH,"visualisation",visualisation);
        segnale_nullo(:,1) = filter_general(segnale_nullo(:,i),tipo_filtro,f_sample,"fL",f_taglio_basso,"fH",f_taglio_alta,"fN",[f_notch f_notch*3 f_notch*5],"percH",percH,"visualisation",visualisation);
end

% Creazione inviluppo
for i=1:n_channel
        apertura_max(:,i) = filter_general(abs(apertura_max(:,i)),tipo_filtro,f_sample,"fH",f_envelope,"visualisation",visualisation,"percH",percH);
        chiusura_max(:,i) = filter_general(abs(chiusura_max(:,i)),tipo_filtro,f_sample,"fH",f_envelope,"visualisation",visualisation,"percH",percH);
        segnale_nullo(:,1) = filter_general(abs(segnale_nullo(:,i)),tipo_filtro,f_sample,"fH",f_envelope,"visualisation",visualisation,"percH",percH);
end

% Standardizzazione

segnale_complessivo = vertcat(apertura_max, chiusura_max,segnale_nullo);

mu = mean(segnale_complessivo);
sigma = std(segnale_complessivo);

apertura_max = (apertura_max - mu) ./ sigma;
chiusura_max = (chiusura_max - mu) ./ sigma;
segnale_nullo = (segnale_nullo - mu) ./ sigma;

% NO, STANDARDIZZARE PER CLASSE È UNA TROIATA, SERVE STANDARDIZZARE PER IL
% VALORE COMPLESSIVO, COME NELLO SCRIPT DI ADDESTRAMENTO
% mu_apertura = mean(apertura_max);
% sigma_apertura = std(apertura_max);
% apertura_max = (apertura_max - mu_apertura) ./ sigma_apertura;
% 
% mu_chiusura = mean(apertura_max);
% sigma_chiusura = std(apertura_max);
% chiusura_max = (chiusura_max - mu_chiusura) ./ sigma_chiusura;
% 
% mu_nullo = mean(segnale_nullo);
% sigma_nullo = std(segnale_nullo);
% segnale_nullo = (segnale_nullo - mu_nullo) ./ sigma_nullo;

% Carica dati contrazioni normali da usare come prototipo
%dataset_completo = load("Prepared_data\dataset_completo.mat");
%dataset_completo = dataset_completo.sig_salvabile;
%apertura_max = dataset_completo(26304:31596,:);
%chiusura_max = dataset_completo(371776:376465,:);


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
        RMS_values_apertura(w, ch) = rms(window_data);
        [psd_values_apertura, f_values_apertura] =  psd_general(window_data,'welch',2000,'window','hamming');
        MNF_values_apertura(w, ch) = mean(sum(psd_values_apertura.*f_values_apertura))/sum(psd_values_apertura);
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
        window_data = chiusura_max(start_idx:end_idx, ch);
        ARV_values_chiusura(w, ch) = mean(abs(window_data));
        RMS_values_chiusura(w, ch) = rms(window_data);
        [psd_values_chiusura, f_values_chiusura] =  psd_general(window_data,'welch',2000,'window','hamming');
        MNF_values_chiusura(w, ch) = mean(sum(psd_values_chiusura.*f_values_chiusura))/sum(psd_values_chiusura);
    end
end

%% Calcola i valori di ARV per ciascun canale per i dati non attivi

% Prima si vanno a spostare randomicamente i valori all'interno delle
% colonne

% Inizializziamo una matrice B della stessa dimensione di A
[nRows, nCols] = size(segnale_nullo);
segnale_nullo_random = zeros(nRows, nCols);

% Per ogni colonna di A, mescoliamo i valori e li mettiamo in B
% for col = 1:nCols
%     % Otteniamo una permutazione casuale degli indici delle righe
%     randIdx = randperm(nRows);
%     % Riorganizziamo gli elementi della colonna corrente
%     segnale_nullo_random(:, col) = segnale_nullo(randIdx, col);
% end
% 
% segnale_nullo = segnale_nullo_random;

num_samples_nullo = size(segnale_nullo, 1);
num_windows_nullo = floor((num_samples_nullo - window_size) / step) + 1;

ARV_values_nullo = zeros(num_windows_nullo, num_channels);

for ch = 1:num_channels
    for w = 1:num_windows_nullo 
        start_idx = (w-1) * step + 1;
        end_idx = start_idx + window_size - 1;
        window_data = segnale_nullo(start_idx:end_idx, ch);
        ARV_values_nullo(w, ch) = mean(abs(window_data));
        RMS_values_nullo(w, ch) = rms(window_data);
        [psd_values_nullo, f_values_nullo] =  psd_general(window_data,'welch',2000,'window','hamming');
        MNF_values_nullo(w, ch) = mean(sum(psd_values_nullo.*f_values_nullo))/sum(psd_values_nullo);
    end
end

%% Calcola i prototipi mediando i valori di ARV per ciascun canale
prototype_apertura_ARV = mean(ARV_values_apertura, 1);
prototype_chiusura_ARV = mean(ARV_values_chiusura, 1);
prototype_nullo_ARV = mean(ARV_values_nullo, 1);

prototype_apertura_RMS = mean(RMS_values_apertura, 1);
prototype_chiusura_RMS = mean(RMS_values_chiusura, 1);
prototype_nullo_RMS = mean(RMS_values_nullo, 1);

prototype_apertura_MNF = mean(MNF_values_apertura, 1);
prototype_chiusura_MNF = mean(MNF_values_chiusura, 1);
prototype_nullo_MNF = mean(MNF_values_nullo, 1);

prototype_apertura = horzcat(prototype_apertura_RMS,prototype_apertura_MNF);
prototype_chiusura = horzcat(prototype_chiusura_RMS,prototype_chiusura_MNF);
prototype_nullo = horzcat(prototype_nullo_RMS,prototype_nullo_MNF);


% prototype_apertura = prototype_apertura_ARV;
% prototype_chiusura = prototype_chiusura_ARV;
% prototype_nullo = prototype_nullo_ARV;

figure
hold on
plot(prototype_apertura, 'o');
plot(prototype_chiusura,'o');
plot(prototype_nullo,'o');
title('Actions prototypes complete')
legend('Opening','Closure','None')

figure
hold on
plot(prototype_apertura_ARV, 'o');
plot(prototype_chiusura_ARV,'o');
plot(prototype_nullo_ARV,'o');
title('Actions prototypes ARV')
legend('Opening','Closure','None')

%% Classificatore che usa la cosine similarity
% Carica i dati di test utilizzando load
ground_truth = load("label_test.mat");
ground_truth = ground_truth.label_test;

test_data = load("test_set.mat");
test_data = test_data.sig_test;

n_channel = length(test_data(1,:));
    
% Filtraggio banda
for i=1:n_channel
    test_data(:,i) = filter_general(test_data(:,i),tipo_filtro,f_sample,"fL",f_taglio_basso,"fH",f_taglio_alta,"fN",[f_notch f_notch*3 f_notch*5],"visualisation",visualisation);
end

% Creazione inviluppo
for i=1:n_channel
    test_data(:,i) = filter_general(abs(test_data(:,i)),tipo_filtro,f_sample,"fH",f_envelope,"visualisation",visualisation,"percH",percH);
end

% Standardizzazione

% Usando valori suoi
%test_data = (test_data-mean(test_data))./std(test_data);  

% Usando valori training
test_data = (test_data-mu)./sigma;



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
        RMS_values_test(w, ch) = rms(window_data);
        [psd_values_test, f_values_test] =  psd_general(window_data,'welch',2000,'window','hamming');
        MNF_values_test(w, ch) = mean(sum(psd_values_test.*f_values_test))/sum(psd_values_test);
    end
end

% Classifica ciascun segmento di test utilizzando la cosine similarity
predicted_classes = zeros(num_windows_test, 1);

for w = 1:num_windows_test
    test_ARV = ARV_values_test(w, :);
    
    similarity_apertura = cosine_similarity(test_ARV, prototype_apertura_ARV);
    similarity_chiusura = cosine_similarity(test_ARV, prototype_chiusura_ARV);
    similarity_nullo = cosine_similarity(test_ARV, prototype_nullo_ARV);

    %test_ARV = horzcat(RMS_values_test(w,:),MNF_values_test(w,:));
    %similarity_apertura = cosine_similarity(test_ARV, prototype_apertura);
    %similarity_chiusura = cosine_similarity(test_ARV, prototype_chiusura);
    %similarity_nullo = cosine_similarity(test_ARV, prototype_nullo);
    
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

% Calcolo metriche
metodo = "Prediction processate";
set = "Test";
label_test = ground_truth_windows;
predizione = predicted_classes;
mostra_cm = false;
classi = {'Rilassata', 'Apertura','Chiusura'};       % Nomi assegnati alle classi

[CM_prediction_processate, acc_prediction_processate, prec_prediction_processate, spec_prediction_processate, sens_prediction_processate, f1_prediction_processate, sens_media_processate, spec_media_processate] = evaluaClassificatore(label_test, predizione, mostra_cm, classi, metodo, set);



%% Applicazione post-process
tic
coda = 1;
buffer_precedenti = [];
buffer_futuri = [];
cambiamento_stato = 0;
predizione_originale = predicted_classes;

predizione_processata = zeros(length(predizione_originale),1);

for index = 1:length(predizione_originale)  % Ciclo for per simulare dati acquisiti live
    if index < lunghezza_buffer_precedenti+1  % Nel caso live, la variabile di controllo sarebbe il numero di campioni già ricevuti
        predizione_processata(index) = predizione_originale(index);
        buffer_precedenti(index) = predizione_originale(index); 
    
    elseif index > length(predizione_originale)-lunghezza_buffer_successivi % Questo controllo invece non sarebbe possibile
        predizione_processata(index) = predizione_originale(index);
    
    else % Caso dove si è distante da inizio e fine
        nuovo_campione = predizione_originale(index);
        [valore_corretto, cambiamento_stato, buffer_precedenti, buffer_futuri, coda] = liveProcess(buffer_precedenti, buffer_futuri, nuovo_campione, cambiamento_stato, coda, lunghezza_buffer_successivi);
        
        if cambiamento_stato ~= 2 && cambiamento_stato ~= -1 && valore_corretto ~= -1
            predizione_processata(index) = valore_corretto;
        
        elseif valore_corretto == -1 && cambiamento_stato == 1 % Caso in cui si sta popolando il vettore_futuri
            %fprintf("Rilevato cambio valore, valutazione correttezza \n");
        
        elseif cambiamento_stato == 2 || cambiamento_stato == -1
            predizione_processata(index-lunghezza_buffer_successivi:index) = valore_corretto;
            cambiamento_stato = 0;
        end
    end
end

elapsed_time = toc;
fprintf('   Termine postprocess. Tempo necessario: %.2f secondi\n', elapsed_time);

% Calcolo metriche post-process
predizione = predizione_processata;
[CM_prediction_processate, acc_prediction_processate, prec_prediction_processate, spec_prediction_processate, sens_prediction_processate, f1_prediction_processate, sens_media_processate, spec_media_processate] = evaluaClassificatore(label_test, predizione, mostra_cm, classi, metodo, set);


figure;

% Plot delle predizioni originali
subplot(3, 1, 1);
plot(predicted_classes);
title('Predizioni Cosine - test');
xlabel('Campioni');
ylabel('[a.u.]');

% Plot delle predizioni processate
subplot(3, 1, 2);
plot(predizione_processata);
title('Predizioni Cosine - test');
xlabel('Campioni');
ylabel('[a.u.]');

% Plot del ground truth
subplot(3, 1, 3);
plot(ground_truth_windows);
title('Ground truth');
xlabel('Campioni');
ylabel('[a.u.]');

linkaxes([subplot(3,1,1), subplot(3,1,2), subplot(3,1,3)]);

%% Funzioni usate

function sim = cosine_similarity(vec1, vec2)
    dot_product = dot(vec1, vec2);
    norm_vec1 = norm(vec1);
    norm_vec2 = norm(vec2);
    sim = dot_product / (norm_vec1 * norm_vec2);
end

function [CM, accuratezza, precisione, specificita, sensibilita, f1Score, sens_media, spec_media] = evaluaClassificatore(label_val, prediction, mostra_cm, classi, nomeMetodo, nomeSet)
    
    CM = confusionmat(label_val, prediction);

    if mostra_cm
        figure();
        confusionchart(CM, classi);
        title(['Confusion Matrix - ', nomeMetodo, ' - ', nomeSet]);
    end

    % Calcolo delle metriche per ogni classe
    numClassi = size(CM, 1);
    precisione = zeros(numClassi, 1);
    sensibilita = zeros(numClassi, 1);
    f1Score = zeros(numClassi, 1);
    specificita = zeros(numClassi, 1);

    for i = 1:numClassi
        precisione(i) = CM(i, i) / sum(CM(:, i));
        sensibilita(i) = CM(i, i) / sum(CM(i, :));

        if (precisione(i) + sensibilita(i)) == 0
            f1Score(i) = 0;
        else
            f1Score(i) = 2 * (precisione(i) * sensibilita(i)) / (precisione(i) + sensibilita(i));
        end

        TN = sum(CM(:)) - sum(CM(i, :)) - sum(CM(:, i)) + CM(i, i);
        FP = sum(CM(:, i)) - CM(i, i);
        specificita(i) = TN / (TN + FP);
    end

    % Calcolo dell'accuratezza complessiva
    total = sum(CM(:));
    accuratezza = sum(diag(CM)) / total;

    sens_media = mean(sensibilita);
    spec_media = mean(specificita);

    % Stampa delle metriche per ogni classe
    fprintf('\n---------------------------\n');
    fprintf('Accuratezza complessiva %s - %s: %.2f%%\n', nomeMetodo, nomeSet, accuratezza * 100);
    fprintf('Sensibilità complessiva %s - %s: %.2f%%\n', nomeMetodo, nomeSet, sens_media * 100);
    fprintf('Specificità complessiva %s - %s: %.2f%%\n\n', nomeMetodo, nomeSet, spec_media * 100);
    fprintf('%-10s %-12s %-12s %-12s %-12s\n', 'Classe', 'Precisione', 'Specificità', 'Sensibilità', 'F1 Score');

    for i = 1:numClassi
        fprintf('%-10d %-12.2f %-12.2f %-12.2f %-12.2f\n', i, precisione(i)*100, specificita(i)*100, sensibilita(i)*100, f1Score(i)*100);
    end

    fprintf('\n---------------------------\n');
end

function [correct_value, cambiamento_stato, buffer_precedenti, buffer_futuri, coda] = liveProcess(buffer_precedenti, buffer_futuri, nuovo_campione, cambiamento_stato, coda, futuri_massimi)

    if all(nuovo_campione ~= buffer_precedenti )
            cambiamento_stato = 1;
    end

    if cambiamento_stato
        if (coda) < futuri_massimi+1                  % Caso in cui il transitorio di osservazione non è ancora finito
            buffer_futuri(coda) = nuovo_campione;
            correct_value = -1;                       % In questo modo si segnala che è durante un transitorio
            buffer_precedenti = buffer_precedenti;    % Il buffer rimane inalterato
            coda = coda+1;                            % Aggiorna quello che è il contatore degli elementi in coda nel buffer futuro

        else                                          % Caso in cui il transitorio di osservazione è finito
            if all(buffer_futuri == buffer_futuri(1)) % Indica che tutti i nuovi valori sono coerenti
                 correct_value = buffer_futuri(1);    % Restituisci il valore corretto
                 cambiamento_stato = 2;               % Segnala termine controllo con risultati ok
                 buffer_precedenti = buffer_futuri(end-length(buffer_precedenti)+1:end);
                 buffer_futuri = [];                  % Resetta il buffer_futuri
                 coda = 1;
            else
                correct_value = buffer_precedenti(end); % Caso in cui i nuovi valori non sono coerenti, restituisce l'ultimo prima del cambiamento
                cambiamento_stato = -1;
                buffer_futuri = [];
                coda = 1;
            end
        end

    else
        correct_value = nuovo_campione;
    end
end