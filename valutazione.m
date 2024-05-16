%% Inizializzazione
clear 
close all
clc

%% ======================== Parametri generali script ======================
mostra_grafici_segnali = false;  
mostra_segnale_per_canale = false;
mostra_cm = true;                                    % Mostra le CM dei vari classificatori
mostra_risultati_singoli = true;                     % Mostra confronto singolo classificatore - Ground Truth
mostra_risultati_complessivi = true;                 % Mostra confronto tutti i classificatori - Ground Truth

classi = {'Rilassata', 'Apertura','Chiusura'};       % Nomi assegnati alle classi

percorso_segnale = "Prepared_data/test_set.mat";
percorso_label = "Prepared_data/label_test.mat";
nome_grafici = "Test set";                           % Nome che viene mostrato nei grafici relativi ai risultati

preprocessa_segnale = true;

warning('off', 'MATLAB:table:ModifiedAndSavedVarnames'); % Disabilita il warning relativo agli header
%% =========================================================================




%% Selezione valutazioni da eseguire

valuta_svm = false;
valuta_lda = true;
valuta_nn = true;

percorso_salvataggio_svm = "Modelli_allenati\svm_model.mat";
percorso_salvataggio_lda = "Modelli_allenati_addestramento\lda_model.mat";
percorso_salvataggio_nn = "Modelli_allenati_addestramento\nn_model.mat";

prediction_parallel = false;
%% =========================================================================




%% ======================== Parametri filtraggio ===========================
tipo_filtro = "cheby2";
f_sample = 2000;                                    % Frequenza campionamento
f_taglio_basso = 20;                                % Frequenza minima del passabanda
f_taglio_alta = 400;                                % Frequenza massima del passabanda
f_notch = 50;                                       % Frequenza del notch
f_envelope = 4;                                     % Frequenza inviluppo
percH = 1.3;                                        % Percentuale frequenza alta
visualisation = "no";                               % Mostra grafici filtraggio
%% =========================================================================



% =========================== Parametri postprocess =======================
applica_postprocess_multiplo = true;                         % Applica funzion di postprocess sul vettore di classificazione

applica_postprocess_singolo = true;
segnale_da_elaborare = 'prediction_nn_test';                             

lunghezza_buffer_precedenti = 400;
lunghezza_buffer_successivi = 400;
% =========================================================================




%% ================= Avvio pool se necessario e non attivo =================
if prediction_parallel
    if isempty(gcp('nocreate'))
            parpool('local', numero_worker); % Avvia in modalità processes
            %parpool('Threads')              % Avvia in modalità thread
    end
end
%% =========================================================================




%% Import segnali ed eventuale preprocess

data = load(percorso_segnale);
varNames = fieldnames(data);

% Verifica che ci sia almeno una variabile nel file
if ~isempty(varNames)
    % Estrai la prima variabile trovata e assegnala a 'test_signal'
    test_signal = data.(varNames{1});
    
else
    disp('Nessuna variabile trovata nel file.');
end

if preprocessa_segnale
    n_channel = length(test_signal(1,:));
    sig_filt= zeros(length(test_signal),n_channel);
    
    for i=1:n_channel
        sig_filt(:,i) = filter_general(test_signal(:,i),tipo_filtro,f_sample,"fL",f_taglio_basso,"fH",f_taglio_alta,"fN",f_notch,"visualisation",visualisation);
    end

    if mostra_segnale_per_canale

        figure;
  
        subplot(5,1,1);
            plot(sig_filt(:,1));
            title('Segnale canale 1 grezzo - train');
            xlabel('Campioni');
            ylabel('[a.u.]');
        
        subplot(5,1,2);
            plot(sig_filt(:,2));
            title('Segnale canale 2 grezzo - train');
            xlabel('Campioni');
            ylabel('[a.u.]');
    
        subplot(5,1,3);
            plot(sig_filt(:,3));
            title('Segnale canale 3 grezzo - train');
            xlabel('Campioni');
            ylabel('[a.u.]');
    
        subplot(5,1,4);
            plot(sig_filt(:,4));
            title('Segnale canale 4 grezzo - train');
            xlabel('Campioni');
            ylabel('[a.u.]');
    
        subplot(5,1,5);
            plot(sig_filt(:,5));
            title('Segnale canale 5 grezzo - train');
            xlabel('Campioni');
            ylabel('[a.u.]');
        
        %Collega gli assi verticali dei due subplot
        linkaxes([subplot(5,1,1), subplot(5,1,2), subplot(5,1,3), subplot(5,1,4), subplot(5,1,5)]);
    end

    % Creazione inviluppo
    envelope = zeros(length(sig_filt),n_channel);

    for i=1:n_channel
        envelope(:,i) = filter_general(abs(sig_filt(:,i)),tipo_filtro,f_sample,"fH",f_envelope,"percH",percH);   
    end

    if mostra_grafici_segnali
        figure
        plot(envelope)
        title('Inviluppo segnale grezzo');
        xlabel('Campioni');
        ylabel('[uV]');
    end

    % Standardizza i valori
    envelope_std = (envelope-mean(envelope))./std(envelope);
    
    if mostra_segnale_per_canale
        figure;
    
        subplot(5,1,1);
            plot(envelope_std(:,1));
            title('Segnale canale 1 inviluppato e standardizzato - train');
            xlabel('Campioni');
            ylabel('[a.u.]');
        
        subplot(5,1,2);
            plot(envelope_std(:,2));
            title('Segnale canale 2 inviluppato e standardizzato - train');
            xlabel('Campioni');
            ylabel('[a.u.]');
    
        subplot(5,1,3);
            plot(envelope_std(:,3));
            title('Segnale canale 3 inviluppato e standardizzato - train');
            xlabel('Campioni');
            ylabel('[a.u.]');
    
        subplot(5,1,4);
            plot(envelope_std(:,4));
            title('Segnale canale 4 inviluppato e standardizzato - train');
            xlabel('Campioni');
            ylabel('[a.u.]');
    
        subplot(5,1,5);
            plot(envelope_std(:,5));
            title('Segnale canale 5 inviluppato e standardizzato - train');
            xlabel('Campioni');
            ylabel('[a.u.]');
        
        %Collega gli assi verticali dei due subplot
        linkaxes([subplot(5,1,1), subplot(5,1,2), subplot(5,1,3), subplot(5,1,4), subplot(5,1,5)]);
    end
    
    % Istruzione per usare il segnale senza inviluppo
    %test_signal = (sig_filt-mean(sig_filt))./std(sig_filt);
    test_signal = envelope_std;

    if mostra_grafici_segnali
        figure
        plot(envelope_std)
        hold on
    end
end   




%% Import label
data = load(percorso_label);
varNames = fieldnames(data);

% Verifica che ci sia almeno una variabile nel file
if ~isempty(varNames)
    % Estrai la prima variabile trovata e assegnala a 'test_signal'
    label_test = data.(varNames{1});
    
else
    disp('Nessuna variabile trovata nel file.');
end



%% Valutazione SVM

if valuta_svm

    load(percorso_salvataggio_svm)
    metodo = "SVM";
    set = nome_grafici;
    
    prediction_svm_test = predict(svm_model, test_signal);
    
    [CM_svm_test, acc_svm_test, prec_svm_test, spec_svm_test, sens_svm_test, f1_svm_test] = evaluaClassificatore(label_test, prediction_svm_test, mostra_cm, classi, metodo, set);
    
    if mostra_risultati_singoli
        figure;
        subplot(2,1,1);
        plot(prediction_svm_test);
        title('Predizioni SVM - test');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        subplot(2,1,2);
        plot(label_test);
        title('Ground truth');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        % Collega gli assi verticali dei due subplot
        linkaxes([subplot(2,1,1), subplot(2,1,2)]);
    
    end
end




%% Valutazione LDA

if valuta_lda

    load(percorso_salvataggio_lda)
    metodo = "LDA";
    set = nome_grafici;

    prediction_lda_test = predict(lda_model, test_signal);

    [CM_lda_test, acc_lda_test, prec_lda_test, spec_lda_test, sens_lda_test, f1_lda_test] = evaluaClassificatore(label_test, prediction_lda_test, mostra_cm, classi, metodo, set);

    if mostra_risultati_singoli
        figure;
        subplot(2,1,1);
        plot(prediction_lda_test);
        title('Predizioni LDA - test');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        subplot(2,1,2);
        plot(label_test);
        title('Ground truth');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        % Collega gli assi verticali dei due subplot
        linkaxes([subplot(2,1,1), subplot(2,1,2)]);
    end
end
%% Valutazione NN

if valuta_nn

    load(percorso_salvataggio_nn)

    set = nome_grafici;

    % Adatta segnali al formato richiesto dalla rete neurale
    test_signal = test_signal';
    label_test = label_test+1;
    label_test = full(ind2vec(label_test'));

    % Calcola predizioni
    prediction_nn_test = net(test_signal);
    prediction_nn_test = vec2ind(prediction_nn_test);  % Converti le probabilità in indici di classe
    
    % Riporta i valori adattati al formato originale
    label_test = vec2ind(label_test)';
    label_test = label_test-1;
    test_signal = test_signal';

    % Riporta alle label convenzionali
    prediction_nn_test = prediction_nn_test'-1; 
    metodo = "NN";
    
    [CM_nn_test, acc_nn_test, prec_nn_test, spec_nn_test, sens_nn_test, f1_nn_test] = evaluaClassificatore(label_test, prediction_nn_test, mostra_cm, classi, metodo, set); 
  
    if mostra_risultati_singoli
        figure;
        subplot(2,1,1);
        plot(prediction_nn_test);
        title('Predizioni NN - test');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        subplot(2,1,2);
        plot(label_test);
        title('Ground truth');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        % Collega gli assi verticali dei due subplot
        linkaxes([subplot(2,1,1), subplot(2,1,2)]);
    end
end




%% Rappresentazione dei risultati complessivi
if mostra_risultati_complessivi && valuta_nn && valuta_lda && valuta_svm
        figure;
        subplot(4,1,1);
        plot(prediction_svm_test);
        title('Predizioni SVM - test');
        xlabel('Campioni');
        ylabel('[a.u.]');
    
        subplot(4,1,2);
        plot(prediction_nn_test);
        title('Predizioni NN - test');
        xlabel('Campioni');
        ylabel('[a.u.]');
    
        subplot(4,1,3);
        plot(prediction_lda_test);
        title('Predizioni LDA - test');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        subplot(4,1,4);
        plot(label_test);
        title('Ground truth');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        % Collega gli assi verticali dei due subplot
        linkaxes([subplot(4,1,1), subplot(4,1,2), subplot(4,1,3), subplot(4,1,4)]);
end


%% Applicazione post-process

if applica_postprocess_singolo

    eval(['predizione_originale = ', segnale_da_elaborare, ';']);  % Permette di scegliere tramite la stringa all'inizio il segnale da postprocessare

    % Postprocess 1

    % Definizione parametri iniziali
    coda = 1;
    buffer_precedenti = [];
    buffer_futuri = [];
    cambiamento_stato = 0;

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
    
    % Postprocess 2
    
    % Definizione parametri iniziali
    coda = 1;
    buffer_precedenti = [];
    buffer_futuri = [];
    cambiamento_stato = 0;

    predizione_processata2 = zeros(length(predizione_originale),1);

    for index = 1:length(predizione_originale)  % Ciclo for per simulare dati acquisiti live
        if index < lunghezza_buffer_precedenti+1  % Nel caso live, la variabile di controllo sarebbe il numero di campioni già ricevuti
            predizione_processata2(index) = predizione_originale(index);
            buffer_precedenti(index) = predizione_originale(index); 
        
        elseif index > length(predizione_originale)-lunghezza_buffer_successivi % Questo controllo invece non sarebbe possibile
            predizione_processata2(index) = predizione_originale(index);
        
        else % Caso dove si è distante da inizio e fine
            nuovo_campione = predizione_originale(index);
            [valore_corretto, cambiamento_stato, buffer_precedenti, buffer_futuri, coda] = liveProcess2(buffer_precedenti, buffer_futuri, nuovo_campione, cambiamento_stato, coda, lunghezza_buffer_successivi);
            
            if cambiamento_stato ~= 2 && cambiamento_stato ~= -1 && valore_corretto ~= -1
                predizione_processata2(index) = valore_corretto;
            
            elseif valore_corretto == -1 && cambiamento_stato == 1 % Caso in cui si sta popolando il vettore_futuri
                %fprintf("Rilevato cambio valore, valutazione correttezza \n");
            
            elseif cambiamento_stato == 2 || cambiamento_stato == -1
                predizione_processata2(index-lunghezza_buffer_successivi:index) = valore_corretto;
                cambiamento_stato = 0;
            end
        end
    end

    if mostra_risultati_singoli
        figure
        plot(predizione_originale)
        hold on
        plot(predizione_processata)
        hold on
        plot(predizione_processata2)
        legend("Predizione originale", "Predizione processata", "Predizione processata 2")
    end

    metodo = "Prediction processate";
    set = "Test";

    [CM_prediction_processate, acc_prediction_processate, prec_prediction_processate, spec_prediction_processate, sens_prediction_processate, f1_prediction_processate] = evaluaClassificatore(label_test, predizione_processata, mostra_cm, classi, metodo, set);
    metodo = "Prediction processate 2";
    [CM_prediction_processate2, acc_prediction_processate2, prec_prediction_processate2, spec_prediction_processate2, sens_prediction_processate2, f1_prediction_processate2] = evaluaClassificatore(label_test, predizione_processata2, mostra_cm, classi, metodo, set);
    
    %% Rappresentazione dati processati
    if mostra_risultati_complessivi
        figure()
        subplot(4,1,1);
        plot(predizione_originale);
        title('Predizioni originali');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        subplot(4,1,2);
        plot(predizione_processata);
        title('Predizioni processate');
        xlabel('Campioni');
        ylabel('[a.u.]');

        subplot(4,1,3);
        plot(predizione_processata2);
        title('Predizioni processate 2');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        subplot(4,1,4);
        plot(label_test);
        title('Ground truth');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        % Collega gli assi verticali dei due subplot
        linkaxes([subplot(4,1,1), subplot(4,1,2), subplot(4,1,3), subplot(4,1,4)]);
    end
end




%%  Funzioni usate
function [CM, accuratezza, precisione, specificita, sensibilita, f1Score] = evaluaClassificatore(label_val, prediction, mostra_cm, classi, nomeMetodo, nomeSet)
    
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

    % Stampa delle metriche per ogni classe
    fprintf('\n---------------------------\n');
    fprintf('Accuratezza complessiva %s - %s: %.2f%%\n', nomeMetodo, nomeSet, accuratezza * 100);
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




function [correct_value, cambiamento_stato, buffer_precedenti, buffer_futuri, coda] = liveProcess2(buffer_precedenti, buffer_futuri, nuovo_campione, cambiamento_stato, coda, futuri_massimi)

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
                 correct_value = mode(buffer_futuri); % Restituisci il valore corretto dato dalla moda presente nel buffer_futuri, ovvero si considera corretto il valore più frequente dopo il cambio
                 cambiamento_stato = 2;               % Segnala termine controllo con risultati ok
                 buffer_precedenti = buffer_futuri(end-length(buffer_precedenti)+1:end);
                 buffer_futuri = [];                 % Resetta il buffer futuri
                 coda = 1;
            else
                %correct_value = buffer_precedenti(end); % Caso in cui i nuovi valori non sono coerenti, restituisce l'ultimo prima del cambiamento
                correct_value = mode(buffer_futuri); % Restituisci il valore corretto dato dalla moda presente nel buffer_futuri, ovvero si considera corretto il valore più frequente dopo il cambio
                cambiamento_stato = -1;
                buffer_futuri = [];
                coda = 1;
            end
        end

    else
        correct_value = nuovo_campione;
    end
end