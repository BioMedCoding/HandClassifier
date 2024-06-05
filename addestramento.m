%% Inizializzazione
clc
clear 
close all


%% ======================== Parametri generali script ======================
generalParameters.mostra_grafici_segnali = false;                      % Mostra grafici relativi ai segnali pre-classificazione
generalParameters.mostra_segnale_per_canale = false;

generalParameters.percorso_dati_aperture = "Original_data/aperture.txt";
generalParameters.percorso_dati_chiusure = "Original_data/chiusure.txt";
generalParameters.percorso_label_training = "Prepared_data/label_dataset_completo";

generalParameters.valore_apertura = 1;                                % Valore label apertura
generalParameters.valore_chiusura = 2;                                % Valore label chiusura
generalParameters.classi = {'Rilassata', 'Apertura','Chiusura'};      % Nomi assegnati alle classi

generalParameters.dati_da_processare = true;                          % Se true carica dati grezzi e preprocessa, altrimenti carica direttamente dati e label già pronti
generalParameters.percorso_dati_preprocessati = "External_data/dataset_completo_preprocessato";
generalParameters.percorso_label_preprocessati = "External_data/label_dataset_completo_preprocessato";

generalParameters.applica_data_augmentation = false;
generalParameters.applica_data_augmentation_rumore_gaussiano = false;
generalParameters.livello_rumore_gaussiano = 0.01; 
generalParameters.applica_data_augmentation_ampiezza_dinamica = false;
generalParameters.amp_range = [0.7, 1.3];                             % Range di variazione da applicare
generalParameters.change_rate = 5;                                    % Velocità di cambiamento dell'ampiezza
% Finora meglio  amp_range = [0.7, 1.3]; | change_rate = 5;  

generalParameters.bilancia_classi = true;
generalParameters.metodo_bilanciamento_classi = 'smote';

generalParameters.allena_svm = true;                                  % Esegui la sezione di addestramento e testing SVM
generalParameters.allena_lda = true;                                  % Esegui la sezione di addestramento e testing LDA
generalParameters.allena_rete_neurale = true; 

generalParameters.rapporto_training_validation = 0.001;
% Con rapporto_training_validation = 0.00005 si usano 61 campioni di
% segnale, ovvero 30 ms di tempo di acquisizione

generalParameters.numero_worker = 14; 

generalParameters.salva_modelli = true;                               % Salva i modelli allenati      
generalParameters.salvataggio_train_val = false;                       % Salva matrici contenenti training e validation set
generalParameters.generalParameters.salvataggio_dataset_completo = false;               % Salva matrice contenente il dataset completo (già effettuato, pertanto disattivato)

%generalParameters.percorso_salvataggio_modelli = strcat("C:\Users\matte\Documents\GitHub\HandClassifier\Modelli_allenati_addestramento_dataAug","\",num2str(generalParameters.rapporto_training_validation)); % Percorso dove salvare i modelli
generalParameters.percorso_salvataggio_train_val = "C:\Users\matte\Documents\GitHub\HandClassifier\Prepared_data_low_data";
generalParameters.percorso_salvataggio_dataset_completo = "C:\Users\matte\Documents\GitHub\HandClassifier\Prepared_data_low_data";

warning('off', 'MATLAB:table:ModifiedAndSavedVarnames'); % Disabilita il warning relativo agli header

%% =========================================================================




%% ======================== Parametri filtraggio ===========================
filterParameters.tipo_filtro = "cheby2";
filterParameters.f_sample = 2000;                                    % Frequenza campionamento
filterParameters.f_taglio_basso = 20;                                % Frequenza minima del passabanda
filterParameters.f_taglio_alta = 400;                                % Frequenza massima del passabanda
filterParameters.f_notch = 50;                                       % Frequenza del notch
filterParameters.f_envelope = 4;                                     % Frequenza inviluppo
filterParameters.percH = 1.3;                                        % Percentuale frequenza alta
filterParameters.visualisation = "no";                               % Mostra grafici filtraggio
%% =========================================================================




%%  ========================Parametri addestramento SVM ====================

trainParametersSVM.hypertuning = false;
trainParametersSVM.useGPU = false;
trainParametersSVM.maxTrainHours = 0.25;
trainParametersSVM.t_hyper = templateSVM('KernelFunction', 'polynomial','PolynomialOrder', 3, 'KernelScale', 'auto');
trainParametersSVM.opts_hyp = struct('AcquisitionFunctionName', 'expected-improvement-plus', ...
    'UseParallel', true, ...
    'MaxObjectiveEvaluations', 300, ... 
    'Verbose', 2, ...                                              
    'ShowPlots', true, ...                                         
    'SaveIntermediateResults', true, ...                           
    'MaxTime', trainParametersSVM.maxTrainHours*3600);  
trainParametersSVM.t_single = templateSVM('KernelFunction', 'rbf', 'KernelScale', 10, 'Solver', 'ISDA');
trainParametersSVM.coding_single = 'onevsone';

trainParametersSVM.saveAllModel = false;
trainParametersSVM.savePath = "C:\Users\matte\Documents\GitHub\HandClassifier\Modelli_allenati\";
trainParametersSVM.trainingRepetitions = 2;

trainParametersSVM.showCM = false;
trainParametersSVM.showText = false;
trainParametersSVM.classes = {'Rilassata', 'Apertura','Chiusura'};

%% =========================================================================




%%  ========================Parametri addestramento NN =====================

%trainParametersNN.rete_custom = false;                % Abilita l'utilzzo della rete neurale custom made
trainParametersNN.savePath =  "C:\Users\matte\Documents\GitHub\HandClassifier\Modelli_allenati\";
trainParametersNN.trainingRepetitions = 2;
trainParametersNN.showCM = false;
trainParametersNN.showText = false;
trainParametersNN.saveAllModel = false;

trainParametersNN.layer = [5 3];

trainParametersNN.trainFunction = 'trainscg';        % Funzione di training della rete neurale
        % Le funzioni di training possibili sono 
        % 'trainlm': Rapida discesa, risultati finali simili ma nel test perde complatamente l'aperatura
        % 'traingd': Discesa del gradiente estremamente lenta, prestazioni scadenti dopo 500 epoche
        % 'traingdm': Discesa del gradiente più rapida, ma comunque lenta, prestazioni leggermente migliori ma comunque basse
        % 'traingdx': Discesa del gradiente rapida, da lavorarci meglio
        % 'trainrp': Discesa del gradiente molto rapida e ottima velocità, da lavorarci meglio
        % 'trainscg': Criterio benchmark, da seguire come riferimento per il momento

trainParametersNN.performanceFunction = 'crossentropy';


trainParametersNN.neuronFunction = 'logsig'; % 'tansig', 'logsig', 'purelin', 'poslin', 'softmax'
        % 'tansig': 85.09% accuretezza processata, 59.43 peggiore (sensibilità 3)
        % 'logsig': 86.7% accuratezza processata, 62.77 peggiore (sensibilità 3)
        % 'purelin': 82.76% accuratezza processata, 55 peggiore (sensibilità 3)
        % 'poslin': 75.12% accuretezza processata, 45.7 peggiore (sensibilità 3)
        % 'softmax': no, peggiore

trainParametersNN.outputLayerFunction = 'softmax';

trainParametersNN.maxEpochs = 500;                   % Numero massimo di epoche di allenamento della rete neurale
trainParametersNN.trainGoal = 0.000005;    % Metrica della rete di allenamento considerata accettabile per interrompere l'allenamento

trainParametersNN.maxFailure = 50;
      
trainParametersNN.lr = 0.02;                          % Learning rate
trainParametersNN.momentum = 0.9;                     % Momento durante l'allenamento

% Definizione del numero di neuroni per layer in base alla struttura
% selezionata della rete
%% =========================================================================




%% ======================== Parametri addestramento LDA ====================
trainParametersLDA.discriminant = 'quadratic';
%discrimType = 'quadratic'; % 'linear', 'quadratic', 'diaglinear', 'diagquadratic', 'pseudolinear','pseudoquadratic'
        % Le metriche qui sotto sono riferite al test set, senza postprocess. Accuratezza complessiva e poi metrica peggiore
        % 'linear': 75.22%, 48.64 sensibilità 3
        % 'quadratic': 87.81%, 72.65% sensibilità 2
        % 'diaglinear': 70.94%, 36.31 sensibilità 2
        % 'diagquadratic': 75.89%, 47.18 sensibilità 3
        % 'pseudolinear': 75.25%, 48.66 sensibilità 3
        % 'pseudoquadratic': 75.25%, 48.62 sensibilità 3

trainParametersLDA.saveAllModel = false;
trainParametersLDA.savePath = "C:\Users\matte\Documents\GitHub\HandClassifier\Modelli_allenati";
trainParametersLDA.trainingRepetitions = 2;

trainParametersLDA.showCM = false;
trainParametersLDA.showText = false;
trainParametersLDA.classes = {'Rilassata', 'Apertura','Chiusura'};

%% =========================================================================




%% ================= Avvio pool se necessario e non attivo =================
if trainParametersSVM.hypertuning
    if isempty(gcp('nocreate'))
            parpool('local', generalParameters.numero_worker); % Avvia in modalità processes
            %parpool('Threads')              % Avvia in modalità thread
    end
end
%% =========================================================================




%% Import segnali 


if generalParameters.dati_da_processare
    fprintf('\nInizio import e process dati \n')
    tic;
    % Import segnali aperture
    sig = readtable(generalParameters.percorso_dati_aperture,"Delimiter", '\t');
    %t_hyp = sig(:,1); % salva colonna dei tempi prima di cancellarla
    sig(:,1) = [];
    sig_aperture = table2array(sig);
    
    
    % Import segnali chiusure
    sig = readtable(generalParameters.percorso_dati_aperture,"Delimiter", '\t');
    %t_hyp = sig(:,1); % salva colonna dei tempi prima di cancellarla
    sig(:,1) = [];
    sig_chiusura = table2array(sig);
    
    % Concatenazione in un unica matrice segnale, prima apertura e poi chiusura
    sig = [sig_aperture; sig_chiusura];
    
    
    
    
    %% Filtraggio segnale
    
    n_channel = length(sig(1,:));
    sig_filt= zeros(length(sig),n_channel);
    
    % Filtraggio segnale
    for i=1:n_channel
        sig_filt(:,i) = filter_general(sig(:,i),filterParameters.tipo_filtro,filterParameters.f_sample,"fL",filterParameters.f_taglio_basso,"fH",filterParameters.f_taglio_alta,"fN",filterParameters.f_notch,"visualisation",filterParameters.visualisation);
    end
    
    if generalParameters.mostra_segnale_per_canale
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
        envelope(:,i) = filter_general(abs(sig_filt(:,i)),filterParameters.tipo_filtro,filterParameters.f_sample,"fH",filterParameters.f_envelope,"percH",filterParameters.percH);   
    end
    
    if generalParameters.mostra_grafici_segnali
        figure
        plot(envelope)
        title('Inviluppo segnale grezzo');
        xlabel('Campioni');
        ylabel('[uV]');
    end
    
    % Standardizza i valori
    
    envelope_std = (envelope-mean(envelope))./std(envelope);
    
    if generalParameters.mostra_segnale_per_canale
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
    
    % Riga per usare il segnale senza inviluppo
    %envelope_std = (sig_filt-mean(sig_filt))./std(sig_filt);
    
    if generalParameters.mostra_grafici_segnali
        figure
        plot(envelope_std)
        title('Segnale finale senza rumore')
    
    end
    
    % Salvataggio dataset completo
    % if salvataggio_dataset_completo
    %     save(strcat(generalParameters.percorso_salvataggio_dataset_completo,"/dataset_completo"), "envelope_std")
    % end
    
    %% Caricamento label segnale di training
    
    data = load(generalParameters.percorso_label_training);
    varNames = fieldnames(data);
    
    % Verifica che ci sia almeno una variabile nel file
    if ~isempty(varNames)
        % Estrai la prima variabile trovata e assegnala a 'test_signal'
        label_dataset_completo = data.(varNames{1});
    else
        disp('Nessuna variabile trovata nel file.');
    end
    
    % Taglio del segnale perché abbia la stessa lunghezza del label
    envelope_std = envelope_std(1:length(label_dataset_completo), :);
    
    if generalParameters.mostra_grafici_segnali
        figure
        plot(label_dataset_completo)
        hold on
        plot(envelope_std)
        title('Segnale e label finali, post elaborazione');
        xlabel('Campioni');
        ylabel('[a.u.]');
    end
    elapsed_time = toc;
    fprintf('   Termine import e process dati. Tempo necessario: %.2f secondi\n', elapsed_time);

else  % Caso di import diretto di dati e label già processati
    fprintf('\nInizio import dati \n')
    tic;
    envelope_std = load(generalParameters.percorso_dati_preprocessati);
    label_dataset_completo = load(generalParameters.percorso_label_preprocessati);
    elapsed_time = toc;
    fprintf('   Termine import dati. Tempo necessario: %.2f secondi\n', elapsed_time);
end
%% Applicazione data augmentation

if generalParameters.applica_data_augmentation
    fprintf('\nInizio Data augmentation \n')
    tic;
    % Salvataggio del segnale originale, per eventuali usi futuri
    envelope_std_originale = envelope_std;

    % Creazione variabile che permette di estendere questa sezione in
    % maniera modulare
    % Ogni step comunque agisce solo sui dati originali, per non sommare
    % le modifiche
    augmentedData = envelope_std;
    augmentedLabels = label_dataset_completo;

    if generalParameters.applica_data_augmentation_rumore_gaussiano
    
        envelope_std_gauss = zeros(size(envelope_std));
    
        for i = 1:size(envelope_std_gauss,1)
    
            envelope_std_gauss(i, :) = add_gaussian_noise(envelope_std(i, :), generalParameters.livello_rumore_gaussiano);

        end

        augmentedData = [augmentedData; envelope_std_gauss];
        augmentedLabels = [augmentedLabels; label_dataset_completo];
    
    end

    if generalParameters.applica_data_augmentation_ampiezza_dinamica
    
        varied_amplitude_signal = zeros(size(envelope_std));
    
        for i = 1:size(envelope_std,1)
        
            varied_amplitude_signal(i, :) = vary_amplitude(envelope_std(i, :), generalParameters.amp_range, generalParameters.change_rate);
    
        end
    
        augmentedData = [augmentedData; varied_amplitude_signal];
        augmentedLabels = [augmentedLabels; label_dataset_completo];
        
    end
    
    if generalParameters.mostra_grafici_segnali
        figure
        plot(augmentedData)
        title('Segnale finale senza rumore')
        hold on
        plot(augmentedLabels)
    end
    elapsed_time = toc;
    fprintf('   Termine Data augmentation. Tempo necessario: %.2f secondi\n', elapsed_time);
    % Assegna alla variabile i nomi previsti in seguito
    envelope_std = augmentedData;
    label_dataset_completo = augmentedLabels;
end

%pause

%% Bilanciamento classi

if generalParameters.bilancia_classi
    fprintf('\nInizio Bilanciamento classi \n')
    tic;
    [envelope_std, label_dataset_completo] = balance_dataset(envelope_std, label_dataset_completo, generalParameters.metodo_bilanciamento_classi);
    elapsed_time = toc;
    fprintf('   Termine Bilanciamento classi. Tempo necessario: %.2f secondi\n', elapsed_time);
end
%pause

%% Divisione training_validation set

fprintf('\nInizio Divisione training e validation set \n')
tic;
% Creazione indici per training e test
num_samp = length(envelope_std(:,1));
index_random = randperm(num_samp);
training_idx = index_random(1:round(generalParameters.rapporto_training_validation*num_samp));
validation_idx = index_random(round(generalParameters.rapporto_training_validation*num_samp):end);

sig_train = envelope_std(training_idx,:);
sig_val = envelope_std(validation_idx,:);
label_train = label_dataset_completo(training_idx,:);
label_val = label_dataset_completo(validation_idx,:);

% Salvataggio training e validation set
if generalParameters.salvataggio_train_val
    mkdir(generalParameters.percorso_salvataggio_train_val); 
    save(strcat(generalParameters.percorso_salvataggio_train_val,"/training_set"), "sig_train")
    save(strcat(generalParameters.percorso_salvataggio_train_val,"/validation_set"), "sig_val")
    save(strcat(generalParameters.percorso_salvataggio_train_val,"/label_train"), "label_train")
    save(strcat(generalParameters.percorso_salvataggio_train_val,"/label_val"), "label_val")
end
elapsed_time = toc;
fprintf('   Termine Divisione training e validation set. Tempo necessario: %.2f secondi\n', elapsed_time);
%% SVM - addestramento

if generalParameters.allena_svm
    [svm_models, best_svm_index,metrics_svm] = trainClassifier('svm',trainParametersSVM,sig_train,label_train,sig_val,label_val, generalParameters,filterParameters);
end


%% LDA - addestramento

if generalParameters.allena_lda
    [lda_models, best_lda_index,metrics_lda] = trainClassifier('lda',trainParametersLDA,sig_train,label_train,sig_val,label_val, generalParameters,filterParameters);
end  

%% Rete neurale - addestramento

if generalParameters.allena_rete_neurale
    [nn_models, best_nn_index,metrics_nn] = trainClassifier('patternNet',trainParametersNN,sig_train,label_train,sig_val,label_val, generalParameters,filterParameters);
end
    
    
    % if rete_custom

        % Al momento qui sotto non funziona ancora
        % % Parametri dei dati
        % numFeatures = 5;             % Numero di caratteristiche per ciascun timestep
        % timeSteps = 50;              % Numero di timesteps per sequenza
        % numTrainingSequences = 300;  % Numero di sequenze di addestramento
        % numValidationSequences = 100;% Numero di sequenze di validazione
        % numClasses = 3;              % Numero di classi per la classificazione
        % minSeqLength = 50;
        % 
        % % Trasposizione dei dati in formato [features, timeSteps, numSequences]
        % sig_train = reshape(sig_train', [numFeatures, timeSteps, numTrainingSequences]);
        % 
        % layers = [
        %     sequenceInputLayer(numFeatures, 'MinLength', timeSteps)
        %     convolution1dLayer(5, 10, 'Padding', 'same')
        %     reluLayer
        %     convolution1dLayer(5, 20, 'Padding', 'same')
        %     reluLayer
        %     fullyConnectedLayer(numClasses)
        %     softmaxLayer
        %     classificationLayer];
        % 
        % options_custom = trainingOptions('adam', ...
        %     'MaxEpochs', 10, ...
        %     'MiniBatchSize', 32, ...
        %     'ValidationData', {sig_val', label_val'}, ...
        %     'Plots', 'training-progress', ...
        %     'ExecutionEnvironment', 'gpu', ... % Imposta l'ambiente di esecuzione su 'gpu'
        %     'OutputFcn', @(info)showProgress(info));
        % 
        % % Addestramento del modello
        % net = trainNetwork(sig_train', label_train', layers, options_custom);

    % else

        % Versione 1 layer
%         if num_layer == 1
%             net = patternnet(layer1, train_function, 'crossentropy');  % Rete con un solo hidden layer
%             net.layers{1}.transferFcn = neuron_function;   % Funzione di trasferimento del layer nascosto
%             net.layers{2}.transferFcn = 'softmax';  % Funzione di trasferimento del layer di output
% 
%             net.trainFcn = train_function;  
%             net.trainParam.epochs = max_epoche;
%             net.trainParam.goal = val_metrica_obiettivo;
%             net.trainParam.max_fail = validation_check;
%             if strcmp(train_function, 'traingdx')  
%                 net.trainParam.lr = lr;
%                 net.trainParam.mc = momentum;
%             end
%         end
% 
%         % Versione 2 layer
%         if num_layer == 2
%             net = patternnet([layer1 layer2], train_function, 'crossentropy');  
%             net.layers{1}.transferFcn = neuron_function;
%             net.layers{2}.transferFcn = neuron_function;
%             net.layers{3}.transferFcn = 'softmax';
% 
%             net.trainFcn = train_function;  
%             net.trainParam.epochs = max_epoche;
%             net.trainParam.goal = val_metrica_obiettivo;
%             net.trainParam.max_fail = validation_check;
%             if strcmp(train_function, 'traingdx')
%                 net.trainParam.lr = lr;
%                 net.trainParam.mc = momentum;
%             end
%         end
% 
%         % Versione 3 layer
%         if num_layer == 3
%             net = patternnet([layer1 layer2 layer3], train_function, 'crossentropy');  
%             net.layers{1}.transferFcn = neuron_function;
%             net.layers{2}.transferFcn = neuron_function;
%             net.layers{3}.transferFcn = neuron_function;   
%             net.layers{4}.transferFcn = 'softmax';  
% 
%             net.trainFcn = train_function;  
%             net.trainParam.epochs = max_epoche;
%             net.trainParam.goal = val_metrica_obiettivo;
%             net.trainParam.max_fail = validation_check;
%             if strcmp(train_function, 'traingdx') 
%                 net.trainParam.lr = lr;
%                 net.trainParam.mc = momentum;
%             end
%         end
% 
%         % Comandi per gestire automaticamente la gestione dell'intero dataset
%         % net.divideParam.trainRatio = 70/100;
%         % net.divideParam.valRatio = 15/100;
%         % net.divideParam.testRatio = 15/100;
% 
%         % Adattamento segnali e label a formato rete
%         sig_train = sig_train'; % Trasposizione per adattare a necessità rete
%         label_train = label_train+1;
%         label_train = full(ind2vec(label_train'));  % Converti in formato one-hot e trasponi
% 
%         % Allena
%         [net, tr] = train(net, sig_train, label_train);
% 
%     end
%     elapsed_time = toc;
%     fprintf('   Termine allenamento NN. Tempo necessario: %.2f secondi\n', elapsed_time);
% 
%     if generalParameters.salva_modelli
%             mkdir(generalParameters.percorso_salvataggio_modelli); 
%             save(fullfile(generalParameters.percorso_salvataggio_modelli, 'nn_model.mat'), 'net');
%     end
% end


% Funzione per aggiungere rumore gaussiano
function augmented_signal = add_gaussian_noise(signal, noise_level)
    noise = noise_level * randn(size(signal));
    augmented_signal = signal + noise;
end

% Funzione per inserire una variazione temporale delle ampiezze
function augmented_signal = vary_amplitude(signal, amp_range, change_rate)
    % Genera un vettore di fattori di scaling che varia nel tempo
    time_vector = 1:length(signal);
    scaling_factors = amp_range(1) + (amp_range(2) - amp_range(1)) * 0.5 * (1 + sin(2 * pi * change_rate * time_vector / length(signal)));
    
    % Applica i fattori di scaling al segnale
    augmented_signal = signal .* scaling_factors;
end


% Funzione per andare a bilanciare il dataset, eccessivamente lenta per
% l'utilizzo
% function [balanced_data, balanced_labels] = balance_dataset(input_data, target_labels, method)
%     % Bilancia il dataset utilizzando downsampling o SMOTE
%     % input_data: matrice dei dati di input (ogni riga è un esempio)
%     % target_labels: vettore delle etichette dei target (classe di ciascun esempio)
%     % method: stringa che indica il metodo ('downsampling' o 'smote')
% 
%     % Trova le etichette uniche e conta il numero di esempi per ciascuna classe
%     [unique_labels, ~, label_indices] = unique(target_labels);
%     label_counts = histc(label_indices, 1:numel(unique_labels));
% 
%     % Trova la classe con il massimo numero di esempi
%     [max_count, max_index] = max(label_counts);
% 
%     % Inizializza le variabili di output
%     balanced_data = [];
%     balanced_labels = [];
% 
%     if strcmp(method, 'downsampling')
%         % Downsampling delle classi maggioritarie
%         min_count = min(label_counts);
%         for i = 1:numel(unique_labels)
%             % Indici degli esempi di questa classe
%             class_indices = find(label_indices == i);
% 
%             % Seleziona un sottoinsieme casuale di min_count esempi
%             selected_indices = randsample(class_indices, min_count);
%             balanced_data = [balanced_data; input_data(selected_indices, :)];
%             balanced_labels = [balanced_labels; target_labels(selected_indices)];
%         end
%     elseif strcmp(method, 'smote')
%         % Oversampling delle classi minoritarie utilizzando SMOTE
%         balanced_data = input_data;
%         balanced_labels = target_labels;
% 
%         for i = 1:numel(unique_labels)
%             if label_counts(i) < max_count
%                 % Numero di esempi da aggiungere per questa classe
%                 num_to_add = max_count - label_counts(i);
% 
%                 % Indici degli esempi di questa classe
%                 class_indices = find(label_indices == i);
% 
%                 % Verifica che ci siano almeno due esempi per interpolare
%                 if numel(class_indices) < 2
%                     error('Non ci sono abbastanza esempi per interpolare nella classe %d', unique_labels(i));
%                 end
% 
%                 % Genera esempi sintetici usando SMOTE
%                 for j = 1:num_to_add
%                     idx1 = class_indices(randi(numel(class_indices)));
%                     idx2 = class_indices(randi(numel(class_indices)));
% 
%                     % Verifica che gli indici siano diversi
%                     while idx1 == idx2
%                         idx2 = class_indices(randi(numel(class_indices)));
%                     end
% 
%                     % Genera un esempio sintetico interpolando tra i due esempi
%                     alpha = rand();
%                     synthetic_example = alpha * input_data(idx1, :) + (1 - alpha) * input_data(idx2, :);
% 
%                     % Aggiungi l'esempio sintetico al dataset
%                     balanced_data = [balanced_data; synthetic_example];
%                     balanced_labels = [balanced_labels; target_labels(idx1)];
%                 end
%             end
%         end
%     else
%         error('Metodo non riconosciuto. Usa "downsampling" o "smote".');
%     end
% end

% Funzione velocizzata per il bilanciamento del dataset
function [balanced_data, balanced_labels] = balance_dataset(input_data, target_labels, method)
    % Bilancia il dataset utilizzando downsampling o SMOTE
    % input_data: matrice dei dati di input (ogni riga è un esempio)
    % target_labels: vettore delle etichette dei target (classe di ciascun esempio)
    % method: stringa che indica il metodo ('downsampling' o 'smote')
    
    % Trova le etichette uniche e conta il numero di esempi per ciascuna classe
    [unique_labels, ~, label_indices] = unique(target_labels);
    label_counts = histc(label_indices, 1:numel(unique_labels));
    
    % Trova la classe con il massimo numero di esempi
    [max_count, ~] = max(label_counts);
    
    if strcmp(method, 'downsampling')
        % Downsampling delle classi maggioritarie
        min_count = min(label_counts);
        balanced_data = [];
        balanced_labels = [];
        for i = 1:numel(unique_labels)
            % Indici degli esempi di questa classe
            class_indices = find(label_indices == i);
            
            % Seleziona un sottoinsieme casuale di min_count esempi
            selected_indices = randsample(class_indices, min_count);
            balanced_data = [balanced_data; input_data(selected_indices, :)];
            balanced_labels = [balanced_labels; target_labels(selected_indices)];
        end

    elseif strcmp(method, 'smote')
        % Oversampling delle classi minoritarie utilizzando SMOTE
        balanced_data = input_data;
        balanced_labels = target_labels;
        
        for i = 1:numel(unique_labels)
            if label_counts(i) < max_count
                % Numero di esempi da aggiungere per questa classe
                num_to_add = max_count - label_counts(i);
                
                % Indici degli esempi di questa classe
                class_indices = find(label_indices == i);
                
                % Verifica che ci siano almeno due esempi per interpolare
                if numel(class_indices) < 2
                    error('Non ci sono abbastanza esempi per interpolare nella classe %d', unique_labels(i));
                end
                
                % Genera esempi sintetici usando SMOTE
                synthetic_examples = zeros(num_to_add, size(input_data, 2));
                for j = 1:num_to_add
                    idx1 = class_indices(randi(numel(class_indices)));
                    idx2 = class_indices(randi(numel(class_indices)));
                    
                    % Verifica che gli indici siano diversi
                    while idx1 == idx2
                        idx2 = class_indices(randi(numel(class_indices)));
                    end
                    
                    % Genera un esempio sintetico interpolando tra i due esempi
                    alpha = rand();
                    synthetic_examples(j, :) = alpha * input_data(idx1, :) + (1 - alpha) * input_data(idx2, :);
                end
                
                % Aggiungi gli esempi sintetici al dataset
                balanced_data = [balanced_data; synthetic_examples];
                balanced_labels = [balanced_labels; repmat(unique_labels(i), num_to_add, 1)];
            end
        end
    else
        error('Metodo non riconosciuto. Usa "downsampling" o "smote".');
    end
end