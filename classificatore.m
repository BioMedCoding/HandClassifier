% Inizializzazione
clear 
close all
%clc

%% Impostazioni script e inizializzazione

% ======================== Parametri generali script ======================
mostra_grafici_segnali = false;                      % Mostra grafici relativi ai segnali pre-classificazione
mostra_cm = false;                                   % Mostra le CM dei vari classificatori
mostra_risultati_singoli = false;                   % Mostra confronto singolo classificatore - Ground Truth
mostra_risultati_complessivi = true;                 % Mostra confronto tutti i classificatori - Ground Truth
valore_apertura = 1;                                % Valore label apertura
valore_chiusura = 2;                                % Valore label chiusura
classi = {'Rilassata', 'Apertura','Chiusura'};      % Nomi assegnati alle classi

allena_svm = false;                                  % Esegui la sezione di addestramento e testing SVM
allena_lda = false;                                  % Esegui la sezione di addestramento e testing LDA
allena_rete_neurale = false;                         % Esegui la sezione di addestramento e testing rete neurale

numero_worker = 14;                                 % Numero di worker da usare per il parallel pool
caricamento_modelli = true;                         % Carica modelli già salvati quando usi il test set
salva_modelli = false;                               % Salva o meno i modelli allenati                           
percorso_salvataggio = "C:\Users\matte\Documents\GitHub\HandClassifier\Modelli_allenati"; % Percorso dove salvare i modelli

valuta_validation = false;                           % Esegui valutazione dei vari modelli sul validation set
valuta_training_completo = true;                    % Esegui valutazione dei vari modelli sul training set completo
valuta_test = true;                                 % Esegui valutazione dei vari modelli sul test set

prediction_parallel = false;                         % Esegui il comando predict usando il parfor (parallel pool)

warning('off', 'MATLAB:table:ModifiedAndSavedVarnames'); % Disabilita il warning relativo agli header
applica_postprocess = true;                         % Applica funzionid postprocess sul vettore di classificazione
% =========================================================================



% ======================== Parametri filtraggio ===========================
tipo_filtro = "cheby2";
f_sample = 2000;                                    % Frequenza campionamento
f_taglio_basso = 20;                                % Frequenza minima del passabanda
f_taglio_alta = 400;                                % Frequenza massima del passabanda
f_notch = 50;                                       % Frequenza del notch
f_envelope = 4;                                     % Frequenza inviluppo
percH = 1.3;                                        % Percentuale frequenza alta
visualisation = "no";                               % Mostra grafici filtraggio
% =========================================================================



%  ========================Parametri addestramento SVM ====================
svm_parameter_hypertuning = false;                  % Abilita hypertuning automatico dei parametri, sfruttando anche parallel pool
svm_calcolo_GPU = false;                            % Abilita l'addestramento dell'SVM tramite l'uso della GPU
ore_esecuzione_massime = 3;                         % Numero massimo di ora per cui continuare l'hypertuning automatico dei parametri

t_hyper = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto');

opts_hyp = struct('AcquisitionFunctionName', 'expected-improvement-plus', ...
    'UseParallel', true, ...
    'MaxObjectiveEvaluations', 30, ... 
    'Verbose', 2, ...                                              
    'ShowPlots', true, ...                                         
    'SaveIntermediateResults', true, ...                           
    'MaxTime', ore_esecuzione_massime*3600);                       

t_single = templateSVM('KernelFunction', 'polynomial', 'PolynomialOrder', 2);
%t_single = templateSVM('KernelFunction', 'rbf', 'BoxConstraint', 21.344, 'KernelScale', 0.55962); % Valore migliore trovato durante hypertuning automatico, con onevsone
coding_single = 'onevsall'; % onevsone, onevsall
% =========================================================================



%  ========================Parametri addestramento NN =====================
max_epoche = 300;                   % Numero massimo di epoche di allenamento della rete neurale
val_metrica_obiettivo = 0.0005;      % Metrica della rete di allenamento considerata accettabile per interrompere l'allenamento
train_function = 'trainscg';        % Funzione di training della rete neurale
layer1 = 5;                         % Numero di neuroni del primo layer
layer2 = 3;                         % Numero di neuroni del secondo layer
% =========================================================================



% ======================== Parametri addestramento LDA ====================
discrimType = 'linear'; % quadratic, diaglinear, diagquadratic, pseudolinear, pseudoquadratic
% =========================================================================

% ================= Avvio pool se necessario e non attivo =================
if prediction_parallel || svm_parameter_hypertuning
    if isempty(gcp('nocreate'))
            parpool('local', numero_worker); % Avvia in modalità processes
            %parpool('Threads')              % Avvia in modalità thread
    end
end
% =========================================================================

%% Import segnali

% Import segnali aperture
sig = readtable('Data\aperture.txt',"Delimiter", '\t');
%t_hyp = sig(:,1); % salva colonna dei tempi prima di cancellarla
sig(:,1) = [];
sig_aperture = table2array(sig);
% Import segnali chiusura
sig = readtable('Data\chiusure.txt',"Delimiter", '\t');
%t_hyp = sig(:,1); % salva colonna dei tempi prima di cancellarla
sig(:,1) = [];
sig_chiusura = table2array(sig);

% Concatenazione in un unica matrice segnale, prima apertura e poi chiusura
sig = [sig_aperture; sig_chiusura];

%% Filtraggio segnale

n_channel = length(sig(1,:));

% Pre-alloca le matrici
sig_filt= zeros(length(sig),n_channel);


% Filtraggio segnale
for i=1:n_channel
    sig_filt(:,i) = filter_general(sig(:,i),tipo_filtro,f_sample,"fL",f_taglio_basso,"fH",f_taglio_alta,"fN",f_notch,"visualisation",visualisation);
end

envelope = zeros(length(sig_filt),n_channel);
% Creazione inviluppo
for i=1:n_channel
    envelope(:,i) = filter_general(abs(sig_filt(:,i)),tipo_filtro,f_sample,"fH",f_envelope,"percH",percH);   
end

if mostra_grafici_segnali
    figure()
    plot(envelope)
    title('Inviluppo segnale grezzo');
    xlabel('Campioni');
    ylabel('[uV]');
end

% Standardizza i valori
envelope_std = (envelope-mean(envelope))./std(envelope);

% Istruzione per usare il segnale senza inviluppo
%envelope_std = (sig_filt-mean(sig_filt))./std(sig_filt);

if mostra_grafici_segnali
    figure()
    plot(envelope_std)
    hold on
end

%% Identificazione manuale delle azioni per etichettare

% APERTURA
% L'offset si è reso encessario a causa di un problema durante l'acquisizione manuale delle epoche di apertura
offset_apertura = 31660;
start_ind = [9769, 42531, 57715, 71952, 88176, 103467, 118417, 136170, 153994, 169935, 183932, 201840, 218319, 236499, 254929, 272381, 287480, 306104, 321158, 336730, 351220, 363781];  % Indici di inizio
end_ind = [20760, 47786, 63427, 78351, 94084, 108280, 124765, 143007, 159928, 176157, 190175, 208197, 224117, 244798, 262204, 278490, 296444, 312559, 326670, 345511, 355970, 369784];  % Indici di fine
start_ind_apertura = start_ind(2:end) - offset_apertura;
end_ind_apertura = end_ind(2:end) - offset_apertura;

n = end_ind_apertura(end);  % Lunghezza del vettore label aperture da creare
label_apertura = creaEtichetta(n, start_ind_apertura, end_ind_apertura, valore_apertura);


% CHIUSURA
start_indices_chiusura = [355753, 371329, 388113, 404343, 420094, 437420, 453354, 470347, 484509, 501457, 517611, 536106, 554011, 570095, 584037, 598100, 613799, 631512, 649291, 666419, 682022, 697089. 715550, 734320];  % Indici di inizio
end_ind_chiusura = [360648, 376921, 393662, 410653, 426676, 443944, 459423, 476138, 488613, 507541, 524189, 542986, 560615, 576562, 590635, 605092, 621867, 637444, 655965, 672968, 689190, 703900, 722697, 741775];  % Indici di fine

n = end_ind_chiusura(end);  % Lunghezza del vettore label chiusure da creare
label_chiusura = creaEtichetta(n, start_indices_chiusura, end_ind_chiusura, valore_chiusura);

if mostra_grafici_segnali
    plot(label_apertura)
    hold on
    plot(label_chiusura)
    title('Inviluppo segnale standardizzato complessivo e segmentazione azioni');
    xlabel('Campioni');
    ylabel('[a.u.]');
end

% Selezione delle porzioni migliori del segnale, in base ad analisi manuale
inizio_chiusura = 350700-12784;
end_chiusura = 661996;
label = [label_apertura(1:end_ind_apertura(end)) , label_chiusura(inizio_chiusura:end_chiusura)]';

% Taglio del segnale perché abbia la stessa lunghezza del label
envelope_std = envelope_std(1:length(label), :);

if mostra_grafici_segnali
    figure()
    plot(label)
    hold on
    plot(envelope_std)
    title('Segnale e label finali, post elaborazione');
    xlabel('Campioni');
    ylabel('[a.u.]');
end

%pause
%% Divisione training_validation set

% Creazione indici per training e test
num_samp = length(envelope_std(:,1));
index_random = randperm(num_samp);
training_idx = index_random(1:round(0.7*num_samp));
validation_idx = index_random(round(0.7*num_samp):end);

sig_train = envelope_std(training_idx,:);
sig_val = envelope_std(validation_idx,:);
label_train = label(training_idx,:);
label_val = label(validation_idx,:);

%% SVM - addestramento

if allena_svm

    if svm_parameter_hypertuning
        
        % Selezione se GPU o CPU
        if svm_calcolo_GPU
            % Trasferimento dei dati sulla GPU
            gpu_sig_train = gpuArray(sig_train);
            gpu_label_train = gpuArray(label_train);
            %svm_model = fitcecoc(gpu_sig_train,gpu_label_train, 'Learners', t_hyper, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opts_hyp, 'ClassNames', classi); 
            svm_model = fitcecoc(gpu_sig_train,gpu_label_train, 'Learners', t_hyper, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opts_hyp); 
            svm_model = gather(svm_model);
        else
            svm_model = fitcecoc(sig_train,label_train, 'Learners', t_hyper, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opts_hyp);
        end
    else    % Addestramento singolo
        if svm_calcolo_GPU
            % Trasferimento dei dati sulla GPU
            gpu_sig_train = gpuArray(sig_train);
            gpu_label_train = gpuArray(label_train);
            %svm_model = fitcecoc(gpu_sig_train,gpu_label_train, 'Learners', t_single, 'Coding', coding_single, 'ClassNames', classi); 
            svm_model = fitcecoc(gpu_sig_train,gpu_label_train, 'Learners', t_single, 'Coding', coding_single); 
            svm_model = gather(svm_model);
        else
            %svm_model = fitcecoc(sig_train,label_train, 'Learners', t_single, 'Coding', coding_single, 'ClassNames', classi);
            svm_model = fitcecoc(sig_train,label_train, 'Learners', t_single, 'Coding', coding_single);
        end
    end

    if salva_modelli
        % Salva il modello allenato in un file .mat
        save(fullfile(percorso_salvataggio, 'svm_model.mat'), 'svm_model');
    end  
end    
%% SVM - predizione e calcolo confusione matrix - VALIDATION

if caricamento_modelli
    load("Modelli_allenati\svm_model.mat");
end

if valuta_validation
    %
    
    if prediction_parallel
        numData = size(sig_val, 1);
        prediction_svm_validation = zeros(numData, 1);  % Preallocazione del vettore delle predizioni
        parfor i = 1:numData
            prediction_svm_validation(i) = predict(svm_model, sig_val(i, :));
        end
    else
        prediction_svm_validation = predict(svm_model,sig_val);
    end

    metodo = "SVM";
    set = "Validation";
    [CM_svm_validation, acc_svm_validation, prec_svm_validation, spec_svm_validation, sens_svm_validation, f1_svm_validation] = evaluaClassificatore(label_val, prediction_svm_validation, mostra_cm, classi, metodo, set); 
end
%pause
%% LDA - addestramento

if allena_lda
    %lda_model = fitcdiscr(sig_train,label_train,  'DiscrimType', discrimType, 'ClassNames', classi);
    lda_model = fitcdiscr(sig_train,label_train,  'DiscrimType', discrimType);
    
    % Visualizza i coefficienti del modello
    %Mdl.Coeffs(1,2).Const    % Costante del decision boundary
    %Mdl.Coeffs(1,2).Linear   % Coefficienti lineari per il decision boundary

    if salva_modelli
        % Salva il modello allenato in un file .mat
        %save('lda_model.mat', 'lda_model');
        save(fullfile(percorso_salvataggio, 'lda_model.mat'), 'lda_model');
    end
end    
%% LDA - predizione e calcolo confusion matrix - VALIDATION
    
if caricamento_modelli
    load("Modelli_allenati\lda_model.mat")
end    

if valuta_validation
    prediction_lda_validation = predict(lda_model, sig_val);

    metodo = "LDA";
    set = "Validation";
    [CM_lad_validation, acc_lda_validation, prec_lda_validation, spec_lda_validation, sens_lda_validation, f1_lda_validation] = evaluaClassificatore(label_val, prediction_lda_validation, mostra_cm, classi, metodo, set); 
end


%pause

%% Cosine similarity



%% Rete neurale - addestramento

if allena_rete_neurale
    % Definizione architettura - sistema completo
    % layers = [
    %     sequenceInputLayer(5)
    %     convolution1dLayer(5, 10, 'Padding', 'same')
    %     reluLayer
    %     maxPooling1dLayer(2, 'Stride', 2)
    %     convolution1dLayer(5, 20, 'Padding', 'same')
    %     reluLayer
    %     fullyConnectedLayer(3)
    %     softmaxLayer
    %     classificationLayer];
    % 
    % % Definizione delle opzioni di addestramento
    % options = trainingOptions('adam', ...
    %     'MaxEpochs', 10, ...
    %     'MiniBatchSize', 32, ...
    %     'ValidationData', {valData, valLabels}, ...
    %     'Plots', 'training-progress' ...
    %     'ExecutionEnvironment', 'gpu', ... % Imposta l'ambiente di esecuzione su 'gpu'
    %      'OutputFcn', @(info)showProgress(info));
    % 
    % % Addestramento del modello
    % net = trainNetwork(sig_train, label_train, layers, options);

    net = patternnet([layer1 layer2], 'trainscg', 'crossentropy');  % Esempio con due hidden layers
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'tansig';
    net.layers{3}.transferFcn = 'softmax';

    net.trainFcn = train_function;  % Conjugate gradient
    net.trainParam.epochs = max_epoche;
    net.trainParam.goal = val_metrica_obiettivo;
    %net.trainParam.useGPU = 'yes';  % Abilita l'uso della GPU

    % Comandi per gestire automaticamente la gestione dell'intero dataset
    % net.divideParam.trainRatio = 70/100;
    % net.divideParam.valRatio = 15/100;
    % net.divideParam.testRatio = 15/100;
    
    %sig_train = sig_train(:, 1:length(label_train));
    sig_train = sig_train'; % Trasposizione per come lavora matlab
    label_train = label_train+1;
    
    label_train = full(ind2vec(label_train'));  % Converti in formato one-hot e trasponi


    [net, tr] = train(net, sig_train, label_train);

    if salva_modelli
            % Salva il modello allenato in un file .mat
            %save('nn_model.mat', 'net');
            save(fullfile(percorso_salvataggio, 'nn_model.mat'), 'net');
    end
end

%% NN - predizione e calcolo confusion matrix - VALIDATION

if caricamento_modelli
    load("Modelli_allenati\nn_model.mat")
end

if valuta_validation
    sig_val = sig_val';
    label_val = label_val+1;
    label_val = full(ind2vec(label_val'));

    prediction_nn_validation = net(sig_val);
    prediction_nn_validation = vec2ind(prediction_nn_validation);  % Converti le probabilità in indici di classe

    % Ripristina label corrette
    label_val = vec2ind(label_val)';
    label_val = label_val-1;
    prediction_nn_validation = prediction_nn_validation'-1; % Riporta alle label convenzionali
    
    metodo = "NN";
    set = "Validation";
    [CM_nn_validation, acc_nn_validation, prec_nn_validation, spec_nn_validation, sens_nn_validation, f1_nn_validation] = evaluaClassificatore(label_val, prediction_nn_validation, mostra_cm, classi, metodo, set); 

end
%% Import dati test set

% Predizione usando test set
% Caricamento etichette dati

if valuta_test
    clear sig
    label_test = load("Data\activation_final.mat");
    label_test = label_test.activation_final';
    
    % Caricamento dati grezzi EMG
    sig = readtable('Data\test.txt',"Delimiter", '\t');
    t_hyp = sig(:,1);
    sig(:,1) = [];
    sig_test = table2array(sig);
    
    n_channel = length(sig_test(1,:));
    
    % Filtraggio segnale
    for i=1:n_channel
        sig_filt_test(:,i) = filter_general(sig_test(:,i),tipo_filtro,f_sample,"fL",f_taglio_basso,"fH",f_taglio_alta,"fN",f_notch,"visualisation",visualisation);
    end
    
    % Creazione inviluppo
    for i=1:n_channel
        envelope_test(:,i) = filter_general(abs(sig_filt_test(:,i)),tipo_filtro,f_sample,"fH",f_envelope,"percH",percH);  
    end
    
    if mostra_grafici_segnali
        figure()
        plot(envelope_test)
        title('Inviluppo segnale grezzo');
        xlabel('Campioni');
        ylabel('[uV]');
    end
    
    % Standardizza i valori
    envelope_std_test = (envelope_test-mean(envelope_test))./std(envelope_test);
    %envelope_std_test = envelope_std_test(1:end-1);
    
    label_test = [label_test; 0];
    
    if mostra_grafici_segnali
        figure()
        plot(envelope_std_test)
        hold on
    end
    
    if mostra_grafici_segnali
        figure()
        plot(label_test)
        hold on
        plot(envelope_std_test)
        title('Segnale e label finali, post elaborazione');
        xlabel('Campioni');
        ylabel('[a.u.]');
    end
    
    %pause
 end     
%% Rete neurale - funzionamento sul test set
if caricamento_modelli
    load("Modelli_allenati\nn_model.mat")
end

if valuta_test
        sig_test = envelope_std_test';
        label_test = label_test+1;
        label_test = full(ind2vec(label_test'));
    
        prediction_nn_test = net(sig_test);
        prediction_nn_test = vec2ind(prediction_nn_test);  % Converti le probabilità in indici di classe
    
        % Calcolo confusione matrix
        label_test = vec2ind(label_test)';
        label_test = label_test-1;
        prediction_nn_test = prediction_nn_test'-1; % Riporta alle label convenzionali

        metodo = "NN";
        set = "Test";
        [CM_nn_test, acc_nn_test, prec_nn_test, spec_nn_test, sens_nn_test, f1_nn_test] = evaluaClassificatore(label_test, prediction_nn_test, mostra_cm, classi, metodo, set); 
  
%% NN - rappresentazione dati di test
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
%% SVM - predizione dati test set
if caricamento_modelli
    load('Modelli_allenati\svm_model.mat')
end

    
%% Predizione usando test set
if valuta_test

    prediction_svm_test = predict(svm_model, envelope_std_test);
    
    metodo = "SVM";
    set = "Test";
    [CM_svm_test, acc_svm_test, prec_svm_test, spec_svm_test, sens_svm_test, f1_svm_test] = evaluaClassificatore(label_test, prediction_svm_test, mostra_cm, classi, metodo, set);

    %% SVM - rappresentazione dati di test
    if mostra_risultati_singoli
        figure;
        subplot(2,1,1);
        plot(prediction_test_svm);
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

%% LDA - predizione dati test set

if caricamento_modelli
    load('Modelli_allenati\lda_model.mat')
end

if valuta_test
    prediction_lda_test = predict(lda_model, envelope_std_test);

    metodo = "LDA";
    set = "Test";
    [CM_lda_test, acc_lda_test, prec_lda_test, spec_lda_test, sens_lda_test, f1_lda_test] = evaluaClassificatore(label_test, prediction_lda_test, mostra_cm, classi, metodo, set);

    %% LDA - Rappresentazione dati di test
    
    if mostra_risultati_singoli
        figure;
        subplot(2,1,1);
        plot(prediction_test_lda);
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

%% Rappresentazione complessivi test set

if valuta_test
    if mostra_risultati_complessivi
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
end

%% SVM - predizione dati TRAINING SET COMPLETO
if caricamento_modelli
    load('Modelli_allenati\svm_model.mat')
end

    % Predizione usando intero training+validation set
    %prediction_complete_svm = predict(svm_model, envelope_std);

if valuta_training_completo
    % Predizione usando training set complato

    prediction_svm_trainC = predict(svm_model, envelope_std);
    
    metodo = "SVM";
    set = "TrainingCompleto";
    [CM_svm_trainC, acc_svm_trainC, prec_svm_trainC, spec_svm_trainC, sens_svm_trainC, f1_svm_trainC] = evaluaClassificatore(label, prediction_svm_trainC, mostra_cm, classi, metodo, set);
    
    
    %% SVM - rappresentazione dati di training test completo
    if mostra_risultati_singoli
        figure;
        subplot(2,1,1);
        plot(prediction_svm_train_completo);
        title('Predizioni SVM - training test completo');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        subplot(2,1,2);
        plot(label);
        title('Ground truth');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        % Collega gli assi verticali dei due subplot
        linkaxes([subplot(2,1,1), subplot(2,1,2)]);
    
    end
end

%% Rete neurale - funzionamento sul training set completo
if caricamento_modelli
    load("Modelli_allenati\nn_model.mat")
end

if valuta_training_completo
        sig_completo = envelope_std';
        label = label+1;
        label = full(ind2vec(label'));
    
        prediction_nn_trainC = net(sig_completo);
        prediction_nn_trainC = vec2ind(prediction_nn_trainC);  % Converti le probabilità in indici di classe
    
        % Riporta il valore dei label a quello corretto
        label = vec2ind(label)';
        label = label-1;
        prediction_nn_trainC = prediction_nn_trainC'-1; % Riporta alle label convenzionali
        
        metodo = "NN";
        set = "TrainingCompleto";
        [CM_nn_trainC, acc_nn_trainC, prec_nn_trainC, spec_nn_trainC, sens_nn_trainC, f1_nn_trainC] = evaluaClassificatore(label, prediction_nn_trainC, mostra_cm, classi, metodo, set);
        
%% NN - rappresentazione dati training set completo
    if mostra_risultati_singoli
        figure;
        subplot(2,1,1);
        plot(prediction_nn_train_completo);
        title('Predizioni NN - training set completo');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        subplot(2,1,2);
        plot(label);
        title('Ground truth');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        % Collega gli assi verticali dei due subplot
        linkaxes([subplot(2,1,1), subplot(2,1,2)]);
    
    end
end

%% LDA - predizione dati training set completo

if caricamento_modelli
    load('Modelli_allenati\lda_model.mat')
end

if valuta_training_completo
    prediction_lda_trainC = predict(lda_model, envelope_std);
    
    metodo = "LDA";
    set = "TrainingCompleto";
    [CM_lda_trainC, acc_lda_trainC, prec_lda_trainC, spec_lda_trainC, sens_lda_trainC, f1_lda_trainC] = evaluaClassificatore(label, prediction_lda_trainC, mostra_cm, classi, metodo, set);

    %% LDA - Rappresentazione dati training test completo
    
    if mostra_risultati_singoli
        figure;
        subplot(2,1,1);
        plot(prediction_lda_train_completo);
        title('Predizioni LDA - training test completo');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        subplot(2,1,2);
        plot(label);
        title('Ground truth');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        % Collega gli assi verticali dei due subplot
        linkaxes([subplot(2,1,1), subplot(2,1,2)]);
    end
end

%% Rappresentazione complessiva training test completo

if valuta_training_completo
    if mostra_risultati_complessivi
        figure()
        subplot(4,1,1);
        plot(prediction_svm_trainC);
        title('Predizioni SVM - training completo');
        xlabel('Campioni');
        ylabel('[a.u.]');
    
        subplot(4,1,2);
        plot(prediction_nn_trainC);
        title('Predizioni NN - training completo');
        xlabel('Campioni');
        ylabel('[a.u.]');
    
        subplot(4,1,3);
        plot(prediction_lda_trainC);
        title('Predizioni LDA - training completo');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        subplot(4,1,4);
        plot(label);
        title('Ground truth');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        % Collega gli assi verticali dei due subplot
        linkaxes([subplot(4,1,1), subplot(4,1,2), subplot(4,1,3), subplot(4,1,4)]);
    end
end

%% SVM - applicazione postprocess per migliorare i risultati
if applica_postprocess
    % TENTATIVO CON FILTRO, NON SMUSSA ABBASTANZA
    % fc = 4;
    % ordine = 6;
    % wn = fc / (f_sample / 2); % Frequenza normalizzata
    % 
    % % Creazione del filtro passa-basso Butterworth
    % [b, a] = butter(ordine, wn, 'low');
    % 
    % prediction_filtered = round(filter(b, a, prediction_test_svm));
    % 
    % figure
    % plot(prediction_test_svm)
    % hold on
    % plot(prediction_filtered)
    
    % Tentativo usando funzione che potrebbe essere usata live

    %lunghezza_precedenti = 1;
    %lunghezza_futuri = 200;    % Equivale a 100 ms

    %buffer_precedenti = zeros(1,lunghezza_precedenti);
    %buffer_futuri= zeros(1,lunghezza_futuri);

    % La gestione dei bordi è affidata al ciclo sul vettore

    % Test funzione 1
    % Chiamata alla funzione per migliorare le etichette
    %prediction_processate = improveClassifierOutput(prediction_nn_test);

    predizione_originale = prediction_nn_test;
    
    coda = 1;
    lunghezza_buffer_precedenti = 400;
    lunghezza_buffer_successivi = 400;
    buffer_precedenti = [];
    buffer_futuri = [];
    cambiamento_stato = 0;

    % Creazione vettore di testing
    inizio = zeros(lunghezza_buffer_precedenti,1);
    segnale_nullo = zeros(lunghezza_buffer_precedenti,1);
    segnale_apertura = ones(50,1);

    %predizione_originale = vertcat(inizio, segnale_nullo, segnale_apertura, segnale_nullo, segnale_nullo);
    %predizione_originale = vertcat(inizio, segnale_nullo, segnale_apertura, segnale_apertura, segnale_apertura, segnale_apertura, segnale_apertura, segnale_nullo, segnale_nullo);
    %predizione_originale = vertcat(inizio, segnale_nullo, segnale_apertura, segnale_nullo, segnale_apertura, segnale_apertura, segnale_apertura, segnale_apertura, segnale_nullo, segnale_nullo);

    predizione_processata = zeros(length(predizione_originale),1);

    for index = 1:length(predizione_originale)  % Ciclo for per simulare dati acquisiti live
        if index < lunghezza_buffer_precedenti+1  % Nel caso live, la variabile di controllo sarebbe il numero di campioni già ricevuti
            predizione_processata(index) = predizione_originale(index);
            buffer_precedenti(index) = predizione_originale(index); 
        elseif index > length(predizione_originale)-lunghezza_buffer_successivi % Questo controllo invece non sarebbe possibile
            predizione_processata(index) = predizione_originale(index);
        else % Caso dove si è distante da inizio e fine
            nuovo_campione = predizione_originale(index);
            % liveProcess(buffer_precedenti, buffer_futuri, nuovo_campione, cambiamento_stato, coda, futuri_massimi)
            [valore_corretto, cambiamento_stato, buffer_precedenti, buffer_futuri, coda] = liveProcess2(buffer_precedenti, buffer_futuri, nuovo_campione, cambiamento_stato, coda, lunghezza_buffer_successivi);
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

    figure
    plot(predizione_originale)
    hold on
    plot(predizione_processata)
    legend("Predizione originale", "Predizione processata")

    % figure
    % plot(prediction_nn_test)
    % hold on
    % plot(prediction_processate + 3)
    % legend('Prediction Original', 'Prediction processata')

    metodo = "Prediction processate";
    set = "Test";
    [CM_prediction_processate, acc_prediction_processate, prec_prediction_processate, spec_prediction_processate, sens_prediction_processate, f1_prediction_processate] = evaluaClassificatore(label_test, predizione_processata, mostra_cm, classi, metodo, set);
    
    %% SVM - rappresentazione dati di test processati
    if mostra_risultati_complessivi
        figure()
        subplot(3,1,1);
        plot(predizione_originale);
        title('Predizioni originali');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        subplot(3,1,2);
        plot(predizione_processata);
        title('Predizioni processate');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        subplot(3,1,3);
        plot(label_test);
        title('Ground truth');
        xlabel('Campioni');
        ylabel('[a.u.]');
        
        % Collega gli assi verticali dei due subplot
        linkaxes([subplot(3,1,1), subplot(3,1,2), subplot(3,1,3)]);
    end
end

%% Funzioni usate
function vec = creaEtichetta(n, start_indices, end_indices, value)
    % Inizializza il vettore di zeri
    vec = zeros(1, n);
    
    % Imposta il valore numerico agli elementi tra gli indici di inizio e fine
    for i = 1:length(start_indices)
        vec(start_indices(i):end_indices(i)) = value;
    end
end

function [CM, accuratezza, precisione, specificita, sensibilita, f1Score] = evaluaClassificatore(label_val, prediction, mostra_cm, classi, nomeMetodo, nomeSet)
    % Calcolo della matrice di confusione
    CM = confusionmat(label_val, prediction);

    % Visualizzazione della confusion matrix se richiesto
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


% Da qui sotto sono tutti i vari test sulle funzioni di post, non guardare


% Questo funziona ma solo offline, con intero vettore
function labels = improveClassifierOutput(rawLabels)
    threshold = 400;        % Soglia di campioni
    n = length(rawLabels);  % Numero totale di etichette
    labels = rawLabels;     % Inizializza l'output con l'input
    
    i = 1;
    while i <= n
        % Trova l'inizio e la fine di una sottoserie
        start = i;
        while i <= n && labels(i) == labels(start)
            i = i + 1;
        end
        finish = i - 1;
        
        % Calcola la lunghezza della sottoserie
        len = finish - start + 1;
        
        % Se la sottoserie è più corta della soglia, modifica le etichette
        if len < threshold && start > 1
            labels(start:finish) = labels(start - 1);
        end
    end
    
    return
end

% Tentativo mio 2
function [correct_value, cambiamento_stato, buffer_precedenti, buffer_futuri, coda] = liveProcess(buffer_precedenti, buffer_futuri, nuovo_campione, cambiamento_stato, coda, futuri_massimi)

    if all(nuovo_campione ~= buffer_precedenti )
            cambiamento_stato = 1;
    end

    if cambiamento_stato
        if (coda) < futuri_massimi+1                    % Caso in cui il transitorio di osservazione non è ancora finito
            buffer_futuri(coda) = nuovo_campione;
            correct_value = -1;                       % In questo modo si segnala che è durante un transitorio
            buffer_precedenti = buffer_precedenti;    % Il buffer rimane inalterato
            coda = coda+1;                            % Aggiorna quello che è il contatore degli elementi in coda nel buffer futuro

        else                                          % Caso in cui il transitorio di osservazione è finito
            if all(buffer_futuri == buffer_futuri(1)) % Indica che tutti i nuovi valori sono coerenti
                 correct_value = buffer_futuri(1);     % Restituisci il valore corretto
                 cambiamento_stato = 2;                % Segnala termine controllo con risultati ok
                 buffer_precedenti = buffer_futuri(end-length(buffer_precedenti)+1:end);
                 buffer_futuri = [];                 % Resetta il buffer futuri
                 coda = 1;
            else
                correct_value = buffer_precedenti(end);
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
        if (coda) < futuri_massimi+1                    % Caso in cui il transitorio di osservazione non è ancora finito
            buffer_futuri(coda) = nuovo_campione;
            correct_value = -1;                       % In questo modo si segnala che è durante un transitorio
            buffer_precedenti = buffer_precedenti;    % Il buffer rimane inalterato
            coda = coda+1;                            % Aggiorna quello che è il contatore degli elementi in coda nel buffer futuro

        else                                          % Caso in cui il transitorio di osservazione è finito
            if all(buffer_futuri == buffer_futuri(1)) % Indica che tutti i nuovi valori sono coerenti
                 %correct_value = buffer_futuri(1);     % Restituisci il valore corretto
                 correct_value = mode(buffer_futuri);
                 cambiamento_stato = 2;                % Segnala termine controllo con risultati ok
                 buffer_precedenti = buffer_futuri(end-length(buffer_precedenti)+1:end);
                 buffer_futuri = [];                 % Resetta il buffer futuri
                 coda = 1;
            else
                correct_value = buffer_precedenti(end);
                cambiamento_stato = -1;
                buffer_futuri = [];
                coda = 1;
            end
        end

    else
        correct_value = nuovo_campione;
    end
end
