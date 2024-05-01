% Inizializzazione
clear all
close all
clc

% Da verificare:
% 1. Salvataggio modelli
% 2, Scegliere una logica per il load dei modelli

%% Impostazioni script e inizializzazione

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


% ======================== Parametri generali script ======================
mostra_grafici = true;                              % Mostra grafici prodotti durante il codice
valore_apertura = 1;                                % Valore label apertura
valore_chiusura = 2;                                % Valore label chiusura
classi = {'Rilassata', 'Apertura','Chiusura'};      % Nomi assegnati alle classi
esegui_svm = false;                                  % Esegui la sezione di addestramento e testing SVM
esegui_lda = false;                                  % Esegui la sezione di addestramento e testing LDA
esegui_rete_neurale = true;                         % Esegui la sezione di addestramento e testing rete neurale
numero_worker = 10;                                 % Numero di worker da usare per il parallel pool
caricamento_modelli = true;                         % Carica modelli già salvati quando usi il test set
salva_modelli = true;                               % Salva o meno i modelli allenati                           
percorso_salvataggio = "C:\Users\matte\Documents\GitHub\HandClassifier\Modelli_allenati"; % Percorso dove salvare i modelli
% =========================================================================


%  ========================Parametri addestramento SVM ====================
svm_parameter_hypertuning = false;                  % Abilita hypertuning automatico dei parametri, sfruttando anche parallel pool
svm_calcolo_GPU = false;                            % Abilita l'addestramento dell'SVM tramite l'uso della GPU
ore_esecuzione_massime = 3;                         % Numero massimo di ora per cui continuare l'hypertuning automatico dei parametri

t_hyp = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto');
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


% ======================== Parametri addestramento LDA ====================
discrimType = 'linear'; % quadratic, diaglinear, diagquadratic, pseudolinear, pseudoquadratic
% =========================================================================

%% Import segnali

% Import segnali aperture
sig = readtable('Data\aperture.txt',"Delimiter", '\t');
t_hyp = sig(:,1);
sig(:,1) = [];
sig_aperture = table2array(sig);
% Import segnali chiusura
sig = readtable('Data\chiusure.txt',"Delimiter", '\t');
t_hyp = sig(:,1);
sig(:,1) = [];
sig_chiusura = table2array(sig);

% Concatenazione in un unica matrice segnale, prima apertura e poi chiusura
sig = [sig_aperture; sig_chiusura];

%% Filtraggio segnale

n_channel = length(sig(1,:));

% Filtraggio segnale
for i=1:n_channel
    sig_filt(:,i) = filter_general(sig(:,i),tipo_filtro,f_sample,"fL",f_taglio_basso,"fH",f_taglio_alta,"fN",f_notch,"visualisation",visualisation);
end

% Creazione inviluppo
for i=1:n_channel
    envelope(:,i) = filter_general(abs(sig_filt(:,i)),tipo_filtro,f_sample,"fH",f_envelope,"percH",percH);   
end

if mostra_grafici
    figure()
    plot(envelope)
    title('Inviluppo segnale grezzo');
    xlabel('Campioni');
    ylabel('[uV]');
end

% Standardizza i valori
envelope_std = (envelope-mean(envelope))./std(envelope);

if mostra_grafici
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

if mostra_grafici
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

if mostra_grafici
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

if esegui_svm

    if svm_parameter_hypertuning
        % Attivazione parallelo pool
        if isempty(gcp('nocreate'))
            parpool('local', numero_worker); % Avvia un pool di lavoratori utilizzando il numero di core disponibile
        end
        % Selezione se GPU o CPU
        if svm_calcolo_GPU
            % Trasferimento dei dati sulla GPU
            gpu_sig_train = gpuArray(sig_train);
            gpu_label_train = gpuArray(label_train);
            %svm_model = fitcecoc(gpu_sig_train,gpu_label_train, 'Learners', t_hyp, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opts_hyp, 'ClassNames', classi); 
            svm_model = fitcecoc(gpu_sig_train,gpu_label_train, 'Learners', t_hyp, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opts_hyp); 
            svm_model = gather(svm_model);
        else
            svm_model = fitcecoc(sig_train,label_train, 'Learners', t_hyp, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opts_hyp);
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
    
    %% SVM - predizione e calcolo confusione matrix
    prediction_svm = predict(svm_model,sig_val);
    
    CM_svm = confusionmat(label_val,prediction_svm);
    
    if mostra_grafici
        figure()
        confusionchart(CM_svm, classi)
    end
    
    %% SVM - calcolo metriche per classe
    
    numClassi = size(CM_svm, 1);
    
    % Pre-allocazione memoria
    precision_svm = zeros(numClassi , 1);      % Quanti positivi lo erano davvero tra quelli positivi, TP/(TP+FP)
    sensibilita_svm = zeros(numClassi , 1);    % Quanti positivi effettivi sono stati identificati come tali, TP/(TP+FP)
    f1Score_svm = zeros(numClassi , 1);        % Media armonica di precision e sensibilità, utile quando la frequenza delle classi non è uguale
    accuretazza_svm = zeros(1, 1);             % Previsioni corrette rispetto al totale delle previsioni. Numero di predizioni corrette rispetto al numero totale di casi osservati. (TP + TN) / (TP+TN+FP+FN)
    specificita_svm = zeros(numClassi, 1);     % Quandi negativi effettivi sono stati identificati. TN / (TN+FP)
    
    for i=1:numClassi
        precision_svm(i) = CM_svm(i,i) / sum(CM_svm(:,i));
        sensibilita_svm(i) = CM_svm(i,i) / sum(CM_svm(i,:));
    
        % Punteggio f1 per classe 1
        if (precision_svm(i) + sensibilita_svm(i)) == 0
            f1score_svm(i) = 0;
        else
            f1Score_svm(i) = 2*(precision_svm(i)*sensibilita_svm(i) / (precision_svm(i) + sensibilita_svm(i)) );
        end
    
        % Calcolo della specificità per la classe i
        TN = sum(CM_svm(:)) - sum(CM_svm(i,:)) - sum(CM_svm(:,i)) + CM_svm(i,i);
        FP = sum(CM_svm(:,i)) - CM_svm(i,i);
        specificita_svm(i) = TN / (TN + FP);
    end
    
    % Calcolo dell'accuratezza complessiva
    total = sum(CM_svm(:));
    accuratezza_svm = sum(diag(CM_svm)) / total;
    
    % Stampa dell'accuratezza complessiva
    fprintf('Accuratezza complessiva SVM: %.2f\n', accuratezza_svm*100);
    
    
    % Stampa dell'intestazione delle metriche
    fprintf('%-10s %-12s %-12s %-12s %-12s\n', 'Classe', 'Precisione', 'Specificità', 'Sensibilità', 'F1 Score');
    
    % Stampa delle metriche per ogni classe
    for i = 1:numClassi
        fprintf('%-10s %-12.2f %-12.2f %-12.2f %-12.2f\n', char(classi(i)), precision_svm(i)*100, specificita_svm(i)*100, sensibilita_svm(i)*100, f1Score_svm(i)*100);
    end
    
    if salva_modelli
        % Salva il modello allenato in un file .mat
        save(fullfile(percorso_salvataggio, 'svm_model.mat'), 'svm_model');
    end             
end

%pause
%% LDA - addestramento

if esegui_lda
    %lda_model = fitcdiscr(sig_train,label_train,  'DiscrimType', discrimType, 'ClassNames', classi);
    lda_model = fitcdiscr(sig_train,label_train,  'DiscrimType', discrimType);
    
    % Visualizza i coefficienti del modello
    %Mdl.Coeffs(1,2).Const    % Costante del decision boundary
    %Mdl.Coeffs(1,2).Linear   % Coefficienti lineari per il decision boundary
    
    %% LDA - predizione e calcolo confusion matrix
    prediction_lda = predict(lda_model, sig_val);
    
    CM_lda = confusionmat(label_val,prediction_lda);
    if mostra_grafici
        figure()
        confusionchart(CM_lda, classi)
    end
    
    %% LDA - calcolo metriche per classe
    numClassi = size(CM_lda, 1);
    
    % Pre-allocazione memoria
    precision_lda = zeros(numClassi , 1);      % Quanti positivi lo erano davvero tra quelli positivi, TP/(TP+FP)
    sensibilita_lda = zeros(numClassi , 1);    % Quanti positivi effettivi sono stati identificati come tali, TP/(TP+FP)
    f1Score_lda = zeros(numClassi , 1);        % Media armonica di precision e sensibilità, utile quando la frequenza delle classi non è uguale
    accuretazza_lda = zeros(1, 1);             % Previsioni corrette rispetto al totale delle previsioni. Numero di predizioni corrette rispetto al numero totale di casi osservati. (TP + TN) / (TP+TN+FP+FN)
    specificita_lda = zeros(numClassi, 1);     % Quandi negativi effettivi sono stati identificati. TN / (TN+FP)
    
    for i=1:numClassi
        precision_lda(i) = CM_lda(i,i) / sum(CM_lda(:,i));
        sensibilita_lda(i) = CM_lda(i,i) / sum(CM_lda(i,:));
    
        % Punteggio f1 per classe 1
        if (precision_lda(i) + sensibilita_lda(i)) == 0
            f1score_lda(i) = 0;
        else
            f1Score_lda(i) = 2*(precision_lda(i)*sensibilita_lda(i) / (precision_lda(i) + sensibilita_lda(i)) );
        end
    
        % Calcolo della specificità per la classe i
        TN = sum(CM_lda(:)) - sum(CM_lda(i,:)) - sum(CM_lda(:,i)) + CM_lda(i,i);
        FP = sum(CM_lda(:,i)) - CM_lda(i,i);
        specificita_lda(i) = TN / (TN + FP);
    end
    
    % Calcolo dell'accuratezza complessiva
    total = sum(CM_lda(:));
    accuratezza_lda = sum(diag(CM_lda)) / total;
    
    % Stampa dell'accuratezza complessiva
    fprintf('Accuratezza complessiva LDA: %.2f\n', accuratezza_lda);
    
    
    % Stampa dell'intestazione delle metriche
    fprintf('%-10s %-12s %-12s %-12s %-12s\n', 'Classe', 'Precisione', 'Specificità', 'Sensibilità', 'F1 Score');
    
    % Stampa delle metriche per ogni classe
    for i = 1:numClassi
        fprintf('%-10d %-12.2f %-12.2f %-12.2f %-12.2f\n', i, precision_lda(i), specificita_lda(i), sensibilita_lda(i), f1Score_lda(i));
    end
    
    if salva_modelli
        % Salva il modello allenato in un file .mat
        %save('lda_model.mat', 'lda_model');
        save(fullfile(percorso_salvataggio, 'lda_model.mat'), 'lda_model');
    end
end


%pause

%% Cosine similarity



%% Rete neurale - addestramento

if esegui_rete_neurale
    % Definizione architettura - chat stupido
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

    net = patternnet([50 30], 'trainscg', 'crossentropy');  % Esempio con due hidden layers
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'tansig';
    net.layers{3}.transferFcn = 'softmax';

    net.trainFcn = 'trainscg';  % Conjugate gradient
    net.trainParam.epochs = 300;
    net.trainParam.goal = 0.0005;
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

    %% Inference rete
    sig_val = sig_val';
    label_val = label_val+1;
    label_val = full(ind2vec(label_val'));

    predictions_nn = net(sig_val);
    predictions_nn = vec2ind(predictions_nn);  % Converti le probabilità in indici di classe

    % Calcolo confusione matrix
    CM_nn = confusionmat(vec2ind(label_val), predictions_nn);
    
    if mostra_grafici
            figure()
            confusionchart(CM_nn, classi)
    end
    
    %% Rete neurale - calcolo metriche per classe
        numClassi = size(CM_nn, 1);
        
        % Pre-allocazione memoria
        precision_nn = zeros(numClassi , 1);      % Quanti positivi lo erano davvero tra quelli positivi, TP/(TP+FP)
        sensibilita_nn = zeros(numClassi , 1);    % Quanti positivi effettivi sono stati identificati come tali, TP/(TP+FP)
        f1Score_nn = zeros(numClassi , 1);        % Media armonica di precision e sensibilità, utile quando la frequenza delle classi non è uguale
        accuretazza_nn = zeros(1, 1);             % Previsioni corrette rispetto al totale delle previsioni. Numero di predizioni corrette rispetto al numero totale di casi osservati. (TP + TN) / (TP+TN+FP+FN)
        specificita_nn = zeros(numClassi, 1);     % Quandi negativi effettivi sono stati identificati. TN / (TN+FP)
        
        for i=1:numClassi
            precision_nn(i) = CM_nn(i,i) / sum(CM_nn(:,i));
            sensibilita_nn(i) = CM_nn(i,i) / sum(CM_nn(i,:));
        
            % Punteggio f1 per classe 1
            if (precision_nn(i) + sensibilita_nn(i)) == 0
                f1score_nn(i) = 0;
            else
                f1Score_nn(i) = 2*(precision_nn(i)*sensibilita_nn(i) / (precision_nn(i) + sensibilita_nn(i)) );
            end
        
            % Calcolo della specificità per la classe i
            TN = sum(CM_nn(:)) - sum(CM_nn(i,:)) - sum(CM_nn(:,i)) + CM_nn(i,i);
            FP = sum(CM_nn(:,i)) - CM_nn(i,i);
            specificita_nn(i) = TN / (TN + FP);
        end
        
        % Calcolo dell'accuratezza complessiva
        total = sum(CM_nn(:));
        accuratezza_nn = sum(diag(CM_nn)) / total;
        
        % Stampa dell'accuratezza complessiva
        fprintf('Accuratezza complessiva rete neurale: %.2f\n', accuratezza_nn);
        
        
        % Stampa dell'intestazione delle metriche
        fprintf('%-10s %-12s %-12s %-12s %-12s\n', 'Classe', 'Precisione', 'Specificità', 'Sensibilità', 'F1 Score');
        
        % Stampa delle metriche per ogni classe
        for i = 1:numClassi
            fprintf('%-10d %-12.2f %-12.2f %-12.2f %-12.2f\n', i, precision_nn(i), specificita_nn(i), sensibilita_nn(i), f1Score_nn(i));
        end
        
        if salva_modelli
            % Salva il modello allenato in un file .mat
            %save('nn_model.mat', 'net');
            save(fullfile(percorso_salvataggio, 'nn_model.mat'), 'net');
        end
end
%% Import dati test set

% Predizione usando test set
% Caricamento etichette dati

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

if mostra_grafici
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

if mostra_grafici
    figure()
    plot(envelope_std_test)
    hold on
end

if mostra_grafici
    figure()
    plot(label_test)
    hold on
    plot(envelope_std_test)
    title('Segnale e label finali, post elaborazione');
    xlabel('Campioni');
    ylabel('[a.u.]');
end

%pause

%% Rete neurale - funzionamento sul test set
    sig_test = envelope_std_test';
    label_test = label_test+1;
    label_test = full(ind2vec(label_test'));

    predictions_nn = net(sig_test);
    predictions_nn = vec2ind(predictions_nn);  % Converti le probabilità in indici di classe

    % Calcolo confusione matrix
    CM_nn = confusionmat(vec2ind(label_test), predictions_nn);
    
    if mostra_grafici
            figure()
            confusionchart(CM_nn, classi)
    end
    
    %% Rete neurale - calcolo metriche per classe sul TEST
        numClassi = size(CM_nn, 1);
        
        % Pre-allocazione memoria
        precision_nn = zeros(numClassi , 1);      % Quanti positivi lo erano davvero tra quelli positivi, TP/(TP+FP)
        sensibilita_nn = zeros(numClassi , 1);    % Quanti positivi effettivi sono stati identificati come tali, TP/(TP+FP)
        f1Score_nn = zeros(numClassi , 1);        % Media armonica di precision e sensibilità, utile quando la frequenza delle classi non è uguale
        accuretazza_nn = zeros(1, 1);             % Previsioni corrette rispetto al totale delle previsioni. Numero di predizioni corrette rispetto al numero totale di casi osservati. (TP + TN) / (TP+TN+FP+FN)
        specificita_nn = zeros(numClassi, 1);     % Quandi negativi effettivi sono stati identificati. TN / (TN+FP)
        
        for i=1:numClassi
            precision_nn(i) = CM_nn(i,i) / sum(CM_nn(:,i));
            sensibilita_nn(i) = CM_nn(i,i) / sum(CM_nn(i,:));
        
            % Punteggio f1 per classe 1
            if (precision_nn(i) + sensibilita_nn(i)) == 0
                f1score_nn(i) = 0;
            else
                f1Score_nn(i) = 2*(precision_nn(i)*sensibilita_nn(i) / (precision_nn(i) + sensibilita_nn(i)) );
            end
        
            % Calcolo della specificità per la classe i
            TN = sum(CM_nn(:)) - sum(CM_nn(i,:)) - sum(CM_nn(:,i)) + CM_nn(i,i);
            FP = sum(CM_nn(:,i)) - CM_nn(i,i);
            specificita_nn(i) = TN / (TN + FP);
        end
        
        % Calcolo dell'accuratezza complessiva
        total = sum(CM_nn(:));
        accuratezza_nn = sum(diag(CM_nn)) / total;
        
        % Stampa dell'accuratezza complessiva
        fprintf('Accuratezza complessiva rete neurale: %.2f\n', accuratezza_nn);
        
        
        % Stampa dell'intestazione delle metriche
        fprintf('%-10s %-12s %-12s %-12s %-12s\n', 'Classe', 'Precisione', 'Specificità', 'Sensibilità', 'F1 Score');
        
        % Stampa delle metriche per ogni classe
        for i = 1:numClassi
            fprintf('%-10d %-12.2f %-12.2f %-12.2f %-12.2f\n', i, precision_nn(i), specificita_nn(i), sensibilita_nn(i), f1Score_nn(i));
        end
        
        if salva_modelli
            % Salva il modello allenato in un file .mat
            save('nn_model.mat', 'net');
        end

%% SVM - predizione dati test set
% Caricamento modello allenato
load('svm_model.mat')

% Predizione usando intero training+validation set
%prediction_complete_svm = predict(svm_model, envelope_std);

% Predizione usando test set
prediction_test_svm = predict(svm_model, envelope_std_test);

CM_svm_test = confusionmat(label_test,prediction_test_svm);
    
    if mostra_grafici
        figure()
        confusionchart(CM_svm_test, classi, classi)
    end

%% SVM - calcolo metriche per classe sul TEST SET
    
numClassi = size(CM_svm_test, 1);

% Pre-allocazione memoria
precision_svm_test = zeros(numClassi , 1);      % Quanti positivi lo erano davvero tra quelli positivi, TP/(TP+FP)
sensibilita_svm_test = zeros(numClassi , 1);    % Quanti positivi effettivi sono stati identificati come tali, TP/(TP+FP)
f1Score_svm_test = zeros(numClassi , 1);        % Media armonica di precision e sensibilità, utile quando la frequenza delle classi non è uguale
accuretazza_svm_test = zeros(1, 1);             % Previsioni corrette rispetto al totale delle previsioni. Numero di predizioni corrette rispetto al numero totale di casi osservati. (TP + TN) / (TP+TN+FP+FN)
specificita_svm_test = zeros(numClassi, 1);     % Quandi negativi effettivi sono stati identificati. TN / (TN+FP)

for i=1:numClassi
    precision_svm_test(i) = CM_svm_test(i,i) / sum(CM_svm_test(:,i));
    sensibilita_svm_test(i) = CM_svm_test(i,i) / sum(CM_svm_test(i,:));

    % Punteggio f1 per classe 1
    if (precision_svm_test(i) + sensibilita_svm_test(i)) == 0
        f1score_svm_test(i) = 0;
    else
        f1Score_svm_test(i) = 2*(precision_svm_test(i)*sensibilita_svm_test(i) / (precision_svm_test(i) + sensibilita_svm_test(i)) );
    end

    % Calcolo della specificità per la classe i
    TN = sum(CM_svm_test(:)) - sum(CM_svm_test(i,:)) - sum(CM_svm_test(:,i)) + CM_svm_test(i,i);
    FP = sum(CM_svm_test(:,i)) - CM_svm_test(i,i);
    specificita_svm_test(i) = TN / (TN + FP);
end

% Calcolo dell'accuratezza complessiva
total = sum(CM_svm_test(:));
accuratezza_svm_test = sum(diag(CM_svm_test)) / total;

% Stampa dell'accuratezza complessiva
fprintf('Accuratezza complessiva: %.2f\n', accuratezza_svm_test*100);


% Stampa dell'intestazione delle metriche
fprintf('%-10s %-12s %-12s %-12s %-12s\n', 'Classe', 'Precisione', 'Specificità', 'Sensibilità', 'F1 Score');

% Stampa delle metriche per ogni classe
for i = 1:numClassi
    fprintf('%-10s %-12.2f %-12.2f %-12.2f %-12.2f\n', char(classi(i)), precision_svm_test(i)*100, specificita_svm_test(i)*100, sensibilita_svm_test(i)*100, f1Score_svm_test(i)*100);
end

%% SVM - rappresentazione dati di test
if mostra_grafici
    figure;
    subplot(2,1,1);
    plot(prediction_test_svm);
    title('Predizioni');
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
%% SVM - applicazione postprocess per migliorare i risultati

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

% tempo_buffer = 500; % Lunghezza del buffer in millisecondi
% tempo_minimo = 300; 
% fattore_moltiplicativo = f_sample/1000;
% buffer_valori = zeros(fattore_moltiplicativo*tempo_buffer,1);
% 
% prediction_processata = zeros(length(prediction_test_svm),1);
% 
% % Processa ogni valore del vettore
% for i = 1:length(prediction_processata)
%     [valoreCorretto, bufferValori] = LivePostProcess(buffer_valori, prediction_test_svm(i), fattore_moltiplicativo*tempo_minimo);
%     prediction_processata(i) = valoreCorretto;
% end

% Tentativo con funzione postuma
lunghezza_minima = 400; % Lunghezza minima dove non corregge serie di valori, in campioni
prediction_processata = correggiVettore(prediction_test_svm, lunghezza_minima);

figure()
plot(prediction_test_svm)
hold on
plot(prediction_processata)
legend('Prediction Original', 'Prediction processata')

CM_svm_test_processato = confusionmat(label_test,prediction_processata);
    
    if mostra_grafici
        figure()
        confusionchart(CM_svm_test_processata, classi)
    end

%% SVM - calcolo metriche per classe sul TEST SET PROCESSATO
    
    numClassi = size(CM_svm_test_processata, 1);
    
    % Pre-allocazione memoria
    precision_svm_test = zeros(numClassi , 1);      % Quanti positivi lo erano davvero tra quelli positivi, TP/(TP+FP)
    sensibilita_svm_test = zeros(numClassi , 1);    % Quanti positivi effettivi sono stati identificati come tali, TP/(TP+FP)
    f1Score_svm_test = zeros(numClassi , 1);        % Media armonica di precision e sensibilità, utile quando la frequenza delle classi non è uguale
    accuretazza_svm_test = zeros(1, 1);             % Previsioni corrette rispetto al totale delle previsioni. Numero di predizioni corrette rispetto al numero totale di casi osservati. (TP + TN) / (TP+TN+FP+FN)
    specificita_svm_test = zeros(numClassi, 1);     % Quandi negativi effettivi sono stati identificati. TN / (TN+FP)
    
    for i=1:numClassi
        precision_svm_test(i) = CM_svm_test_processata(i,i) / sum(CM_svm_test_processata(:,i));
        sensibilita_svm_test(i) = CM_svm_test_processata(i,i) / sum(CM_svm_test_processata(i,:));
    
        % Punteggio f1 per classe 1
        if (precision_svm_test(i) + sensibilita_svm_test(i)) == 0
            f1score_svm_test(i) = 0;
        else
            f1Score_svm_test(i) = 2*(precision_svm_test(i)*sensibilita_svm_test(i) / (precision_svm_test(i) + sensibilita_svm_test(i)) );
        end
    
        % Calcolo della specificità per la classe i
        TN = sum(CM_svm_test_processata(:)) - sum(CM_svm_test_processata(i,:)) - sum(CM_svm_test_processata(:,i)) + CM_svm_test_processata(i,i);
        FP = sum(CM_svm_test_processata(:,i)) - CM_svm_test_processata(i,i);
        specificita_svm_test(i) = TN / (TN + FP);
    end
    
    % Calcolo dell'accuratezza complessiva
    total = sum(CM_svm_test_processata(:));
    accuratezza_svm_test = sum(diag(CM_svm_test_processata)) / total;
    
    % Stampa dell'accuratezza complessiva
    fprintf('Accuratezza complessiva: %.2f\n', accuratezza_svm_test*100);
    
    
    % Stampa dell'intestazione delle metriche
    fprintf('%-10s %-12s %-12s %-12s %-12s\n', 'Classe', 'Precisione', 'Specificità', 'Sensibilità', 'F1 Score');
    
    % Stampa delle metriche per ogni classe
    for i = 1:numClassi
        fprintf('%-10s %-12.2f %-12.2f %-12.2f %-12.2f\n', char(classi(i)), precision_svm_test(i)*100, specificita_svm_test(i)*100, sensibilita_svm_test(i)*100, f1Score_svm_test(i)*100);
    end

%% SVM - rappresentazione dati di test processati
if mostra_grafici
    figure;
    subplot(3,1,1);
    plot(prediction_test_svm);
    title('Predizioni originali');
    xlabel('Campioni');
    ylabel('[a.u.]');
    
    subplot(3,1,2);
    plot(prediction_processata);
    title('Predizioni processate');
    xlabel('Campioni');
    ylabel('[a.u.]');
    
    subplot(3,1,3);
    plot(label_test);
    title('Ground truth');
    xlabel('Campioni');
    ylabel('[a.u.]');
    
    % Collega gli assi verticali dei due subplot
    linkaxes([subplot(2,1,1), subplot(2,1,2)]);
end

%% LDA - predizione dati test set

if caricamento_modelli
    load('lda_model.mat')
end
prediction_test_lda = predict(lda_model, envelope_std_test);

CM_lda_test = confusionmat(label_test,prediction_test_lda);
    
    if mostra_grafici
        figure()
        confusionchart(CM_lda_test, classi)
    end

%% LDA - calcolo metriche per classe sul TEST SET
    
    numClassi = size(CM_lda_test, 1);
    
    % Pre-allocazione memoria
    precision_lda_test = zeros(numClassi , 1);      % Quanti positivi lo erano davvero tra quelli positivi, TP/(TP+FP)
    sensibilita_lda_test = zeros(numClassi , 1);    % Quanti positivi effettivi sono stati identificati come tali, TP/(TP+FP)
    f1Score_lda_test = zeros(numClassi , 1);        % Media armonica di precision e sensibilità, utile quando la frequenza delle classi non è uguale
    accuretazza_lda_test = zeros(1, 1);             % Previsioni corrette rispetto al totale delle previsioni. Numero di predizioni corrette rispetto al numero totale di casi osservati. (TP + TN) / (TP+TN+FP+FN)
    specificita_lda_test = zeros(numClassi, 1);     % Quandi negativi effettivi sono stati identificati. TN / (TN+FP)
    
    for i=1:numClassi
        precision_lda_test(i) = CM_lda_test(i,i) / sum(CM_lda_test(:,i));
        sensibilita_lda_test(i) = CM_lda_test(i,i) / sum(CM_lda_test(i,:));
    
        % Punteggio f1 per classe 1
        if (precision_lda_test(i) + sensibilita_lda_test(i)) == 0
            f1score_lda_test(i) = 0;
        else
            f1Score_lda_test(i) = 2*(precision_lda_test(i)*sensibilita_lda_test(i) / (precision_lda_test(i) + sensibilita_lda_test(i)) );
        end
    
        % Calcolo della specificità per la classe i
        TN = sum(CM_lda_test(:)) - sum(CM_lda_test(i,:)) - sum(CM_lda_test(:,i)) + CM_lda_test(i,i);
        FP = sum(CM_lda_test(:,i)) - CM_lda_test(i,i);
        specificita_lda_test(i) = TN / (TN + FP);
    end
    
    % Calcolo dell'accuratezza complessiva
    total = sum(CM_lda_test(:));
    accuratezza_lda_test = sum(diag(CM_lda_test)) / total;
    
    % Stampa dell'accuratezza complessiva
    fprintf('Accuratezza complessiva: %.2f\n', accuratezza_lda_test*100);
    
    
    % Stampa dell'intestazione delle metriche
    fprintf('%-10s %-12s %-12s %-12s %-12s\n', 'Classe', 'Precisione', 'Specificità', 'Sensibilità', 'F1 Score');
    
    % Stampa delle metriche per ogni classe
    for i = 1:numClassi
        fprintf('%-10s %-12.2f %-12.2f %-12.2f %-12.2f\n', char(classi(i)), precision_lda_test(i)*100, specificita_lda_test(i)*100, sensibilita_lda_test(i)*100, f1Score_lda_test(i)*100);
    end

%% LDA - Rappresentazione dati di test

if mostra_grafici
    figure;
    subplot(2,1,1);
    plot(prediction_test_lda);
    title('Predizioni');
    xlabel('Campioni');
    ylabel('[a.u.]');
    
    subplot(2,1,2);
    plot(label_test);
    title('Ground truth e predizioni');
    xlabel('Campioni');
    ylabel('[a.u.]');
    
    % Collega gli assi verticali dei due subplot
    linkaxes([subplot(2,1,1), subplot(2,1,2)]);
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


% Qui sotto si hanno tentativi di implementazioni di funzioni di postproces, anche live
function [valoreCorretto, bufferValori] = LivePostProcess(bufferValori, nuovoValore, soglia)
    
    % Aggiungi il nuovo valore al buffer togliendo il valore più vecchio per fargli posto
    bufferValori = vertcat(bufferValori(2:end), nuovoValore);    

    if all(nuovoValore ~= bufferValori(end-soglia+1:end))
        valoreCorretto = bufferValori(end-2);
    else
        valoreCorretto = nuovoValore;
    end
    
    % Restituisci il valore corretto e il buffer aggiornato
    bufferValori(end) = valoreCorretto;
end

function output = correggiVettore(inputVec, minLen)
    % Input:
    % inputVec - Vettore di input contenente i valori 0, 1, 2
    % minLen - Lunghezza minima di numeri consecutivi che non richiede modifica
    %
    % Output:
    % output - Vettore corretto

    output = inputVec;
    n = length(inputVec);
    idx = 1;

    while idx <= n
        % Trova la lunghezza della serie attuale
        startIdx = idx;
        while idx <= n && inputVec(idx) == inputVec(startIdx)
            idx = idx + 1;
        end
        len = idx - startIdx;

        % Se la serie è troppo breve e non si trova all'inizio o alla fine
        if len < minLen && startIdx > 1 && idx <= n
            % Trova il valore precedente
            prevVal = inputVec(startIdx - 1);
            
            % Decidi se estendere il valore precedente o quello successivo
            if idx <= n
                nextVal = inputVec(idx);
                % Calcola il numero di elementi successivi identici
                nextLen = 0;
                for j = idx:n
                    if inputVec(j) == nextVal
                        nextLen = nextLen + 1;
                    else
                        break;
                    end
                end

                % Scegli il valore da propagare in base alla maggior presenza
                if nextLen > len
                    output(startIdx:idx-1) = nextVal;
                else
                    output(startIdx:idx-1) = prevVal;
                end
            else
                % Propaga il valore precedente se la serie è alla fine
                output(startIdx:idx-1) = prevVal;
            end
        end
    end
end

% function output = correggiVettore(inputVec, minLen)
%     % Input:
%     % inputVec - Vettore di input contenente i valori 0, 1, 2
%     % minLen - Lunghezza minima di numeri consecutivi che non richiede modifica
%     %
%     % Output:
%     % output - Vettore corretto
% 
%     % Inizializzazione del vettore di output e variabili
%     output = inputVec;
%     n = length(inputVec);
%     currentVal = inputVec(1);  % Inizializza al primo elemento
%     startIdx = 1;  % Indice di inizio di una serie
% 
%     for i = 2:n
%         if inputVec(i) ~= currentVal
%             if i - startIdx < minLen  % Se la serie è troppo breve
%                 if startIdx > 1 % Evita cambi all'inizio del vettore
%                     if (i + minLen - 1 <= n) % Evita cambi alla fine del vettore
%                         % Analizza il valore seguente alla serie breve
%                         nextVal = inputVec(i);
%                         countNext = 1;
%                         for j = i+1:n
%                             if inputVec(j) == nextVal
%                                 countNext = countNext + 1;
%                             else
%                                 break;
%                             end
%                         end
%                         % Sostituisce con il valore più frequente vicino alla serie breve
%                         if countNext >= i - startIdx
%                             output(startIdx:i-1) = nextVal;
%                         else
%                             output(startIdx:i-1) = inputVec(startIdx-1);
%                         end
%                     end
%                 end
%             end
%             currentVal = inputVec(i);
%             startIdx = i;
%         end
%     end
% 
%     % Ultima serie
%     if n + 1 - startIdx < minLen && startIdx > 1
%         output(startIdx:end) = inputVec(startIdx-1);
%     end
% end





% function processed_output = postProcess(input_vector, length_threshold, value)
% 
%     processed_output = input_vector;
%     n = length(processed_output);
% 
%     % Indice che scorre il vettore completo
%     i=1;    
% 
%     while i<=n
%         % Controlla se il valore corrente è il valore da trovare
%         if vettore(i) == valore_da_trovare
%             % Inizia a contare quanti volte si ripete il valore
%             inizio_serie = i;
%             while i <= n && vettore(i) == valore_da_trovare
%                 i = i + 1;
%             end
%             fine_serie = i - 1;
% 
%             % Calcola la lunghezza della serie trovata
%             lunghezza_serie = fine_serie - inizio_serie + 1;
% 
%             % Se la lunghezza della serie è minore della soglia, azzera quella serie
%             if lunghezza_serie < soglia
%                 processed_output(inizio_serie:fine_serie) = 0;
%             end
%         else
%             i = i + 1;  % Incrementa l'indice se il valore corrente non è quello da trovare
%         end
%     end
% 
% end


