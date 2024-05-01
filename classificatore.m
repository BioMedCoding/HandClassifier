% Inizializzazione
clear all
close all
clc

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
esegui_svm = false;                                     % Esegui la sezione di addestramento e testing SVM
esegui_lda = false;                                     % Esegui la sezione di addestramento e testing LDA
numero_worker = 10;                                 % Numero di worker da usare per il parallel pool
% =========================================================================


%  ========================Parametri addestramento SVM ====================
svm_parameter_hypertuning = false;                  % Abilita hypertuning automatico dei parametri, sfruttando anche parallel pool
svm_calcolo_GPU = false;                            % Abilita l'addestramento dell'SVM tramite l'uso della GPU
ore_esecuzione_massime = 10;                        % Numero massimo di ora per cui continuare l'hypertuning automatico dei parametri

t_hyp = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto');
opts_hyp = struct('AcquisitionFunctionName', 'expected-improvement-plus', ...
    'UseParallel', true, ...
    'MaxObjectiveEvaluations', 30, ... 
    'Verbose', 2, ...                                              
    'ShowPlots', true, ...                                         
    'SaveIntermediateResults', true, ...                           
    'MaxTime', ore_esecuzione_massime*3600);                       

%t_single = templateSVM('KernelFunction', 'polynomial', 'PolynomialOrder', 2);
t_single = templateSVM('KernelFunction', 'rbf', 'BoxConstraint', 21.344, 'KernelScale', 0.55962); % Valore migliore trovato durante hypertuning automatico, con onevsone
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
    envelope(:,i) = filter_general(abs(sig_filt(:,i)),tipo_filtro,f_sample,"fH",f_envelope,"percH",percH);   % CONTROLLA QUI LA CORRETTEZZA, ma penso sia ok
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

pause
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
            svm_model = fitcecoc(gpu_sig_train,gpu_label_train, 'Learners', t_hyp, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opts_hyp, 'ClassNames', classi); 
            svm_model = gather(svm_model);
        else
            svm_model = fitcecoc(sig_train,label_train, 'Learners', t_hyp, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opts_hyp, 'ClassNames', classi);
        end
    else    % Addestramento singolo
        if svm_calcolo_GPU
            % Trasferimento dei dati sulla GPU
            gpu_sig_train = gpuArray(sig_train);
            gpu_label_train = gpuArray(label_train);
            svm_model = fitcecoc(gpu_sig_train,gpu_label_train, 'Learners', t_single, 'Coding', coding_single, 'ClassNames', classi); 
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
    fprintf('Accuratezza complessiva: %.2f\n', accuratezza_svm*100);
    
    
    % Stampa dell'intestazione delle metriche
    fprintf('%-10s %-12s %-12s %-12s %-12s\n', 'Classe', 'Precisione', 'Specificità', 'Sensibilità', 'F1 Score');
    
    % Stampa delle metriche per ogni classe
    for i = 1:numClassi
        fprintf('%-10s %-12.2f %-12.2f %-12.2f %-12.2f\n', char(classi(i)), precision_svm(i)*100, specificita_svm(i)*100, sensibilita_svm(i)*100, f1Score_svm(i)*100);
    end
    
    % Stampa delle metriche per ogni classe 
    %fprintf('Classe\tPrecisione\tRichiamo\tF1 Score\n'); 
    %for i = 1:numClassi 
    %    fprintf('%d\t\t%.2f\t\t\t%.2f\t\t\t%.2f\n', i, precision(i), sensibilita(i), f1Score(i)); 
    %end
    
    % Save the trained model to a .mat file
    save('svm_model.mat', 'svm_model');
end

pause
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
    figure()
    confusionchart(CM_lda)
    
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
    fprintf('Accuratezza complessiva: %.2f\n', accuratezza_lda);
    
    
    % Stampa dell'intestazione delle metriche
    fprintf('%-10s %-12s %-12s %-12s %-12s\n', 'Classe', 'Precisione', 'Specificità', 'Sensibilità', 'F1 Score');
    
    % Stampa delle metriche per ogni classe
    for i = 1:numClassi
        fprintf('%-10d %-12.2f %-12.2f %-12.2f %-12.2f\n', i, precision_lda(i), specificita_lda(i), sensibilita_lda(i), f1Score_lda(i));
    end
    
    save('lda_model.mat', 'lda_model');
end


pause

%% Cosine similarity


%% Test su dati completi e calcolo metriche
load('svm_model.mat')

prediction_complete_svm = predict(svm_model, envelope_std);

%% Rappresentazione dati di test
%figure()
%plot(prediction_complete_svm)
%hold on
%plot(label)
%title('Confronto tra ground truth e predizioni')
%xlabel('Campioni')
%ylabel('[a.u.]')
%legend('Predizione', 'Ground truth')

%% Bimbo special
figure;
subplot(2,1,1);
plot(prediction_complete_svm);
title('Predizioni');
xlabel('Campioni');
ylabel('[a.u.]');

subplot(2,1,2);
plot(label);
title('Ground truth e predizioni');
xlabel('Campioni');
ylabel('[a.u.]');

% Collega gli assi verticali dei due subplot
linkaxes([subplot(2,1,1), subplot(2,1,2)]);

%% Funzioni usate
function vec = creaEtichetta(n, start_indices, end_indices, value)
    % Inizializza il vettore di zeri
    vec = zeros(1, n);
    
    % Imposta il valore numerico agli elementi tra gli indici di inizio e fine
    for i = 1:length(start_indices)
        vec(start_indices(i):end_indices(i)) = value;
    end
end



