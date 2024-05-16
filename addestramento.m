% Inizializzazione
clear 
close all
clc


%% ======================== Parametri generali script ======================
mostra_grafici_segnali = false;                      % Mostra grafici relativi ai segnali pre-classificazione
mostra_segnale_per_canale = false;

percorso_dati_aperture = "Original_data/aperture.txt";
percorso_dati_chiusure = "Original_data/chiusure.txt";
percorso_label_training = "Prepared_data/label_dataset_completo";

valore_apertura = 1;                                % Valore label apertura
valore_chiusura = 2;                                % Valore label chiusura
classi = {'Rilassata', 'Apertura','Chiusura'};      % Nomi assegnati alle classi

allena_svm = false;                                  % Esegui la sezione di addestramento e testing SVM
allena_lda = true;                                  % Esegui la sezione di addestramento e testing LDA
allena_rete_neurale = true; 

rapporto_training_validation = 0.7;
salvataggio_train_val = true;
numero_worker = 14; 

salva_modelli = true;                               % Salva i modelli allenati                           
percorso_salvataggio_modelli = "C:\Users\matte\Documents\GitHub\HandClassifier\Modelli_allenati_addestramento"; % Percorso dove salvare i modelli
percorso_salvataggio_train_val = "C:\Users\matte\Documents\GitHub\HandClassifier\Prepared_data";

warning('off', 'MATLAB:table:ModifiedAndSavedVarnames'); % Disabilita il warning relativo agli header

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




%%  ========================Parametri addestramento SVM ====================
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
coding_single = 'onevsall'; % 'onevsone', 'onevsall'
%% =========================================================================




%%  ========================Parametri addestramento NN =====================

rete_custom = false;                % Abilita l'utilzzo della rete neurale custom made

max_epoche = 500;                   % Numero massimo di epoche di allenamento della rete neurale
val_metrica_obiettivo = 0.00005;    % Metrica della rete di allenamento considerata accettabile per interrompere l'allenamento
      
lr = 0.05;                          % Learning rate
momentum = 0.9;                     % Momento durante l'allenamento

train_function = 'trainscg';        % Funzione di training della rete neurale
        % Le funzioni di training possibili sono 
        % 'trainlm': Rapida discesa, risultati finali simili ma nel test perde complatamente l'aperatura
        % 'traingd': Discesa del gradiente estremamente lenta, prestazioni scadenti dopo 500 epoche
        % 'traingdm': Discesa del gradiente più rapida, ma comunque lenta, prestazioni leggermente migliori ma comunque basse
        % 'traingdx': Discesa del gradiente rapida, da lavorarci meglio
        % 'trainrp': Discesa del gradiente molto rapida e ottima velocità, da lavorarci meglio
        % 'trainscg': Criterio benchmark, da seguire come riferimento per il momento

neuron_function = 'logsig'; % 'tansig', 'logsig', 'purelin', 'poslin', 'softmax'
        % 'tansig': 85.09% accuretezza processata, 59.43 peggiore (sensibilità 3)
        % 'logsig': 86.7% accuratezza processata, 62.77 peggiore (sensibilità 3)
        % 'purelin': 82.76% accuratezza processata, 55 peggiore (sensibilità 3)
        % 'poslin': 75.12% accuretezza processata, 45.7 peggiore (sensibilità 3)
        % 'softmax': no, peggiore

num_layer = 2;
        % 1  layer da 5: 86.33% accuretazze complessiva (postProcess), 57.54 peggiore (sensibilità 3) 500 epoche
        % 3 layer (10, 5, 3): 87.8% accuretezza complessiva (postProcess), 64.01 peggiore (sensibilità 3) 500 epoche
        % 3 layer (10, 5, 3): 86.78% accuretezza complessiva (postProcess), 59.83 peggiore (sensibilità 3) 1000 epoche arrestato prima 

% Definizione del numero di neuroni per layer in base alla struttura
% selezionata della rete
if num_layer == 1
    layer1 = 5;
end

if num_layer == 2
    layer1 = 5;
    layer2 = 3;
end

if num_layer == 3
    layer1 = 10;                     
    layer2 = 5;                        
    layer3 = 3;
end
%% =========================================================================




%% ======================== Parametri addestramento LDA ====================
discrimType = 'quadratic'; % 'linear', 'quadratic', 'diaglinear', 'diagquadratic', 'pseudolinear','pseudoquadratic'
        % Le metriche qui sotto sono riferite al test set, senza postprocess. Accuratezza complessiva e poi metrica peggiore
        % 'linear': 75.22%, 48.64 sensibilità 3
        % 'quadratic': 87.81%, 72.65% sensibilità 2
        % 'diaglinear': 70.94%, 36.31 sensibilità 2
        % 'diagquadratic': 75.89%, 47.18 sensibilità 3
        % 'pseudolinear': 75.25%, 48.66 sensibilità 3
        % 'pseudoquadratic': 75.25%, 48.62 sensibilità 3

%% =========================================================================




%% ================= Avvio pool se necessario e non attivo =================
if svm_parameter_hypertuning
    if isempty(gcp('nocreate'))
            parpool('local', numero_worker); % Avvia in modalità processes
            %parpool('Threads')              % Avvia in modalità thread
    end
end
%% =========================================================================




%% Import segnali

% Import segnali aperture
sig = readtable(percorso_dati_aperture,"Delimiter", '\t');
%t_hyp = sig(:,1); % salva colonna dei tempi prima di cancellarla
sig(:,1) = [];
sig_aperture = table2array(sig);


% Import segnali chiusure
sig = readtable(percorso_dati_chiusure,"Delimiter", '\t');
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
    sig_filt(:,i) = filter_general(sig(:,i),tipo_filtro,f_sample,"fL",f_taglio_basso,"fH",f_taglio_alta,"fN",f_notch,"visualisation",visualisation);
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

% Riga per usare il segnale senza inviluppo
%envelope_std = (sig_filt-mean(sig_filt))./std(sig_filt);

if mostra_grafici_segnali
    figure
    plot(envelope_std)
    hold on
end

%% Caricamento label segnale di training

data = load(percorso_label_training);
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

if mostra_grafici_segnali
    figure
    plot(label_dataset_completo)
    hold on
    plot(envelope_std)
    title('Segnale e label finali, post elaborazione');
    xlabel('Campioni');
    ylabel('[a.u.]');
end




%% Divisione training_validation set

% Creazione indici per training e test
num_samp = length(envelope_std(:,1));
index_random = randperm(num_samp);
training_idx = index_random(1:round(rapporto_training_validation*num_samp));
validation_idx = index_random(round(rapporto_training_validation*num_samp):end);

sig_train = envelope_std(training_idx,:);
sig_val = envelope_std(validation_idx,:);
label_train = label_dataset_completo(training_idx,:);
label_val = label_dataset_completo(validation_idx,:);

% Salvataggio training e validation set
if salvataggio_train_val
    save(strcat(percorso_salvataggio_train_val,"/training_set"), "sig_train")
    save(strcat(percorso_salvataggio_train_val,"/validation_set"), "sig_val")
    save(strcat(percorso_salvataggio_train_val,"/label_train"), "label_train")
    save(strcat(percorso_salvataggio_train_val,"/label_val"), "label_val")
end
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
            svm_model = fitcecoc(gpu_sig_train,gpu_label_train, 'Learners', t_single, 'Coding', coding_single); 
            svm_model = gather(svm_model);
        else
            svm_model = fitcecoc(sig_train,label_train, 'Learners', t_single, 'Coding', coding_single);
        end
    end

    if salva_modelli
        % Salva il modello allenato in un file .mat
        save(fullfile(percorso_salvataggio_modelli, 'svm_model.mat'), 'svm_model');
    end  
end  

%% LDA - addestramento

if allena_lda
    lda_model = fitcdiscr(sig_train,label_train,  'DiscrimType', discrimType);
    
    % Visualizza i coefficienti del modello
    %Mdl.Coeffs(1,2).Const    % Costante del decision boundary
    %Mdl.Coeffs(1,2).Linear   % Coefficienti lineari per il decision boundary

    if salva_modelli
        % Salva il modello allenato in un file .mat
        save(fullfile(percorso_salvataggio_modelli, 'lda_model.mat'), 'lda_model');
    end
end  

%% Rete neurale - addestramento

if allena_rete_neurale
    % Definizione architettura - sistema completo ma non testato
    % Definizione dei layer della rete neurale
    if rete_custom

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

    else

        % Versione 1 layer
        if num_layer == 1
            net = patternnet(layer1, train_function, 'crossentropy');  % Rete con un solo hidden layer
            net.layers{1}.transferFcn = neuron_function;   % Funzione di trasferimento del layer nascosto
            net.layers{2}.transferFcn = 'softmax';  % Funzione di trasferimento del layer di output
            
            net.trainFcn = train_function;  
            net.trainParam.epochs = max_epoche;
            net.trainParam.goal = val_metrica_obiettivo;
            if strcmp(train_function, 'traingdx')  
                net.trainParam.lr = lr;
                net.trainParam.mc = momentum;
            end
        end
    
        % Versione 2 layer
        if num_layer == 2
            net = patternnet([layer1 layer2], train_function, 'crossentropy');  
            net.layers{1}.transferFcn = neuron_function;
            net.layers{2}.transferFcn = neuron_function;
            net.layers{3}.transferFcn = 'softmax';
        
            net.trainFcn = train_function;  
            net.trainParam.epochs = max_epoche;
            net.trainParam.goal = val_metrica_obiettivo;
            if strcmp(train_function, 'traingdx')
                net.trainParam.lr = lr;
                net.trainParam.mc = momentum;
            end
        end
    
        % Versione 3 layer
        if num_layer == 3
            net = patternnet([layer1 layer2 layer3], train_function, 'crossentropy');  
            net.layers{1}.transferFcn = neuron_function;
            net.layers{2}.transferFcn = neuron_function;
            net.layers{3}.transferFcn = neuron_function;   
            net.layers{4}.transferFcn = 'softmax';  
            
            net.trainFcn = train_function;  
            net.trainParam.epochs = max_epoche;
            net.trainParam.goal = val_metrica_obiettivo;
            if strcmp(train_function, 'traingdx') 
                net.trainParam.lr = lr;
                net.trainParam.mc = momentum;
            end
        end
    
        % Comandi per gestire automaticamente la gestione dell'intero dataset
        % net.divideParam.trainRatio = 70/100;
        % net.divideParam.valRatio = 15/100;
        % net.divideParam.testRatio = 15/100;
        
        % Adattamento segnali e label a formato rete
        sig_train = sig_train'; % Trasposizione per adattare a necessità rete
        label_train = label_train+1;
        label_train = full(ind2vec(label_train'));  % Converti in formato one-hot e trasponi
    
        % Allena
        [net, tr] = train(net, sig_train, label_train);
    
        end

    if salva_modelli
            % Salva il modello allenato in un file .mat
            save(fullfile(percorso_salvataggio_modelli, 'nn_model.mat'), 'net');
    end
end