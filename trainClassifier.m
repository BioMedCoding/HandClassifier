function [trained_models, best_model_index,metrics] = trainClassifier(classifier,trainParameters,sig_train,label_train,sig_val,label_val,generalParameters,filterParameters)

switch classifier
    case 'SVM'
        fprintf('\nStarted SVM training \n')
        tic;

        % Initialize metrics structure array
        metrics(trainParameters.trainingRepetitions) = struct('CM', [], 'accuracy', [], 'precision', [], 'specificity', [], 'sensibility', [], 'f1Score', [], 'mean_sensibility', [], 'mean_specificity', []);

        % Create a list of subfolders in the save path
        folderContents = dir(trainParameters.savePath);
        isSubdir = [folderContents.isdir];
        subfolders = folderContents(isSubdir);

        % Filter only subfolders named with a number
        subfolderNames = {subfolders.name};
        numericNames = cellfun(@(x) str2double(x), subfolderNames);
        numericNames(isnan(numericNames)) = [];

        % Find the highest present number
        if isempty(numericNames)
            newFolderNumber = 1; % If no subfolder, start from one
        else
            newFolderNumber = max(numericNames) + 1; % Otherwise, increment the highest present number
        end

        % Create a new folder with progressiv name
        newExperimentFolder = fullfile(trainParameters.savePath, num2str(newFolderNumber));
        mkdir(newExperimentFolder);

        for repetition=1:trainParameters.trainingRepetitions
            fprintf('\n     Starting %d repetition\n',repetition)
            if trainParameters.hypertuning  % Use parameters hypertuning 
                % Select GPU or CPU
                if trainParameters.useGPU
                    % Convert into GPU array
                    gpu_sig_train = gpuArray(sig_train);
                    gpu_label_train = gpuArray(label_train);
                    svm_model = fitcecoc(gpu_sig_train,gpu_label_train, 'Learners', t_hyper, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opts_hyp); 
                    svm_model = gather(svm_model);
                else
                    svm_model = fitcecoc(sig_train,label_train, 'Learners', t_hyper, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opts_hyp);
                end
            else    % Addestramento singolo
                if trainParameters.useGPU   % Use a specific training 
                    % Convert into GPU array
                    gpu_sig_train = gpuArray(sig_train);
                    gpu_label_train = gpuArray(label_train); 
                    svm_model = fitcecoc(gpu_sig_train,gpu_label_train, 'Learners', t_single, 'Coding', coding_single); 
                    svm_model = gather(svm_model);
                else
                    svm_model = fitcecoc(sig_train,label_train, 'Learners', trainParameters.t_single, 'Coding', trainParameters.coding_single);
                end
            end
            
            svm_models{repetition} = svm_model;

            prediction_svm_val = predict(svm_models{repetition}, sig_val);
            metrics_single = evaluateClassifier(label_val, prediction_svm_val, trainParameters.showCM, trainParameters.showText, classifier, 'Validation');

            % Store metrics in the structure
            %metrics{repetition} = metrics_single;
            % Store metrics in the structure array
            metrics(repetition).CM = metrics_single.CM;
            metrics(repetition).accuracy = metrics_single.accuracy;
            metrics(repetition).precision = metrics_single.precision;
            metrics(repetition).specificity = metrics_single.specificity;
            metrics(repetition).sensibility = metrics_single.sensibility;
            metrics(repetition).f1Score = metrics_single.f1Score;
            metrics(repetition).mean_sensibility = metrics_single.mean_sensibility;
            metrics(repetition).mean_specificity = metrics_single.mean_specificity;

            if trainParameters.saveAllModel
                % Save every model with its name
                save(fullfile(newExperimentFolder, sprintf('svm_model_%d.mat', repetition)), 'svm_model');
            end  

            fprintf('\n         Finished %d repetition\n',repetition)
        end

        elapsed_time = toc;
        fprintf('Finished SVM training. Elapsed time: %.2f seconds\n', elapsed_time);

         % Find the best model based on mean_sensibility + mean_specificity
        mean_sensibilities = [metrics.mean_sensibility];
        mean_specificities = [metrics.mean_specificity];
        [~, best_model_index] = max(mean_sensibilities + mean_specificities);


       if trainParameters.saveAllModel
            % Rename the best model trained
            movefile(fullfile(newExperimentFolder, sprintf('svm_model_%d.mat', best_model_index(1))), fullfile(newExperimentFolder, 'best_svm_model.mat'));
        else
            % Save just the best model trainedd
            best_svm_model = svm_models{best_model_index}; % Assign to a temporary variable
            save(fullfile(newExperimentFolder, 'best_svm_model.mat'), 'best_svm_model');
       end

       % Save metrics, training, general and filter parameters structure to a file
       save(fullfile(newExperimentFolder, 'metrics.mat'), 'metrics');
       save(fullfile(newExperimentFolder, 'trainParameters.mat'), 'trainParameters');
       save(fullfile(newExperimentFolder, 'generalParameters.mat'), 'generalParameters');
       save(fullfile(newExperimentFolder, 'filterParameters.mat'), 'filterParameters');
       
       trained_models = svm_models;

    otherwise
        fprintf('\nClassificatore non supportato')

end


end