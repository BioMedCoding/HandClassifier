function metrics = evaluateClassifier(label_val, prediction, show_cm, show_text, classes, methodName, nameSet)
    
    CM = confusionmat(label_val, prediction);

    if show_cm
        figure();
        confusionchart(CM, classes);
        title(['Confusion Matrix - ', methodName, ' - ', nameSet]);
    end

    % Calculate per class metrics
    classNumber = size(CM, 1);
    precision = zeros(classNumber, 1);
    sensibility = zeros(classNumber, 1);
    f1Score = zeros(classNumber, 1);
    specificity = zeros(classNumber, 1);

    for i = 1:classNumber
        precision(i) = CM(i, i) / sum(CM(:, i));
        sensibility(i) = CM(i, i) / sum(CM(i, :));

        if (precision(i) + sensibility(i)) == 0
            f1Score(i) = 0;
        else
            f1Score(i) = 2 * (precision(i) * sensibility(i)) / (precision(i) + sensibility(i));
        end

        TN = sum(CM(:)) - sum(CM(i, :)) - sum(CM(:, i)) + CM(i, i);
        FP = sum(CM(:, i)) - CM(i, i);
        specificity(i) = TN / (TN + FP);
    end

    % Calculate complessive accuracy
    total = sum(CM(:));
    accuracy = sum(diag(CM)) / total;

    mean_sensibility = mean(sensibility);
    mean_specificity = mean(specificity);

    % Print per class metrics
    if show_text
        fprintf('\n---------------------------\n');
        fprintf('Complessive accuracy %s - %s: %.2f%%\n', methodName, nameSet, accuracy * 100);
        fprintf('Complessive sensibility %s - %s: %.2f%%\n', methodName, nameSet, mean_sensibility * 100);
        fprintf('Complessive specificity %s - %s: %.2f%%\n\n', methodName, nameSet, mean_specificity * 100);
        fprintf('%-10s %-12s %-12s %-12s %-12s\n', 'Classe', 'Precisione', 'Specificità', 'Sensibilità', 'F1 Score');
    
        for i = 1:classNumber
            fprintf('%-10d %-12.2f %-12.2f %-12.2f %-12.2f\n', i, precision(i)*100, specificity(i)*100, sensibility(i)*100, f1Score(i)*100);
        end
    
        fprintf('\n---------------------------\n');
    end
    metrics.CM = CM;
    metrics.accuracy = accuracy;
    metrics.precision = precision;
    metrics.specificity = specificity;
    metrics.sensibility = sensibility;
    metrics.f1Score = f1Score;
    metrics.mean_sensibility = mean_sensibility;
    metrics.mean_specificity = mean_specificity;

end