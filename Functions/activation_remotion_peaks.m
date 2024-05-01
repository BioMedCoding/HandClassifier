function clean_activation = activation_remotion_peaks(activation,peak_width,rm_value)
    
    % Initialisation of counter
    counter = 0;
    % Initialisation of clean_activation vector equal to activation vector
    clean_activation = activation;
    % Analysis of activation vector sample by sample
    for i = 1:length(activation)
        % if the activation vector is equal to the rm_value we start to
        % counter how many samples of this value are present
        if activation(i) == rm_value
            counter = counter + 1;
        else
            % if the activation vector is not more equal to the rm_value
            % and the counter is lower than peak_width variable it means
            % that it is a peak, so we remove it
            if counter < round(peak_width)
                clean_activation(i-counter:i-1) = abs(rm_value-1);
            end
            % Reset the counter
            counter = 0;
        end
    end
    % check for the last samples
    if counter < round(peak_width)
        clean_activation(end-counter+1:end) = abs(rm_value-1);
    end
end
