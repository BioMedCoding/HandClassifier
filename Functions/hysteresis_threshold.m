function activation = hysteresis_threshold(envelope, th_L, th_H)
    
    % Initialisation of activation vector
    activation = zeros(1, length(envelope));
    % Initialisation of previous_state variable
    previous_state = 0;
    % Comparison between the envelope and the 2 thresholds
    for i = 1:length(envelope)
        % if the envelope is under the lower threshold we put both
        % activation vector and previous_state variable equal to 0
        if envelope(i) < th_L
            activation(i) = 0;
            previous_state = 0;  
        % if the envelope is upper the higher threshold we put both
        % activation vector and previous_state variable equal to 1
        elseif envelope(i) > th_H
            activation(i) = 1;
            previous_state = 1;  
        % if the envelope is between the 2 thresholds we put the
        % activation vector equal to the previous_state variable
        else
            activation(i) = previous_state;
        end
    end
    
end