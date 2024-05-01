function [sig_filt] = filter_general(x,method,fs,f,parameter)
    arguments
        x
        method char {mustBeMember(method,{'butter','cheby1','cheby2','fir'})}
        fs (1,1) {mustBeNumeric}
        f.fL (1,1) {mustBeNumeric} = NaN
        f.percL (1,1) {mustBeNumeric,mustBeLessThan(f.percL,1)} = 0.8
        f.RpL (1,1) {mustBeNumeric} = 1
        f.RsL (1,1) {mustBeNumeric} = 20
        f.fH (1,1) {mustBeNumeric} = NaN
        f.percH (1,1) {mustBeNumeric,mustBeGreaterThan(f.percH,1)} = 1.1
        f.RpH (1,1) {mustBeNumeric} = 1
        f.RsH (1,1) {mustBeNumeric} = 20
        f.fN {mustBeNumeric} = NaN
        f.band_width (1,1) {mustBeNumeric} = 2
        f.RsN (1,1) {mustBeNumeric} = 20
        parameter.visualisation char {mustBeMember(parameter.visualisation,{'yes','no'})} = 'no' 
    end
    
    fNy = fs/2;
    sig = x;
    
    switch method
        case 'butter'
            % High Pass Filter
            if ~isnan(f.fL)
                N = buttord(f.fL/fNy,(f.percL*f.fL)/fNy , f.RpL, f.RsL);
                [bH,aH] = butter(N,f.fL/fNy,"high");
                if strcmp(parameter.visualisation,'yes')
                    figure()
                    freqz(bH,aH,1e5,fs);
                end
                if isstable(bH,aH)
                    x = filtfilt(bH,aH,x);
                else
                    error('HPF not stable, try to relax the specifications')
                end
            end
            % Low Pass Filter
            if ~isnan(f.fH)
                N = buttord(f.fH/fNy,(f.percH*f.fH)/fNy , f.RpH, f.RsH);
                [bL,aL] = butter(N,f.fH/fNy,"low");
                 if strcmp(parameter.visualisation,'yes')
                    figure()
                    freqz(bL,aL,1e5,fs);
                end
                if isstable(bL,aL)
                    x = filtfilt(bL,aL,x);
                else
                    error('LPF not stable, try to relax the specifications')
                end
            end

        case 'cheby1'
            % High Pass Filter
            if ~isnan(f.fL)
                N = cheb1ord(f.fL/fNy,(f.percL*f.fL)/fNy , f.RpL, f.RsL);
                [bH,aH] = cheby1(N,f.RpL,f.fL/fNy,"high");
                if strcmp(parameter.visualisation,'yes')
                    figure()
                    freqz(bH,aH,1e5,fs);
                end
                if isstable(bH,aH)
                    x = filtfilt(bH,aH,x);
                else
                    error('HPF not stable, try to relax the specifications')
                end
            end
            % Low Pass Filter
            if ~isnan(f.fH)
                N = cheb1ord(f.fH/fNy,(f.percH*f.fH)/fNy , f.RpH, f.RsH);
                [bL,aL] = cheby1(N,f.RpH,f.fH/fNy,"low");
                if strcmp(parameter.visualisation,'yes')
                    figure()
                    freqz(bL,aL,1e5,fs);
                end
                if isstable(bL,aL)
                    x = filtfilt(bL,aL,x);
                else
                    error('LPF not stable, try to relax the specifications')
                end
            end

        case 'cheby2'
            % High Pass Filter
            if ~isnan(f.fL)
                N = cheb2ord(f.fL/fNy,(f.percL*f.fL)/fNy , f.RpL, f.RsL);
                [bH,aH] = cheby2(N,f.RsL,(f.percL*f.fL)/fNy,"high");
                if strcmp(parameter.visualisation,'yes')
                    figure()
                    freqz(bH,aH,1e5,fs);
                end
                if isstable(bH,aH)
                    x = filtfilt(bH,aH,x);
                else
                    error('HPF not stable, try to relax the specifications')
                end
            end
            % Low Pass Filter
            if ~isnan(f.fH)
                N = cheb2ord(f.fH/fNy,(f.percH*f.fH)/fNy , f.RpH, f.RsH);
                [bL,aL] = cheby2(N,f.RsH,(f.percH*f.fH)/fNy,"low");
                if strcmp(parameter.visualisation,'yes')
                    figure()
                    freqz(bL,aL,1e5,fs);
                end
                if isstable(bL,aL)
                    x = filtfilt(bL,aL,x);
                else
                    error('LPF not stable, try to relax the specifications')
                end
            end
            
        case 'fir'
            % High Pass Filter
            if ~isnan(f.fL)
                N = 50;
                aH = 1;
                bH = fir1(N,f.fL/fNy,"high");
                if strcmp(parameter.visualisation,'yes')
                    figure()
                    freqz(bH,aH,1e5,fs);
                end
                x = filter(bH,aH,x);
            end
            % Low Pass Filter
            if ~isnan(f.fH)
                N = 50;
                aL = 1;
                bL = fir1(N,f.fH/fNy,"low");
                if strcmp(parameter.visualisation,'yes')
                    figure()
                    freqz(bL,aL,1e5,fs);
                end
                x = filter(bL,aL,x);
            end

        otherwise
            fprintf('metodo non trovato')
    end

    if ~isnan(f.fN)
        for fn_num = 1 : length(f.fN) 
            [bN,aN] = rico(f.RsN,f.band_width,f.fN(fn_num),fs);
            if strcmp(parameter.visualisation,'yes')
                    figure()
                    freqz(bN,aN,1e5,fs);
            end
            if isstable(bN,aN)
                x = filtfilt(bN,aN,x);
            else
                error('Notch filter not stable at frequency: %d \n try to relax the specifications',f.fN(fn_num));
            end
        end
    end

    sig_filt = x;
end

