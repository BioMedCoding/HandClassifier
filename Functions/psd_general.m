% FUNZIONE per il calcolo della Power Spettral Density, creata per aiutare
% a scegliere agevolmente il metodo più opportuno
%
% PSD = psd_general(x,method,fs);
%
% INPUT - x ->  segnale sottoforma di vettore numerico di cui si vuole
%               stimare ls PSD
%
%       - method -> metodo che si desidera utilizzare per la stima della
%                   PSD, i possibili metodi sono :
%                   1) METODO DEL PERIODOGRAMMA SEMPLICE, prevede una
%                   finestra rettangolare della dimensione di tutto il
%                   segnale, ovvero con la massimo risoluzione teorica
%                   possibile.
%                   2) METODO DI BARTLETT, prevede una finestra a scelta
%                   ( parameter.window per l'elenco delle possbili
%                   finestre ), di una dimensione variabile in base alla
%                   risoluzione teorica desiderta ( parameter.ris_teorica
%                   per approfondimento ), overlap delle finestre pari a 0%
%                   3) METODO DI WELCH, prevede una finestra a scelta (
%                   parameter.window per l'elenco delle possibili finestre)
%                   di dimensione variabile in base alla risoluzione 
%                   teorica desiderta ( parameter.ris_teorica per 
%                   approfondimento ), overlap della finestra variabile (
%                   parameter.perc_overlap per approfondimento)
%                   4) METODO DI BURG PARAMETRICO, è un metodo parametrico
%                   che prende in input un ordine massimo da valutare (
%                   parameter.order per approfondimento), non necessita
%                   di overlap o di finestre ma si può cambiare la
%                   risoluzione apparente
%                   5) METODO DI COVARIANZA MODIFICATA PARAMETRICA, è un
%                   metodo parametrico che prende in input un ordine
%                   massimo da valutare ( parameter.order per 
%                   approfondimento), non necessita di overlap o di
%                   finestre ma si può cambiare la risoluzione apparente
%                   6) METODO DI YULER-WALKER PARAMETRICO, è un metodo
%                   parametrico che prende in input un ordine massimo da
%                   valutare ( parameter.order per approfondimento), non
%                   necessita di overlap o di finestre ma si può cambiare
%                   la risoluzione apparente
%                   7) CORRELOGRAMMA, metodo non parametrico che sfruta
%                   l'autocorrelazione sia polarizzata che non (
%                   parameter.biased per approfondimento)
%
%       - fs -> frequenza di campionamento del segnale
%
%       - parameter ->  struttura contenente vari parametri utili per 
%                       impostare al meglio la stima della PSD al cui
%                       interno troviamo vari input possibili che seguono
%       - parameter.ris_teorica ->  valore numerico che definisce la 
%                                   risoluzione teorica responsabile, se il
%                                   metodo lo permette, della dimensione 
%                                   della finestra: 
%                                   *dim_finestra = fs/ris_teorica*
%                                   DEFAULT per garantire la massima
%                                   risoluzione spettrale verrà impostata
%                                   pari a fs/lunghezza_segnale
%
%       PSD = psd_general(x,method,fs,'ris_teorica',risoluzione_teorica_VAL);
%
%       - parameter.ris_apparente ->    valore numerico che definisce la 
%                                       risoluzione apparente, responsabile 
%                                       del numero di punti su cui si vuole 
%                                       valutare la PSD:
%                                       *NFFT = fs/ris_apparente*
%                                       DEFAULT pari al 10% della
%                                       risoluzione teorica per i metodi
%                                       NON parametrici
%                                       DEFAULT pari a 1 per i metodi
%                                       parametrici
%
%       PSD = psd_general(x,method,fs,'ris_apparente',risoluzione_apparente_VAL);
%
%       - parameter.window ->   stringa con il nome della tipologia di
%                               finestra che si intende si intende
%                               utilizzare, se il metodo lo permette, tra
%                               le finestre possibili ci sono:
%                               1) 'rectwin' : finestra rettangolare
%                               2) 'triang' : finestra triangolare
%                               3) 'hann' : finestra di Hann (coseno)
%                               4) 'hamming' : finestra di Hamming ( coseno
%                               rialzato)
%                               DEFAULT verrà impostata una finestra
%                               rettangolare
%
%       PSD = psd_general(x,method,fs,'window',window_STRING);
%
%       - paramter.perc_overlap ->  valore numerico per la percentuale di
%                                   overlap che ci deve essere tra le
%                                   finestre
%                                   DEFAULT viene impostata pari allo 0% in
%                                   linea con il default della risoluzione
%                                   teorica
%
%       PSD = psd_general(x,method,fs,'perc_overlap',perc_overlap_VAL);
%
%       - parameter.order ->    valore numerico contenente il massimo
%                               ordine valutato per il calcolo della
%                               varianza asintotica mediante una funzione
%                               che dipenderà dal metodo parametrico
%                               scelto:
%                               'arburg' : metodo di Burg
%                               'armcov' : metodo della Covarianza
%                               Modificata
%                               'aryule' :metodo di Yule-Walker
%                               DEFAULT viene impostato pari al 10% della
%                               lunghezza del segnale arrotondato
%                               all'intero più vicino
%
%       PSD = psd_general(x,method,fs,'order',max_order_VAL);
%
%       - parameter.biased ->   valore utilizzato per determinare se
%                               utilizzare il metodo polarizzato o non per
%                               il correlogramma: 1-> polarizzato
%                                                 0-> non polarizzato
%                               DEFAULT impostato su 1, polarizzato
%
%       - parameter.normalization ->    valore per definire se la psd in 
%                                       output la si vuole normalizzata 
%                                       ('yes'), normalizzata sull'energia 
%                                       totale del segnale ('area_unit') o
%                                       non normalizzata
%                                       DEFAULT impostato su 
%                                       normalizzazione 'yes'
%
%       PSD = psd_general(x,method,fs,'normalization',normalization_VAL);
%
%       - parameter.visualisation ->    stringa per definire se si vuole
%                                       una rappresentazione grafica di
%                                       quanto sta avvenendo nella funzione
%                                       e della PSD in output con tutti i
%                                       parametri precedentemente definiti
%                                       'yes' verranno mostrati i grafici,
%                                       'no' NON verranno mostrati i
%                                       grafici
%                                       DEFAULT i grafici non verranno
%                                       rappresentati ('no)
%
%       PSD = psd_general(x,method,fs,'visualisation',visualisation_STRING);
%
%
% OUTPUT - PSD ->   vettore contenente la Power Spettral Density valutata
%                   in vari punti quanti definiti dalla variabile NFFT
%                   ( parameter.ris_apparente per approfondimenti)
%
%        PSD = psd_general(x,method,fs);
%        
%        - Freq ->  vettore delle frequenze, lungo quanto la PSD dove il
%                   valore di partenza sarà sempre 0 mentre l'ultimo sarà 
%                   pari alla frequenza di Nyquist del segnale:
%                   FNy = fs/2;
%
%        [PSD, Freq] = psd_general(x,method,fs);
%
%        - R_Teorica ->  risoluzione teorica utilizzata dal metodo NON
%                        parametrico.
%                        In caso di metodo parametrico verrà restituito
%                        l'ordine utilizzato dal metodo per la valutazione
%                        della PSD
%
%        [PSD, Freq, R_Teorica] = psd_general(x,method,fs);
%
%        - R_Apparente ->   risoluzione apparente utilizzta dal metodo
%
%        [PSD, Freq, R_Teoric, R_Apparente] = psd_general(x,method,fs);
%
%        - Perc_Overlap ->  percentuale di overlap utilizzato dal metodo
%
%        [PSD, Freq, R_Teoric, R_Apparente, Perc_Overlap] = psd_general(x,method,fs);
%
%        - window_vect ->   vettore contenente la finestra utilizzata per
%                           finestrare il segnale
%
%        [PSD, Freq, R_Teoric, R_Apparente, Perc_Overlap, window_vect] = psd_general(x,method,fs);
%
%

function [PSD, Freq, R_Teorica, R_Apparente, Perc_Overlap, window_vect] = psd_general(x,method,fs,parameter)

    arguments
        x 
        method char {mustBeMember(method,{'period','bartlett','welch','burg','covar_mod','yuler','correlogramma'})}
        fs (1,1) {mustBeNumeric}
        parameter.ris_teorica = 'none'
        parameter.ris_apparente = 'none'
        parameter.window char {mustBeMember(parameter.window,{'rectwin','triang','hann','hamming'})} = 'rectwin'
        parameter.perc_overlap {mustBeGreaterThanOrEqual(parameter.perc_overlap,0),mustBeLessThan(parameter.perc_overlap,1)} = 0;
        parameter.order = 'none'
        parameter.biased (1,1) {mustBeGreaterThanOrEqual(parameter.biased,0),mustBeInteger,mustBeLessThanOrEqual(parameter.biased,1)} = 1
        parameter.normalization char {mustBeMember(parameter.normalization,{'yes','no','area_unit'})} = 'yes'
        parameter.visualisation char {mustBeMember(parameter.visualisation,{'yes','no'})} = 'no' 
    end
    
    switch method
    % PERIODOGRAMMA SEMPLICE ##############################################
    % input possibli : - ris_apparente -> risoluzione apparente della PSD
    %                                     di default impostata pari al 10%
    %                                     della risoluzione teorica
    %                  - normalization -> 1 restituisce PSD normalizzata
    %                                     0 restituisce PSD non
    %                                     normalizzata
    %                                     di default verrà restituita la
    %                                     PSD normalizzata
    %                  - visualisation -> serve per mostrare o meno dei
    %                                     grafici con i passaggi intermedi
    %                                     o non, di default non verranno
    %                                     mostrati
    %
        case 'period'
            % nel periodogramma semplice la finestra è grande quanto tutto 
            % il segnale
            dim_finestra = length(x);
            % la risoluzione teorica dipenderà dalla dimensione del segnale
            if strcmp(parameter.ris_teorica,'none')
                parameter.ris_teorica = fs/dim_finestra;
            else
                parameter.ris_teorica = fs/dim_finestra;
                warning('Nel periodogramma semplice (%s) la risoluzione teorica sarà la massima possibile\n risoluzione_teorica = frequenza_campionamento/lunghezza_segnale\n!!!risoluzione teorica aggiornata!!!\nris_teorica = %f\n',method,parameter.ris_teorica);
            end
            if strcmp(parameter.ris_apparente,'none')
                % se non viene impostata una risoluzione apparente viene
                % considerata di default pari al 10% della risoluzione 
                % teorica
                parameter.ris_apparente = 0.1*parameter.ris_teorica;
            end
            % numero di punti su cui calcolare la PSD
            NFFT = round(fs/parameter.ris_apparente);
            % overla pari allo 0%
            if parameter.perc_overlap == 0
                overlap = 0*dim_finestra;
            else
                overlap = 0*dim_finestra;
                warning('Nel periodogramma semplice  (%s) l'' overlap tra le finestre deve essere nullo\n overlap = 0*dimensione_finestra\n!!!overlap aggiornato!!!\noverlap = %f\n',method,overlap);
            end
            % controllo sul tipo di finestra utilizzato
            if ~strcmp(parameter.window,'rectwin')
                window_vect = rectwin(dim_finestra);
                warning('Nel periodogramma semplice  (%s) la finestra utilizzata è sempre quella rettangolare\n!!!finestra aggiornata!!!',method)
            else
                window_vect = rectwin(dim_finestra);
            end
            % calcolo la PSD
            [PSD, Freq] = pwelch(x-mean(x),window_vect,overlap,NFFT,fs);
            % se normalization = yes la PSD viene restituita normalizzata
            if strcmp(parameter.normalization,'yes')
                PSD = PSD/max(PSD);
            % se normalization = 'area_unit' verrà restituita una psd con
            % area unitaria ovvero normalizzata sull'energia totale del
            % segnale
            elseif strcmp(parameter.normalization,'area_unit')
                PSD = PSD/sum(PSD);
            end
            % Definizione degli output
            % risoluzione teorica
            R_Teorica = parameter.ris_teorica;
            % risoluzione apparente
            R_Apparente = parameter.ris_apparente;
            % percentuale di overlap
            Perc_Overlap = overlap/dim_finestra;
            % mostro risultato finale se la variabile visualisation è
            % impostata su 'yes'
            if strcmp(parameter.visualisation,'yes')
                titolo = sprintf('Power Spettral Density [%s]',method);
                figure()
                plot(Freq,PSD)
                xlabel('frequency [Hz]')
                ylabel('PSD')
                title(titolo)
                grid on 
                hold off
            end
        %##################################################################

    % METODO DI BARTLETT ##################################################
    % input possibli : - ris_teorica -> risoluzione teorica della PSD di
    %                                   default impostata pari alla massima
    %                                   risoluzione teorica possibile pari
    %                                   al rapporto tra la frequenza di
    %                                   campionamento e la lunghezza del
    %                                   segnale
    %                  -ris_apparente -> risoluzione apparente della PSD
    %                                    di default impostata pari al 10%
    %                                    della risoluzione teorica
    %                  -window -> finestra utilizzata per finestrare il
    %                             segnale durante il calcolo della PSD, di
    %                             default verrà considerata una finestra
    %                             rettangolare
    %                  - normalization -> 1 restituisce PSD normalizzata
    %                                     0 restituisce PSD non
    %                                     normalizzata
    %                                     di default verrà restituita la
    %                                     PSD normalizzata
    %                  - visualisation -> serve per mostrare o meno dei
    %                                     grafici con i passaggi intermedi
    %                                     o non, di default non verranno
    %                                     mostrati
    %
        case 'bartlett'
            % se non viene impostata una risoluzione teorica viene  
            % considerata di default pari al massimo possibile generando 
            % quindi una dimensione della finestra pari alla lunghezza del 
            % segnale
            if strcmp(parameter.ris_teorica,'none')
                dim_finestra = length(x);
                parameter.ris_teorica = fs/dim_finestra;
            else
                % dimensione finestra se viene fornita una risoluzione 
                % teorica
                dim_finestra = fs/parameter.ris_teorica;
            end
            % se non viene impostata una risoluzione apparente viene
            % considerata di default pari al 10% della risoluzione teorica
            if strcmp(parameter.ris_apparente,'none')
                parameter.ris_apparente = 0.1*parameter.ris_teorica;
            end
            % numero di punti su cui calcolare la PSD
            NFFT = round(fs/parameter.ris_apparente);
            % overla pari allo 0%
            if parameter.perc_overlap == 0
                overlap = 0*dim_finestra;
            else
                overlap = 0*dim_finestra;
                warning('Nel metodo di Bartlett (%s) l'' overlap tra le finestre deve essere nullo\n overlap = 0*dimensione_finestra\n!!!overlap aggiornato!!!\noverlap = %f\n',method,overlap);
            end
            % creazione della finestra
            switch parameter.window
                case 'rectwin'
                    window_vect = rectwin(dim_finestra);
                case 'triang'
                    window_vect = triang(dim_finestra);
                case 'hann'
                    window_vect = hann(dim_finestra);
                case 'hamming'
                    window_vect = hamming(dim_finestra);
                otherwise
                    window_vect = rectwin(dim_finestra);
                    warning('!!!finestra non trovata!!!\n');
            end
            % calcolo la PSD
            [PSD, Freq] = pwelch(x-mean(x),window_vect,overlap,NFFT,fs);
            % se normalization = yes la PSD viene restituita normalizzata
            if strcmp(parameter.normalization,'yes')
                PSD = PSD/max(PSD);
            % se normalization = 'area_unit' verrà restituita una psd con
            % area unitaria ovvero normalizzata sull'energia totale del
            % segnale
            elseif strcmp(parameter.normalization,'area_unit')
                PSD = PSD/sum(PSD);
            end
            % Definizione degli output
            % risoluzione teorica
            R_Teorica = parameter.ris_teorica;
            % risoluzione apparente
            R_Apparente = parameter.ris_apparente;
            % percentuale di overlap
            Perc_Overlap = overlap/dim_finestra;
            % mostro risultato finale se la variabile visualisation è
            % impostata su 'yes'
            if strcmp(parameter.visualisation,'yes')
                titolo = sprintf('Power Spettral Density [%s]',method);
                figure()
                plot(Freq,PSD)
                xlabel('frequency [Hz]')
                ylabel('PSD')
                title(titolo)
                grid on 
                hold off
            end
        %##################################################################

    % METODO DI WELCH #####################################################
    % input possibli : - ris_teorica -> risoluzione teorica della PSD di
    %                                   default impostata pari alla massima
    %                                   risoluzione teorica possibile pari
    %                                   al rapporto tra la frequenza di
    %                                   campionamento e la lunghezza del
    %                                   segnale
    %                  -ris_apparente -> risoluzione apparente della PSD
    %                                    di default impostata pari al 10%
    %                                    della risoluzione teorica
    %                  -window -> finestra utilizzata per finestrare il
    %                             segnale durante il calcolo della PSD, di
    %                             default verrà considerata una finestra
    %                             rettangolare
    %                  - normalization -> 1 restituisce PSD normalizzata
    %                                     0 restituisce PSD non
    %                                     normalizzata
    %                                     di default verrà restituita la
    %                                     PSD normalizzata
    %                  - visualisation -> serve per mostrare o meno dei
    %                                     grafici con i passaggi intermedi
    %                                     o non, di default non verranno
    %                                     mostrati
    %
        case 'welch'
            % se non viene impostata una risoluzione teorica viene  
            % considerata di default pari al massimo possibile generando 
            % quindi una dimensione della finestra pari alla lunghezza del 
            % segnale
            if strcmp(parameter.ris_teorica,'none')
                dim_finestra = length(x);
                parameter.ris_teorica = fs/dim_finestra;
            else
                % dimensione finestra se viene fornita una risoluzione 
                % teorica
                dim_finestra = fs/parameter.ris_teorica;
            end
            % se non viene impostata una risoluzione apparente viene
            % considerata di default pari al 10% della risoluzione teorica
            if strcmp(parameter.ris_apparente,'none')
                parameter.ris_apparente = 0.1*parameter.ris_teorica;
            end
            % numero di punti su cui calcolare la PSD
            NFFT = round(fs/parameter.ris_apparente);
            % se non viene impostata una percentuale di overlap questa sarà
            % considerata di default pari a 0%
            overlap = round(parameter.perc_overlap*dim_finestra);
            % creazione della finestra
            switch parameter.window
                case 'rectwin'
                    window_vect = rectwin(dim_finestra);
                case 'triang'
                    window_vect = triang(dim_finestra);
                case 'hann'
                    window_vect = hann(dim_finestra);
                case 'hamming'
                    window_vect = hamming(dim_finestra);
                otherwise
                    window_vect = rectwin(dim_finestra);
                    warning('!!!finestra non trovata!!!\n');
            end
            % calcolo la PSD
            [PSD, Freq] = pwelch(x-mean(x),window_vect,overlap,NFFT,fs);
            % se normalization = yes la PSD viene restituita normalizzata
            if strcmp(parameter.normalization,'yes')
                PSD = PSD/max(PSD);
            % se normalization = 'area_unit' verrà restituita una psd con
            % area unitaria ovvero normalizzata sull'energia totale del
            % segnale
            elseif strcmp(parameter.normalization,'area_unit')
                PSD = PSD/sum(PSD);
            end
            % Definizione degli output
            % risoluzione teorica
            R_Teorica = parameter.ris_teorica;
            % risoluzione apparente
            R_Apparente = parameter.ris_apparente;
            % percentuale di overlap
            Perc_Overlap = overlap/dim_finestra;
            % mostro risultato finale se la variabile visualisation è
            % impostata su 'yes'
            if strcmp(parameter.visualisation,'yes')
                titolo = sprintf('Power Spettral Density [%s]',method);
                figure()
                plot(Freq,PSD)
                xlabel('frequency [Hz]')
                ylabel('PSD')
                title(titolo)
                grid on 
                hold off
            end
        %##################################################################


    % METODO DI BURG ######################################################
    % input possibli : -ris_apparente -> risoluzione apparente della PSD
    %                                    di default impostata pari a 1 Hz
    %                  -order -> ordine massimo valutato dal metodo, di
    %                            default pari al 10% della lunghezza del
    %                            segnale arrotondato all'intero più vicino
    %                  - normalization -> 1 restituisce PSD normalizzata
    %                                     0 restituisce PSD non
    %                                     normalizzata
    %                                     di default verrà restituita la
    %                                     PSD normalizzata
    %                  - visualisation -> serve per mostrare o meno dei
    %                                     grafici con i passaggi intermedi
    %                                     o non, di default non verranno
    %                                     mostrati
        case 'burg'
            % se non viene impostato un ordine massimo valutabile, verrà
            % considerato un ordine pari al 10% della lunghezza del segnale
            % arrotondato all'intero più vicino
            if strcmp(parameter.order,'none')
                parameter.order = round(0.1*length(x));
                if parameter.order == 1
                    parameter.order = 2;
                end
            end
            % inizializzazione vettore degli errori lungo pari all'ordine
            % assimo che si vuole analizzare -1
            e = zeros(1,parameter.order-1);
            % Calcolo della varianza asintotica per ordini del modello tra
            % 2 e parameter.order
            for NN = 2:parameter.order 
                [~,e(NN-1)] = arburg(x-mean(x),NN); 
            end
            % la varianza asintotica sarà il valore più piccolo ottenuto nel vettore
            % delgi errori ( l'ultimo)
            [m,I] = min(e);
            % varianza asintotica + 5% varianza asintotica
            asint = 1.05*m;
            % Trovare i valori di varianza che sono minori della varianza asintotica
            % aumentata del 5%
            ind =  find(e<asint);
            % mostro l'andamento nella valutazione dell'errore con il
            % relativo valore asintotico, il minimo e l'insieme dei valori
            % considerati per la scelta dell'ordine
            if strcmp(parameter.visualisation,'yes')
                % mostro l'andamento dell'errore al variare dell'ordine del modello
                figure()
                plot(2:ind(1)+1 ,e(1:ind(1)),'md-',ind(1)+1:ind(end)+1,e(ind),'cd-');
                hold on 
                plot(I+1,m,'ro','LineWidth',7);
                hold on
                plot(2:parameter.order,ones(1,length(e)).*asint,'b-')
                xlabel('ordine del modello')
                ylabel('errore')
                title('andamento dell errore in funzione dell ordine del modello')
                legend('errore','errore < della varianza asintotica + 5%','errore minimo','varianza asintotica + 5%')
                grid on
                hold off
            end
            % Scegliere l'ordine per il metodo parametrico 
            if length(ind)>=2
                order_param =  ind(2)+1;
            elseif length(ind)==1
                if parameter.order == 2
                    warning('ordine massimo valutato molto basso (%d)\nE'' stato valutato solo l''ordine 2\n',parameter.order)
                else
                    warning('ordine massimo valutato riduce di molto l''errore(%d)\nconsiderare  di prendere un ordine più piccolo\n',parameter.order)
                end
                order_param =  ind(1)+1;
            end
            % se non viene impostata una risoluzione apparente viene
            % considerata di default pari a 1Hz per avere un campione per
            % ogni frequenza unitaria
            if strcmp(parameter.ris_apparente,'none')
                parameter.ris_apparente = 1;
            end
            % numero di punti su cui calcolare la PSD
            NFFT = round(fs/parameter.ris_apparente);
            % calcolo la PSD
            [PSD, Freq] = pburg(x-mean(x),order_param,NFFT,fs);
            % se normalization = yes la PSD viene restituita normalizzata
            if strcmp(parameter.normalization,'yes')
                PSD = PSD/max(PSD);
            % se normalization = 'area_unit' verrà restituita una psd con
            % area unitaria ovvero normalizzata sull'energia totale del
            % segnale
            elseif strcmp(parameter.normalization,'area_unit')
                PSD = PSD/sum(PSD);
            end
            % Definizione degli output
            % ordine utilizzato per la stima parametrica
            R_Teorica = order_param;
            % risoluzione apparente
            R_Apparente = parameter.ris_apparente;
            % percentuale di overlap perde di significato nella stima
            % parametrica
            Perc_Overlap = 'none';
            % finestra utilizzata perde di significato nella stima
            % parametrica
            window_vect = 'none';
            % mostro risultato finale se la variabile visualisation è
            % impostata su 'yes'
            if strcmp(parameter.visualisation,'yes')
                titolo = sprintf('Power Spettral Density Parametrica [%s]',method);
                figure()
                plot(Freq,PSD)
                xlabel('frequency [Hz]')
                ylabel('PSD')
                title(titolo)
                grid on 
                hold off
            end
        %##################################################################

    % METODO DELLA COVARIANZA MODIFICATA ##################################
    % input possibli : -ris_apparente -> risoluzione apparente della PSD
    %                                    di default impostata pari a 1 Hz
    %                  -order -> ordine massimo valutato dal metodo, di
    %                            default pari al 10% della lunghezza del
    %                            segnale arrotondato all'intero più vicino
    %                  - normalization -> 1 restituisce PSD normalizzata
    %                                     0 restituisce PSD non
    %                                     normalizzata
    %                                     di default verrà restituita la
    %                                     PSD normalizzata
    %                  - visualisation -> serve per mostrare o meno dei
    %                                     grafici con i passaggi intermedi
    %                                     o non, di default non verranno
    %                                     mostrati
        case 'covar_mod'
            % se non viene impostato un ordine massimo valutabile, verrà
            % considerato un ordine pari al 10% della lunghezza del segnale
            % arrotondato all'intero più vicino
            if strcmp(parameter.order,'none')
                parameter.order = round(0.1*length(x));
                if parameter.order == 1
                    parameter.order = 2;
                end
            end
            % inizializzazione vettore degli errori lungo pari all'ordine
            % assimo che si vuole analizzare -1
            e = zeros(1,parameter.order-1);
            % Calcolo della varianza asintotica per ordini del modello tra
            % 2 e parameter.order
            for NN = 2:parameter.order 
                [~,e(NN-1)] = armcov(x-mean(x),NN); 
            end
            % la varianza asintotica sarà il valore più piccolo ottenuto nel vettore
            % delgi errori ( l'ultimo)
            [m,I] = min(e);
            % varianza asintotica + 5% varianza asintotica
            asint = 1.05*m;
            % Trovare i valori di varianza che sono minori della varianza asintotica
            % aumentata del 5%
            ind =  find(e<asint);
            % mostro l'andamento nella valutazione dell'errore con il
            % relativo valore asintotico, il minimo e l'insieme dei valori
            % considerati per la scelta dell'ordine
            if strcmp(parameter.visualisation,'yes')
                % mostro l'andamento dell'errore al variare dell'ordine del modello
                figure()
                plot(2:ind(1)+1 ,e(1:ind(1)),'md-',ind(1)+1:ind(end)+1,e(ind),'cd-');
                hold on 
                plot(I+1,m,'ro','LineWidth',7);
                hold on
                plot(2:parameter.order,ones(1,length(e)).*asint,'b-')
                xlabel('ordine del modello')
                ylabel('errore')
                title('andamento dell errore in funzione dell ordine del modello')
                legend('errore','errore < della varianza asintotica + 5%','errore minimo','varianza asintotica + 5%')
                grid on
                hold off
            end
            % Scegliere l'ordine per il metodo parametrico 
            if length(ind)>=2
                order_param =  ind(2)+1;
            elseif length(ind)==1
                if parameter.order == 2
                    warning('ordine massimo valutato molto basso (%d)\nE'' stato valutato solo l''ordine 2\n',parameter.order)
                else
                    warning('ordine massimo valutato riduce di molto l''errore(%d)\nconsiderare  di prendere un ordine più piccolo\n',parameter.order)
                end
                order_param =  ind(1)+1;
            end
            % se non viene impostata una risoluzione apparente viene
            % considerata di default pari a 1Hz per avere un campione per
            % ogni frequenza unitaria
            if strcmp(parameter.ris_apparente,'none')
                parameter.ris_apparente = 1;
            end
            % numero di punti su cui calcolare la PSD
            NFFT = round(fs/parameter.ris_apparente);
            % calcolo la PSD
            [PSD, Freq] = pmcov(x-mean(x),order_param,NFFT,fs);
            % se normalization = yes la PSD viene restituita normalizzata
            if strcmp(parameter.normalization,'yes')
                PSD = PSD/max(PSD);
            % se normalization = 'area_unit' verrà restituita una psd con
            % area unitaria ovvero normalizzata sull'energia totale del
            % segnale
            elseif strcmp(parameter.normalization,'area_unit')
                PSD = PSD/sum(PSD);
            end
            % Definizione degli output
            % ordine utilizzato per la stima parametrica
            R_Teorica = order_param;
            % risoluzione apparente
            R_Apparente = parameter.ris_apparente;
            % percentuale di overlap perde di significato nella stima
            % parametrica
            Perc_Overlap = 'none';
            % finestra utilizzata perde di significato nella stima
            % parametrica
            window_vect = 'none';
            % mostro risultato finale se la variabile visualisation è
            % impostata su 'yes'
            if strcmp(parameter.visualisation,'yes')
                titolo = sprintf('Power Spettral Density Parametrica [%s]',method);
                figure()
                plot(Freq,PSD)
                xlabel('frequency [Hz]')
                ylabel('PSD')
                title(titolo)
                grid on 
                hold off
            end
        %##################################################################

    % METODO DI YULER-WALKER ##############################################
    % input possibli : -ris_apparente -> risoluzione apparente della PSD
    %                                    di default impostata pari a 1 Hz
    %                  -order -> ordine massimo valutato dal metodo, di
    %                            default pari al 10% della lunghezza del
    %                            segnale arrotondato all'intero più vicino
    %                  - normalization -> 1 restituisce PSD normalizzata
    %                                     0 restituisce PSD non
    %                                     normalizzata
    %                                     di default verrà restituita la
    %                                     PSD normalizzata
    %                  - visualisation -> serve per mostrare o meno dei
    %                                     grafici con i passaggi intermedi
    %                                     o non, di default non verranno
    %                                     mostrati
        case 'yuler'
            % se non viene impostato un ordine massimo valutabile, verrà
            % considerato un ordine pari al 10% della lunghezza del segnale
            % arrotondato all'intero più vicino
            if strcmp(parameter.order,'none')
                parameter.order = round(0.1*length(x));
                if parameter.order == 1
                    parameter.order = 2;
                end
            end
            % inizializzazione vettore degli errori lungo pari all'ordine
            % assimo che si vuole analizzare -1
            e = zeros(1,parameter.order-1);
            % Calcolo della varianza asintotica per ordini del modello tra
            % 2 e parameter.order
            for NN = 2:parameter.order 
                [~,e(NN-1)] = aryule(x-mean(x),NN); 
            end
            % la varianza asintotica sarà il valore più piccolo ottenuto nel vettore
            % delgi errori ( l'ultimo)
            [m,I] = min(e);
            % varianza asintotica + 5% varianza asintotica
            asint = 1.05*m;
            % Trovare i valori di varianza che sono minori della varianza asintotica
            % aumentata del 5%
            ind =  find(e<asint);
            % mostro l'andamento nella valutazione dell'errore con il
            % relativo valore asintotico, il minimo e l'insieme dei valori
            % considerati per la scelta dell'ordine
            if strcmp(parameter.visualisation,'yes')
                % mostro l'andamento dell'errore al variare dell'ordine del modello
                figure()
                plot(2:ind(1)+1 ,e(1:ind(1)),'md-',ind(1)+1:ind(end)+1,e(ind),'cd-');
                hold on 
                plot(I+1,m,'ro','LineWidth',7);
                hold on
                plot(2:parameter.order,ones(1,length(e)).*asint,'b-')
                xlabel('ordine del modello')
                ylabel('errore')
                title('andamento dell errore in funzione dell ordine del modello')
                legend('errore','errore < della varianza asintotica + 5%','errore minimo','varianza asintotica + 5%')
                grid on
                hold off
            end
            % Scegliere l'ordine per il metodo parametrico 
            if length(ind)>=2
                order_param =  ind(2)+1;
            elseif length(ind)==1
                if parameter.order == 2
                    warning('ordine massimo valutato molto basso (%d)\nE'' stato valutato solo l''ordine 2\n',parameter.order)
                else
                    warning('ordine massimo valutato riduce di molto l''errore(%d)\nconsiderare  di prendere un ordine più piccolo\n',parameter.order)
                end
                order_param =  ind(1)+1;
            end
            % se non viene impostata una risoluzione apparente viene
            % considerata di default pari a 1Hz per avere un campione per
            % ogni frequenza unitaria
            if strcmp(parameter.ris_apparente,'none')
                parameter.ris_apparente = 1;
            end
            % numero di punti su cui calcolare la PSD
            NFFT = round(fs/parameter.ris_apparente);
            % calcolo la PSD
            [PSD, Freq] = pyulear(x-mean(x),order_param,NFFT,fs);
            % se normalization = yes la PSD viene restituita normalizzata
            if strcmp(parameter.normalization,'yes')
                PSD = PSD/max(PSD);
            % se normalization = 'area_unit' verrà restituita una psd con
            % area unitaria ovvero normalizzata sull'energia totale del
            % segnale
            elseif strcmp(parameter.normalization,'area_unit')
                PSD = PSD/sum(PSD);
            end
            % Definizione degli output
            % ordine utilizzato per la stima parametrica
            R_Teorica = order_param;
            % risoluzione apparente
            R_Apparente = parameter.ris_apparente;
            % percentuale di overlap perde di significato nella stima
            % parametrica
            Perc_Overlap = 'none';
            % finestra utilizzata perde di significato nella stima
            % parametrica
            window_vect = 'none';
            % mostro risultato finale se la variabile visualisation è
            % impostata su 'yes'
            if strcmp(parameter.visualisation,'yes')
                titolo = sprintf('Power Spettral Density Parametrica [%s]',method);
                figure()
                plot(Freq,PSD)
                xlabel('frequency [Hz]')
                ylabel('PSD')
                title(titolo)
                grid on 
                hold off
            end
        %##################################################################

    % METODO DEL CORRELOGRAMMA ############################################
    % input possibli : - ris_teorica -> risoluzione teorica della PSD di
    %                                   default impostata pari alla massima
    %                                   risoluzione teorica possibile pari
    %                                   al rapporto tra la frequenza di
    %                                   campionamento e la lunghezza del
    %                                   segnale
    %                  -ris_apparente -> risoluzione apparente della PSD
    %                                    di default impostata pari al 10%
    %                                    della risoluzione teorica
    %                  -window -> finestra utilizzata per finestrare il
    %                             segnale durante il calcolo della PSD, di
    %                             default verrà considerata una finestra
    %                             rettangolare
    %                  - biased -> per definire se utilizzare la
    %                              crosscorrelazione polarizzata o meno:
    %                              1 -> crosscorrelazione polarizzata
    %                              0 -> crosscorrelazione non polarizzata
    %                              di default verrà impostato il metodo
    %                              polarizzato
    %                  - normalization -> 1 restituisce PSD normalizzata
    %                                     0 restituisce PSD non
    %                                     normalizzata
    %                                     di default verrà restituita la
    %                                     PSD normalizzata
    %                  - visualisation -> serve per mostrare o meno dei
    %                                     grafici con i passaggi intermedi
    %                                     o non, di default non verranno
    %                                     mostrati
    %        
        case 'correlogramma'
            % se non viene impostata una risoluzione teorica viene  
            % considerata di default pari al massimo possibile
            if strcmp(parameter.ris_teorica,'none')
                parameter.ris_teorica = fs/length(x);
                ntlag = round(fs/parameter.ris_teorica);
            else
                ntlag = round(fs/parameter.ris_teorica);
            end
            % se non viene impostata una risoluzione apparente viene
            % considerata di default pari al 10% della risoluzione teorica
            if strcmp(parameter.ris_apparente,'none')
                parameter.ris_apparente = 0.1*parameter.ris_teorica;
            end
            % numero di punti su cui calcolare la PSD
            NFFT = round(fs/parameter.ris_apparente);
            %   controllo sulla variabile biased
            %   se == 1 applicola la cross correlazione polarizzata
            if parameter.biased == 1
                acs = xcorr(x-mean(x),ntlag,'biased');
            %   se == 0 applico la crosscorrelazione non polarizzata
            elseif parameter.biased == 0
                acs = xcorr(x-mean(x),ntlag,'unbiased');
            %   se non è nessuna delle due restituisco un segnale di errore        
            else
                error('\nerrore sul parametro biased\n');
            end
            % creazione della finestra
            dim_finestra = ntlag*2+1;
            switch parameter.window
                case 'rectwin'
                    window_vect = rectwin(dim_finestra);
                case 'triang'
                    window_vect = triang(dim_finestra);
                case 'hann'
                    window_vect = hann(dim_finestra);
                case 'hamming'
                    window_vect = hamming(dim_finestra);
                otherwise
                    window_vect = rectwin(dim_finestra);
                    warning('!!!finestra non trovata!!!\n');
            end
            %   finestro il segnale, per farlo devo poltiplicare acs per la finestra
            %   scelta e i due vettori devono essere uno riga e l'altro colonna.
            %   non sapendo l'orientamento a priori faccio un controllo in cui applico
            %   eventualmente una trasposizione
            if (length(x(1,:))==1)&&(length(window_vect(1,:))~=1)
                acs = acs.*window_vect;
            else
                acs = acs'.*window_vect;
            end
            % mostro la cross-correlazione se la variabile visualisation è
            % impostata su 'yes'
            if strcmp(parameter.visualisation,'yes')
                titolo = sprintf('Crosscorrelazione finestrata');
                figure()
                plot(acs)
                xlabel('campioni')
                ylabel('XCorr')
                title(titolo)
                grid on 
                hold off
            end
            %   per la trasformata di furier è necessario che acs sia un vettore
            %   colonna per cui in caso contrario la traspongo
            if (length(acs(1,:))~=1)
                acs = acs';
            end
            %   calcolo la psd come il valore assoluto della trasformata di Fourier 
            %   della crosscorrelazione su un numero di punti pari a NFFT    
            PSD = abs(fft(acs,NFFT));
            %   rimuovo la duplicazione spettrale 
            PSD = PSD(1:length(PSD)/2+1);
            % se normalization = yes la PSD viene restituita normalizzata
            if strcmp(parameter.normalization,'yes')
                PSD = PSD/max(PSD);
            % se normalization = 'area_unit' verrà restituita una psd con
            % area unitaria ovvero normalizzata sull'energia totale del
            % segnale
            elseif strcmp(parameter.normalization,'area_unit')
                PSD = PSD/sum(PSD);
            end
            % Definizione degli output
            %asse delle frequenze
            Freq = (0:length(PSD)-1)/length(PSD)*(fs/2);
            % risoluzione teorica
            R_Teorica = parameter.ris_teorica;
            % risoluzione apparente
            R_Apparente = parameter.ris_apparente;
            % percentuale di overlap
            Perc_Overlap = 0;
            % mostro risultato finale se la variabile visualisation è
            % impostata su 'yes'
            if strcmp(parameter.visualisation,'yes')
                titolo = sprintf('Power Spettral Density [%s]',method);
                figure()
                plot(Freq,PSD)
                xlabel('frequency [Hz]')
                ylabel('PSD')
                title(titolo)
                grid on 
                hold off
            end
        %##################################################################
        otherwise
            fprintf('metodo non trovato')           
    end

end