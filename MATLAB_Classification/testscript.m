ipAddress = '127.0.0.1';
port = 5001;

tcpObj = tcpclient(ipAddress, port, 'Timeout', 20);

disp(['Connection Status: ', tcpObj.Status]);

if strcmp(tcpObj.Status, 'open')

    try
        dataReceived = read(tcpObj);

        receivedString = char(dataReceived);
        
        disp(['Data received: ', receivedString]);

        dataColumns = strsplit(receivedString, ',');
        
        dataColumnsNumeric = str2double(dataColumns);

        fs = 125;
        [b, a] = butter(4, [0.5 10] / (fs / 2));
        
        filteredECG = filtfilt(b, a, dataColumnsNumeric);

        ecgData = filteredECG;

        ecgData = ecgData(:)'; 
        
        targetSize = 187;

        if length(ecgData) < targetSize
            paddedECG = [ecgData, zeros(1, targetSize - length(ecgData))];
        end
        
        reshapedECG = reshape(paddedECG, 1, targetSize);

        ecgsignal = (reshapedECG - min(reshapedECG) ./ max(reshapedECG) - min(reshapedECG));

        load("trainedECGModel1.mat");

        predictedClass = predict(model, ecgsignal);

        disp(['The predicted class is: ', predictedClass]);

        predictedClassStr = char(predictedClass);

        disp(predictedClassStr);
        
        write(tcpObj, predictedClassStr);


    catch ME
        disp(['Error reading data: ', ME.message]);
    end
else
    disp('Failed to connect to the server.');
end
