server = tcpserver('127.0.0.1', 5000, 'ConnectionChangedFcn', @connectionStatus); 
disp('TCP Server is waiting for connection...');

% Callback to indicate when the client connects/disconnects
function connectionStatus(serverObj, ~)
    if serverObj.Connected
        disp('Client connected! Sending data...');
        
        % Send an array of strings or some data to the client
        dataToSend = 'ECG signal 999, 222, 111';
        write(serverObj, uint8(dataToSend));  % Convert string to uint8 to send via TCP
        
        disp('Data sent to client.');
    else
        disp('Client disconnected.');
    end
end
