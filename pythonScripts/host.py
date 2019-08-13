import socket                   # Import socket module

s = socket.socket()              # Create a socket object
#host = '192.168.0.18'            # ip address of server at home
host = '192.168.137.98'         # ip address of server at schoool
port = 50000                     # Reserve a port for your service every new transfer wants a new port or you must wait.

s.connect((host, port))
s.send("Hello server!".encode())   # send hello to server not needed but done to ensure connection is established

fileName=s.recv(1024).decode()
print(fileName)
with open(fileName, 'wb') as f:
    print ('file opened')
    while True:
        print('receiving data...')
        data = s.recv(1024)
        print('received', (data))
        print('\n')
        if not data:  # if there is no file break
            break
        # write data to a file
        f.write(data)  # if fiel data exists write the file
# doing fraud dection
# doing speaker recognition
# result
s.send(result.encode())   # send result to server
f.close()
print('Successfully get the file')
# s.close() you dobt want to close connection with socket 
print('connection closed')
