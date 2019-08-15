import socket                   # Import socket module

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)              # Create a socket object
host = '192.168.0.12'            # ip address of server at homey

#host = '192.168.137.98'         # ip address of server at schoool
port = 50001                    # Reserve a port for your service every new transfer wants a new port or you must wait.
buffSize = 1024
s.connect((host, port))

s.send("file host".encode())

fileName=s.recv(1024).decode()
print(fileName)

with open(fileName, 'wb') as f:
    print ("file opened")
    while True:
        print("receiving data...")
        data = s.recv(1024)
        print('data=%s', (data))
        if not data:
            break
        # write data to a file
        f.write(data)

f.close()
print("Successfully get the file")
# doing fraud dection
# doing speaker recognitio
# result = "this a test"
# s.send(result.encode())   # send result to server
s.close() #you dont want to close connection with socket 
print("connection closed")