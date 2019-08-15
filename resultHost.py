import socket                   # Import socket module

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)              # Create a socket object
host = '192.168.0.12'            # ip address of server at homey

#host = '192.168.137.98'         # ip address of server at schoool
port = 50001                    # Reserve a port for your service every new transfer wants a new port or you must wait.
buffSize = 1024
s.connect((host, port))

s.send("result host".encode())

fileName="rand.wav"

# doing fraud detection
# doing speaker recognition

# get result  
# s.send(result.encode())   # send result to server
s.close() #you dont want to close connection with socket 
print("connection closed")