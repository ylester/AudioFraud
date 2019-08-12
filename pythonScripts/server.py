# server.py

def run_server():
    import socket                   # Import socket module
    import serial                   # Import library for serial communication
    import easygui
    import os
    ser = serial.Serial('/dev/ttyUSB0',115200) # open UArt port on usb between pi and arduino

    port = 50001                    # Reserve a port for your service every new transfer wants a new port or you must wait.
    s = socket.socket()             # Create a socket object
    #host = "192.168.0.18"           # server ip from if config at home
    host = '192.168.137.52'       # serve ip at school from laptop hotspot using ifconfig
    s.bind((host, port))            # Bind to the port
    s.listen(5)                     # Now wait for client connection.

    print ('Server listening....')

    while True:
        conn, addr = s.accept()     # Establish connection with client.
        print ('Got connection from', addr)
        data = conn.recv(1024)
        print('Server received', repr(data))

        filePath= easygui.fileopenbox(msg="Choose a wav file", title=None, default="*.wav", filetypes='*.wav')  #gui to chose file
        fileName= os.path.basename(filePath)
        f = open(fileName,'rb')
        conn.send(fileName.encode())
        ser.write(fileName.encode())
        dataPacket = f.read(1024)
        while (dataPacket):
            conn.send(dataPacket)
            print('Sent ',repr(dataPacket))
            print('\n')
            dataPacket = f.read(1024)
        f.close()

        print('Done sending')
        conn.send(b'Thank you for connecting')
        conn.close()
if __name__ == "__main__":
    run_server()