# server.py

def run_server():
    import socket                   # Import socket module
    import serial                   # Import library for serial communication
    import os                       # 
    import subprocess               # import library for command line interface
    import time                     # for delays and sleep
    
    ser = serial.Serial('/dev/ttyUSB0',115200) # open UArt port on usb between pi and arduino

    port = 50001                    # Reserve a port for your service every new transfer wants a new port or you must wait.
    s = socket.socket()             # Create a socket object
    #host = "192.168.0.18"          # server ip from if config at home
    host = '192.168.137.98'         # serve ip at school from laptop hotspot using ifconfig
    s.bind((host, port))            # Bind to the port
    s.listen(5)                     # Now wait for client connection.

    print ('Server listening....')

    while True:
        conn, addr = s.accept()     # Establish connection with client.
        print ('Got connection from', addr)  
        # ________ Record audio file
        recordCMD = "arecord -D hw:1,0 -d 5 -f cd -r 48000 rand.wav -c 1"
        runAgain = input("Do you want to start? Enter y/n: ")   #v hitting y and enter will automatically start recording
        if runAgain == "y":
            print("Will execute in command line\n"+ recordCMD)  # for debugging
            p = subprocess.Popen(recordCMD, shell=True, stdout=subprocess.PIPE)
            print ("recording")
            time.sleep(5) # since recording is for 5 seconds wait that long 
            print ("Done rec.")
        else
            break    
        # at this point you have a wav file 
        #data = conn.recv(1024)   receives hello from client 
        #print('Server received', repr(data))
        
        f = open('rand.wav','rb')
        # conn.send(fileName.encode())
      
        dataPacket = f.read(1024)  # buffering file in packets to send to client
        while (dataPacket):
            conn.send(dataPacket)
            print('Sent ',repr(dataPacket))
            print('\n')
            dataPacket = f.read(1024)
        f.close()

        print('Done sending file')

        result = conn.recv(1024)      # receives result from client
        ser.write(result.encode())    # sends result to hadrware 
        conn.send(b'Thank you for connecting')
        conn.close()
if __name__ == "__main__":
    run_server()
