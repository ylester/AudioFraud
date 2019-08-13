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
    host = '192.168.137.52'         # serve ip at school from laptop hotspot using ifconfig
    s.bind((host, port))            # Bind to the port
    s.listen(5)                     # Now wait for client connection.

    print ('Server listening....')

    while True:
        conn, addr = s.accept()     # Establish connection with client.
        # ________ Record audio file
        recordCMD = "arecord -D hw:1,0 -d 5 -f cd -r 48000 rand.wav -c 1"
        runAgain = input("Do you want to start? Enter y/n: ")
        if runAgain == "y":
            print("Will execute in command line\n"+ recordCMD)  # for debugging
            p = subprocess.Popen(recordCMD, shell=True, stdout=subprocess.PIPE)
            print ("recording")
            time.sleep(5)
            print ("Done rec.")
        else
            break    
        print ('Got connection from', addr)
        data = conn.recv(1024)
        print('Server received', repr(data))
        f = open('data/rand.wav','rb')
        # conn.send(fileName.encode())
        ser.write(result.encode())    # sends result to hadrware
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
