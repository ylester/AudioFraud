import socket
import fraud
import speaker_recognition

"""This module is not complete.. Do NOT push to master"""

# The socket class configures the server
# and listens to incoming connections
s = socket.socket()

# Binds the socket to all IP addresses on local machine
s.bind(('0.0.0.0', 9600))

# Server listens to connections
s.listen(0)

# Constantly checks server to receive data from client
while True:
    # Client handling code
    client, addr = s.accept()

    while True:
        content = client.recv(32)
        if len(content) == 0:
            break
        else:
            # Send content to fraud detector module
            # and speaker recognition module
            fraud.sendaudio(content)
            speaker_recognition.sendaudio(content)