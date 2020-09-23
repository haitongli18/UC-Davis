import socket

port = 5065
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('', port))

while True:
    bin_data = s.recvfrom(1024)
    string = bin_data[0].decode()
    print(string)
