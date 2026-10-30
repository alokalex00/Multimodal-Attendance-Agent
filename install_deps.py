"""Fast dependency installer forcing IPv4 to prevent Windows IPv6 DNS stall."""
import socket
import sys

# Force IPv4 socket resolution
orig_getaddrinfo = socket.getaddrinfo


def getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


socket.getaddrinfo = getaddrinfo_ipv4

from pip._internal.cli.main import main

if __name__ == "__main__":
    args = sys.argv[1:]
    sys.exit(main(args))
