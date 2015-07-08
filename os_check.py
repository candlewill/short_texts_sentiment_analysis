__author__ = 'NLP-PC'
from os import name as os_name


def get_os_name():
    os_n = os_name
    if os_n == 'posix':
        out = 'ubuntu'
    elif os_n == 'nt':
        out = 'windows'
    else:
        out = None
    return out


if __name__ == '__main__':
    print(get_os_name())
