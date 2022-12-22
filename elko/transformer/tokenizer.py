import sys

def ascii_tokenizer(x:str, cutoff:int=128):
    return list(filter(lambda x: x < cutoff, [ord(xx) for xx in x]))


if __name__ == '__main__':
    print(sys.argv[1], ascii_tokenizer(sys.argv[1]))
