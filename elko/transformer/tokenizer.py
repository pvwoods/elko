import sys


def ascii_tokenizer(x: str, cutoff: int = 128):
    return list(filter(lambda x: x < cutoff, [ord(xx) for xx in x]))


def name_tokenizer(x: str):
    """
    a very simple tokenizer, assumes that these are names composed
    of the 26 characters of the alphabet
    """
    return [0] + [ord(xi) - ord("a") for xi in x.lower().strip()] + [26]


if __name__ == "__main__":
    print(sys.argv[1], ascii_tokenizer(sys.argv[1]))
