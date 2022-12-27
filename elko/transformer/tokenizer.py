import sys


def ascii_tokenizer(x: str, cutoff: int = 128):
    return list(filter(lambda x: x < cutoff, [ord(xx) for xx in x]))


def name_tokenizer(x: str):
    """
    a very simple tokenizer, assumes that these are names composed
    of the 26 characters of the alphabet
    """
    return [0] + [ord(xi) - ord("a") for xi in x.lower().strip()] + [26]

class BPETokenizer():

    def __init__(self, file_paths:list[str]):

        self.file_paths = file_paths
        self.frequencies = Counter()

    def process_files(self):
        for path in self.file_paths:
            tokens = list(open(path, 'r').read().encode('utf-8'))
            self.frequencies += Counter(tokens)
    
    def merge(target_vocab_size:int):
        cur_vocab_size = len(self.frequencies)
        remapped_
        while cur_vocab_size > target_vocab_size:
            (t1, c1), (t2, c2) = self.frequencies.most_common(2)
            self.frequencies[f'm[{t1}][{t2}]']


if __name__ == "__main__":
    print(sys.argv[1], ascii_tokenizer(sys.argv[1]))
