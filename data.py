import torch
import pdb

# handles movies.txt data
class Dataset(torch.utils.data.Dataset):
    def __init__(self, txt_file):
        super(Dataset, self).__init__()
        self.txt_file = txt_file
        with open(self.txt_file, encoding='utf-8') as fin:
            self.corpus = fin.readlines()
        self.seq_len = 3
        #self.corpus = self.corpus[:500]
        self.corpus = [sentence for sentence in self.corpus if len(sentence.split()) >= self.seq_len]
        print(f'Total number of topics: {len(self.corpus)}')

        # build vocabulary        
        self.build_vocab(outfile="vocab.txt")

        # create sequences of length 20 (or seq_len)
        self.sequences = [self.create_sequence(sentence, seq_len=self.seq_len) for sentence in self.corpus]
        # merge lists of lists into a singel list
        self.sequences = sum(self.sequences, [])
        batch_size = 2 # 32
        maxsamples = len(self.sequences) // batch_size
        self.sequences = self.sequences[:maxsamples*batch_size]


    def __len__(self):
        return len(self.sequences)


    def __getitem__(self, index):
        seq = self.sequences[index]
        tokens = seq.split()
        x = tokens[:-1]
        y = tokens[1:]
        # convert to indices
        unk_token_ix = self.word_to_ix['<unk>']
        x = [self.word_to_ix.get(tok, unk_token_ix) for tok in x]
        y = [self.word_to_ix.get(tok, unk_token_ix) for tok in y]
        return torch.LongTensor(x), torch.LongTensor(y)


    def create_sequence(self, text, seq_len=5):
        tokens = text.split()
        sequences = []
        # if len of tokens is >= seq_len
        if len(tokens) >= seq_len:
            for i in range(0, len(tokens)-seq_len+1):
                seq = tokens[i: i+seq_len]
                sequences.append(' '.join(seq))
            return sequences
        
        # if len of tokens is < seq_len
        pdb.set_trace()
        return [text]


    def build_vocab(self, outfile=None):
        # get a unique list of words in the corpus
        vocab = set(word for sentence in self.corpus for word in sentence.split())
        vocab = sorted(list(vocab))
        # create word_to_ix map
        word_to_ix = {'<pad>': 0, '<unk>': 1}
        for word in vocab:
            word_to_ix[word] = len(word_to_ix)

        # create index to word map
        ix_to_word= {index: word for word, index in word_to_ix.items()}
        print(f'Number of unique words in vocabulary: {len(word_to_ix)}')
        self.word_to_ix = word_to_ix
        self.ix_to_word = ix_to_word

        with open(outfile, "w", encoding='utf-8') as fout:
            for word, ix in self.word_to_ix.items():
                if ix != 0:
                    fout.write('\n')
                fout.write(word)

if __name__ == '__main__':
    ds = Dataset('telugu.txt')
    print(ds.corpus[0])