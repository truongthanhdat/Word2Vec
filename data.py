import numpy as np

#corpus_raw = 'He is the king . The king is royal . She is the royal  queen '
corpus_raw = 'A few months after Hindleys return     Heathcliff and Catherine walk to Thrushcross Grange to spy on Edgar and Isabella Linton   who live there .  After being discovered   they try to run away but are caught .  Catherine is injured by the Lintons dog and taken into the house to recuperate   while Heathcliff is sent home .  Catherine stays with the Lintons .  The Lintons are landed gentry and Catherine is influenced by their elegant appearance and genteel manners .  When she returns to Wuthering Heights   her appearance and manners are more ladylike   and she laughs at Heathcliffs unkempt appearance .  The next day   knowing that the Lintons are to visit   Heathcliff   upon Nellys advice   tries to dress up   in an effort to impress Catherine   but he and Edgar get into an argument and Hindley humiliates Heathcliff by locking him in the attic .  Catherine tries to comfort Heathcliff   but he vows revenge on Hindley .  The following year   Frances Earnshaw gives birth to a son   named Hareton   but she dies a few months later .  Hindley descends into drunkenness .  Two more years pass   and Catherine and Edgar Linton become friends   while she becomes more distant from Heathcliff .  Edgar visits Catherine while Hindley is away and they declare themselves lovers soon afterwards .  Catherine confesses to Nelly that Edgar has proposed marriage and she has accepted   although her love for Edgar is not comparable to her love for Heathcliff   whom she cannot marry because of his low social status and lack of education .  She hopes to use her position as Edgars wife to raise Heathcliffs standing .  Heathcliff overhears her say that it would "degrade" her to marry him (but not how much she loves him)   and he runs away and disappears without a trace .  Distraught over Heathcliffs departure   Catherine makes herself ill .  Nelly and Edgar begin to pander to her every whim to prevent her from becoming ill again . '
# convert to lower case
corpus_raw = corpus_raw.lower()

words = []
for word in corpus_raw.split():
    if word != '.' and word != ',': # because we don't want to treat . as a word
        words.append(word)

words = set(words) # so that all duplicate words are removed
word2int = {}
int2word = {}
vocab_size = len(words) # gives the total number of unique words

for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

# raw sentences is a list of sentences.
raw_sentences = corpus_raw.split('.')
sentences = []
for sentence in raw_sentences:
    sentences.append(sentence.split())

WINDOW_SIZE = 4

data = []
for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :
            if nb_word != word:
                data.append([word, nb_word])

# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

x_train = [] # input word
y_train = [] # output word

for data_word in data:
    x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))
    y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))

# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
