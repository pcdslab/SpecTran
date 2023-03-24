from functools import total_ordering
from collections import defaultdict
import random
from sortedcollections import SortedList

@total_ordering
class VocabToken:
    def __init__(self, mzs, token_index, priority):
        self.mzs = mzs
        self.token_index = token_index
        self.priority = priority
        self.length = len(mzs)

    def _is_valid_operand(self, other):
        return hasattr(other, "mzs") and hasattr(other, "priority")

    def __eq__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.mzs == other.mzs

    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented

        # longer tokens always have higher priority than lower tokens, since we attempt to tokenize with them first
        if (len(self.mzs) > len(other.mzs)):
            return True

        if (len(self.mzs) < len(other.mzs)):  
            return False

        # for tokens of the same length, order by highest priority (frequency in the training data)
        return self.priority > other.priority

    def __hash__(self) -> int:
        return hash(tuple(self.mzs))

class BasicSpectraVocab:
    def __init__(self, max_token_length):
        self.mz_to_tokens = defaultdict(lambda:SortedList()) # The map of each single m/z value to the tokens that it appears in
        self.all_tokens = []
        self.max_token_length = max_token_length
        self.tokens_by_length = defaultdict(lambda:SortedList()) # keep track of tokens by their length (number of m/z values)

        # keep reference to which token is the mask and which is the unknown so we can look them up directly (these are set when loading the vocab)
        self.mask_token = None
        self.unk_token = None
        self.pad_token = None

    # Load the vocabulary from the file. Each line of the vocabulary must be a token (a single m/z value or a pair) and its frequency.
    # This assumes that the vocabulary file is sorted from most to least frequent 
    def load_from_file(self, filename):
        with open(filename, "r") as inf:
            lines = inf.readlines()

            current_ix = 0
            for line in lines:
                splitLine = line.replace("\n", "")
                splitLine = splitLine.replace(")", "")
                splitLine = splitLine.replace("(", "")
                splitLine = splitLine.replace(" ", "")
                splitLine = splitLine.split(",")

                if len(splitLine) == 2:
                    # single token (mz, frequency)
                    freq = int(splitLine[1])
                    mzs = [splitLine[0]]
                else:
                    # pair token, ((mz1, mz2), frequency)
                    freq = int(splitLine[2])
                    mzs = [splitLine[0], splitLine[1]]

                new_token = VocabToken(mzs=mzs, token_index=current_ix, priority=freq)        
                self._add_token(new_token, isSpecial=False)
                current_ix += 1

        # add the special tokens, which are considered to have the lowest frquency
        special_tokens = ['<pad>', '<mask>', '<unk>']

        first_special_index = current_ix

        for specialTokenIndex in range(0, len(special_tokens)):
            index = first_special_index + specialTokenIndex

            token = special_tokens[specialTokenIndex]

            token_to_add = VocabToken(mzs=[token], token_index=index, priority=0)
            
            if token == '<mask>':
                self.mask_token = token_to_add

            if token == '<unk>':
                self.unk_token = token_to_add  

            if token == '<pad>':
                self.pad_token = token_to_add      

            self._add_token(token_to_add, isSpecial=True)      

    def _add_token(self, vocabToken, isSpecial):
        self.all_tokens.append(vocabToken)

        if not isSpecial:
            self.tokens_by_length[vocabToken.length].add(vocabToken)
            for mz in vocabToken.mzs:
                self.mz_to_tokens[int(mz)].add(vocabToken)

    # returns: sortedList of tokens that contain this spectrum, based on their custom comparator
    # TODO: consider optimizing this to only return tokens that start with the peak, since we process the peaks in order 
    def tokens_containing_peak(self, mz_int):
        return self.mz_to_tokens[mz_int]

    def get_mask_token(self):
        return self.mask_token    

    # When choosing a random token in the masking process, we ensure that we use a token with the same length (number of m/z values)
    def get_random_token(self, token_length):   
        return random.choice(self.tokens_by_length[token_length]) 