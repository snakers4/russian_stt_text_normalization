import re
import torch
from string import printable, punctuation
from tqdm import tqdm


class Normalizer():
    def __init__(self,
                 device='cpu',
                 jit_model='jit_s2s.pt'):
        super(Normalizer, self).__init__()

        self.device = torch.device(device)

        self.init_vocabs()

        self.model = torch.jit.load(jit_model)
        self.model = self.model.to(self.device)
        self.model.eval()

    def init_vocabs(self):
        # vocabs
        rus_letters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
        spec_symbols = '¼³№¾⅞½⅔⅓⅛⅜²'
        # numbers + eng + punctuation + space + rus
        self.src_vocab = {token: i+5 for i, token in enumerate(printable[:-5] + rus_letters + '«»—' + spec_symbols)}
        # punctuation + space + rus
        self.tgt_vocab = {token: i+5 for i, token in enumerate(punctuation + rus_letters + ' ' + '«»—')}

        unk = '#UNK#'
        pad = '#PAD#'
        sos = '#SOS#'
        eos = '#EOS#'
        tfo = '#TFO#'
        for i, token in enumerate([unk, pad, sos, eos, tfo]):
            self.src_vocab[token] = i
            self.tgt_vocab[token] = i

        for i, token_name in enumerate(['unk', 'pad', 'sos', 'eos', 'tfo']):
            setattr(self, '{}_index'.format(token_name), i)

        inv_src_vocab = {v: k for k, v in self.src_vocab.items()}
        self.src2tgt = {src_i: self.tgt_vocab.get(src_symb, -1) for src_i, src_symb in inv_src_vocab.items()}

    def keep_unknown(self, string):
        reg = re.compile(r'[^{}]+'.format(''.join(self.src_vocab.keys())))
        unk_list = re.findall(reg, string)

        unk_ids = [range(m.start()+1, m.end()) for m in re.finditer(reg, string) if m.end() - m.start() > 1]
        flat_unk_ids = [i for sublist in unk_ids for i in sublist]

        upd_string = ''.join([s for i, s in enumerate(string) if i not in flat_unk_ids])
        return upd_string, unk_list

    def norm_string(self, string):
        # assert len(string) < 200

        if len(string) == 0:
            return string
        string, unk_list = self.keep_unknown(string)

        token_src_list = [self.src_vocab.get(s, self.unk_index) for s in list(string)]
        src = token_src_list + [self.eos_index] + [self.pad_index]

        src2tgt = [self.src2tgt[s] for s in src]
        src2tgt = torch.LongTensor(src2tgt).to(self.device)

        src = torch.LongTensor(src).unsqueeze(0).to(self.device)
        out = self.model(src, src2tgt)
        pred_words = self.decode_words(out, unk_list)
        return pred_words

    def norm_text(self, text):
        abstracts = text.split('\n')
        res_abstracts = []
        for abstract in tqdm(abstracts):
            temp_result = ''
            result = []
            words = abstract.split(' ')
            for i, word in enumerate(words):
                temp_result = ' '.join([temp_result, word])
                if len(temp_result) > 150 or len(temp_result) > 100 and not re.search(r'[0-9a-zA-Z]', word) or i == len(words) - 1:
                    result.append(self.norm_string(temp_result.strip(' ')))
                    temp_result = ''
            res_abstracts.append(' '.join(result))
        return '\n'.join(res_abstracts)

    def decode_words(self, pred, unk_list=[]):
        pred = pred.cpu().numpy()
        pred_words = "".join(self.lookup_words(x=pred,
                                               vocab={i: w for w, i in self.tgt_vocab.items()},
                                               unk_list=unk_list))
        return pred_words

    def lookup_words(self, x, vocab, unk_list=[]):
        result = []
        for i in x:
            if i == self.unk_index:
                if len(unk_list) > 0:
                    result.append(unk_list.pop(0))
                else:
                    continue
            else:
                result.append(vocab[i])
        return [str(t) for t in result]
