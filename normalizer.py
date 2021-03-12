import re
import torch
from string import printable, punctuation
from tqdm import tqdm
import warnings


class Normalizer:
    def __init__(self,
                 device='cpu',
                 jit_model='jit_s2s.pt'):
        super(Normalizer, self).__init__()

        self.device = torch.device(device)

        self.init_vocabs()

        self.model = torch.jit.load(jit_model, map_location=device)
        self.model.eval()
        self.max_len = 150

    def init_vocabs(self):
        # Initializes source and target vocabularies

        # vocabs
        rus_letters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
        spec_symbols = '¼³№¾⅞½⅔⅓⅛⅜²'
        # numbers + eng + punctuation + space + rus
        self.src_vocab = {token: i + 5 for i, token in enumerate(printable[:-5] + rus_letters + '«»—' + spec_symbols)}
        # punctuation + space + rus
        self.tgt_vocab = {token: i + 5 for i, token in enumerate(punctuation + rus_letters + ' ' + '«»—')}

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

        unk_ids = [range(m.start() + 1, m.end()) for m in re.finditer(reg, string) if m.end() - m.start() > 1]
        flat_unk_ids = [i for sublist in unk_ids for i in sublist]

        upd_string = ''.join([s for i, s in enumerate(string) if i not in flat_unk_ids])
        return upd_string, unk_list

    def _norm_string(self, string):
        # Normalizes chunk

        if len(string) == 0:
            return string
        string, unk_list = self.keep_unknown(string)

        token_src_list = [self.src_vocab.get(s, self.unk_index) for s in list(string)]
        src = token_src_list + [self.eos_index] + [self.pad_index]

        src2tgt = [self.src2tgt[s] for s in src]
        src2tgt = torch.LongTensor(src2tgt).to(self.device)

        src = torch.LongTensor(src).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(src, src2tgt)
        pred_words = self.decode_words(out, unk_list)
        if len(pred_words) > 199:
            warnings.warn("Sentence {} is too long".format(string), Warning)
        return pred_words

    def norm_text(self, text):
        # Normalizes text

        # Splits sentences to small chunks with weighted length <= max_len:
        # * weighted length - estimated length of normalized sentence
        #
        # 1. Full text is splitted by "ending" symbols (\n\t?!.) to sentences;
        # 2. Long sentences additionally splitted to chunks: by spaces or just dividing too long words

        splitters = '\n\t?!'
        parts = [p for p in re.split(r'({})'.format('|\\'.join(splitters)), text) if p != '']
        norm_parts = []
        for part in tqdm(parts):
            if part in splitters:
                norm_parts.append(part)
            else:
                weighted_string = [7 if symb.isdigit() else 1 for symb in part]
                if sum(weighted_string) <= self.max_len:
                    norm_parts.append(self._norm_string(part))
                else:
                    spaces = [m.start() for m in re.finditer(' ', part)]
                    start_point = 0
                    end_point = 0
                    curr_point = 0

                    while start_point < len(part):
                        if curr_point in spaces:
                            if sum(weighted_string[start_point:curr_point]) < self.max_len:
                                end_point = curr_point + 1
                            else:
                                norm_parts.append(self._norm_string(part[start_point:end_point]))
                                start_point = end_point

                        elif sum(weighted_string[end_point:curr_point]) >= self.max_len:
                            if end_point > start_point:
                                norm_parts.append(self._norm_string(part[start_point:end_point]))
                                start_point = end_point
                            end_point = curr_point - 1
                            norm_parts.append(self._norm_string(part[start_point:end_point]))
                            start_point = end_point
                        elif curr_point == len(part):
                            norm_parts.append(self._norm_string(part[start_point:]))
                            start_point = len(part)

                        curr_point += 1
        return ''.join(norm_parts)

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
