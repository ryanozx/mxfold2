from itertools import groupby
from torch.utils.data import Dataset
import torch
import math

class FastaDataset(Dataset):
    def __init__(self, fasta):
        it = self.fasta_iter(fasta)
        try:
            self.data = list(it)
        except RuntimeError:
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def fasta_iter(self, fasta_name):
        fh = open(fasta_name)
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))

        for header in faiter:
            # drop the ">"
            headerStr = header.__next__()[1:].strip()

            # join all sequence lines to one.
            seq = "".join(s.strip() for s in faiter.__next__())

            yield (headerStr, seq, torch.tensor([]))


class BPseqDataset(Dataset):
    """
    Given a .lst file containing the locations of .bpseq files, BPseqDataset creates a dataset that contains
    (filename, sequence, pairing_tensor) tuples
    """
    def __init__(self, bpseq_list):
        self.data = []
        with open(bpseq_list) as list_file:
            for file_entry in list_file:
                file_entry = file_entry.rstrip('\n').split()
                if len(file_entry) == 1:
                    bpseq_file_path = file_entry[0]
                    self.data.append(self.read(bpseq_file_path))
                elif len(file_entry) == 2:
                    fasta_file_path, label_file_path = file_entry
                    self.data.append(self.read_pdb(fasta_file_path, label_file_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def read(self, filename):
        with open(filename) as bpseq_file:
            is_structure_known = True
            base_pairs = [0]
            bases = ['']
            for line in bpseq_file:
                if line.startswith('#'):
                    # skip comment lines
                    continue

                line = line.rstrip('\n').split()

                if len(line) == 3:
                    # each nucleotide is represented by a line in the bpseq file, with the format <position> <base> <pairing partner (0 if none)>
                    if not is_structure_known:
                        raise('invalid format: {}'.format(filename))
                    idx, base, pair = line
                    # if the pair is a special symbol, we map them to a negative value
                    pos = 'x.<>|'.find(pair)
                    if pos >= 0:
                        idx, pair = int(idx), -pos
                    else:
                        idx, pair = int(idx), int(pair)
                    bases.append(base)
                    base_pairs.append(pair)
                elif len(line) == 4:
                    raise('disabled handling files with probs')
                    is_structure_known = False
                    idx, base, nll_unpaired, nll_paired = line
                    bases.append(base)
                    nll_unpaired = math.nan if nll_unpaired=='-' else float(nll_unpaired)
                    nll_paired = math.nan if nll_paired=='-' else float(nll_paired)
                    base_pairs.append([nll_unpaired, nll_paired])
                else:
                    raise('invalid format: {}'.format(filename))
        
        if is_structure_known:
            seq = ''.join(bases)
            return (filename, seq, torch.tensor(base_pairs))
        else:
            seq = ''.join(bases)
            base_pairs.pop(0)
            return (filename, seq, torch.tensor(base_pairs))

    def fasta_iter(self, fasta_name):
        fh = open(fasta_name)
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))

        for header in faiter:
            # drop the ">"
            headerStr = header.__next__()[1:].strip()

            # join all sequence lines to one.
            seq = "".join(s.strip() for s in faiter.__next__())

            yield (headerStr, seq)

    def read_pdb(self, seq_filename, label_filename):
        it = self.fasta_iter(seq_filename)
        h, seq = next(it)

        p = []
        with open(label_filename) as f:
            for l in f:
                l = l.rstrip('\n').split()
                if len(l) == 2 and l[0].isdecimal() and l[1].isdecimal():
                    p.append([int(l[0]), int(l[1])])

        return (h, seq, torch.tensor(p))