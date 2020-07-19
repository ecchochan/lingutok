**An Linguistic Tokenizer for English**


# Purpose

```
Morphology is the study of the internal structure of words and forms a core part of linguistic study today.

cane, caning, can, canning
transdisciplinary
deanonymization
ignorantly

```


```prolog
morph(P)olog(R)y(S) is(R) the(R) study(R) of(R) the(R) interne(R)al(S) struct(R)ure(S) of(R) word(R)s(S) and(R) forms(R) a(R) core(R) part(R) of(R) lingu(R)ist(S)ic(S) study(R) to(P)day(R) .

cane(R) , cane(R)ing(S) , can(R) , can(R)ing(S)
trans(P)discipline(R)ary(S)
de(P)anonym(R)ize(S)ation(S)
ignore(R)ant(S)ly(S)
```


# Features

1. Tokenize English words into meaningful parts by brute-force exploring possible segmentations
2. 


# Installation

```bash
sudo pip3 install git+https://github.com/ecchochan/lingutok.git 
```


# Usage

```python
import lingutok
lingutok.load()

encoded = lingutok.tokenize('Happy Birthday to me')
>> Encoded('happy (R) birth (R) day (S) to (R) me (R)')

encoded.ids
>> [9366, 7154, 5864, 12623, 10309]

encoded.offsets
>> [0, 6, 6, 15, 18]

encoded.offsets_span
>> [(0, 6), (6, 15), (6, 15), (15, 18), (18, 20)]

encoded.casing
>> [True, True, True, False, False]

encoded.size
>> 5

encoded.text
>> 'Happy Birthday to me'


encoded2 = lingutok.tokenize('Untitled Last Checkpoints')
>> Encoded('un (P) title (R) ed (S) last (R) check (R) point (R) s (S)')

```


# Customization
```python
import lingutok

# Set the data directory
lingutok.set_root('/mnt/d/gits/test/resources')

# Generate the vocab file
data_dir = './'
data_fn = 'my-lingutok'
lingutok.generate_trie(data_dir, data_fn)

# Load the vocab files
lingutok.load(data_dir, data_fn)



```
