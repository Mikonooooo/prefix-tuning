# prefix-tuning

## Setup
Install conda environment from `environment.yaml`

For JDK
- Java 1.8
- Perl 5.8.8 or higher with the [XML::Twig](http://search.cpan.org/~mirod/XML-Twig-3.49/Twig.pm) CPAN module


```
./e2e-metrics/measure_scores.py src/target.txt src/model-output.txt 
```

NO BEAM
|Cached Prefix       | Not Cached Prefix-Tuned   | Fine-tuned
|==============      | ==============            | ==============
|BLEU: 0.6480        | BLEU: 0.6571              | BLEU: 0.6721
|NIST: 8.3238        | NIST: 8.3984              | NIST: 8.5870
|METEOR: 0.4431      | METEOR: 0.4439            | METEOR: 0.4546
|ROUGE_L: 0.6726     | ROUGE_L: 0.6785           | ROUGE_L: 0.6957
|CIDEr: 2.1222       | CIDEr: 2.1928             | CIDEr: 2.2885


Fine tune medium
==============
BLEU: 0.6882
NIST: 8.7973
METEOR: 0.4579
ROUGE_L: 0.7134
CIDEr: 2.4135

Fine tune small:
==============
BLEU: 0.7032
NIST: 8.9437
METEOR: 0.4622
ROUGE_L: 0.7216
CIDEr: 2.4671

Prefix paper params, medium:
==============
BLEU: 0.6888
NIST: 8.6871
METEOR: 0.4589
ROUGE_L: 0.7122
CIDEr: 2.4576

Prefix paper params, small:
==============
BLEU: 0.6835
NIST: 8.6906
METEOR: 0.4492
ROUGE_L: 0.7044
CIDEr: 2.4005

Prefix No warmup w/ BEAM SCORES:
==============
BLEU: 0.6854
NIST: 8.7147
METEOR: 0.4429
ROUGE_L: 0.7036
CIDEr: 2.3444

Prefix No warmup w/ BEAM SCORES with cache fix? SCORES:
==============
BLEU: 0.6854
NIST: 8.7147
METEOR: 0.4429
ROUGE_L: 0.7036
CIDEr: 2.3444

Everything above + pref_len=10:
==============
BLEU: 0.6886
NIST: 8.7939
METEOR: 0.4514
ROUGE_L: 0.7089
CIDEr: 2.3661

pref_len=20:
==============
BLEU: 0.6930
NIST: 8.8408
METEOR: 0.4553
ROUGE_L: 0.7112
CIDEr: 2.4386

pref_len=1:
==============
BLEU: 0.5944
NIST: 5.8914
METEOR: 0.3671
ROUGE_L: 0.6535
CIDEr: 1.5519

GPT Finetuned w/ BEAM SCORES:
==============
BLEU: 0.7032
NIST: 8.9437
METEOR: 0.4622
ROUGE_L: 0.7216
CIDEr: 2.4671


Prefix No warmup / regular generate SCORES:
==============
BLEU: 0.6702
NIST: 8.5515
METEOR: 0.4436
ROUGE_L: 0.6834
CIDEr: 2.2581


Prefix k=800
==============
BLEU: 0.6541
NIST: 8.3552
METEOR: 0.4490
ROUGE_L: 0.6792
CIDEr: 2.1205






