# prefix-tuning

## Setup
Install conda environment from `environment.yaml`

For JDK
- Java 1.8
- Perl 5.8.8 or higher with the [XML::Twig](http://search.cpan.org/~mirod/XML-Twig-3.49/Twig.pm) CPAN module


```
./e2e-metrics/measure_scores.py src/target.txt src/model-output.txt 
```

Cached Prefix       | Not Cached Prefix-Tuned   | Fine-tuned
==============      | ==============            | ==============
BLEU: 0.6480        | BLEU: 0.6571              | BLEU: 0.6721
NIST: 8.3238        | NIST: 8.3984              | NIST: 8.5870
METEOR: 0.4431      | METEOR: 0.4439            | METEOR: 0.4546
ROUGE_L: 0.6726     | ROUGE_L: 0.6785           | ROUGE_L: 0.6957
CIDEr: 2.1222       | CIDEr: 2.1928             | CIDEr: 2.2885


SCORES:
==============
BLEU: 0.6721
NIST: 8.5870
METEOR: 0.4546
ROUGE_L: 0.6957
CIDEr: 2.2885




