RuotianLuo
Cross entropy loss (Cider score on validation set without beam search; 25epochs):
fc 0.92
att2in 0.95
att2in2 0.99
topdown 1.01

(self critical training is in https://github.com/ruotianluo/self-critical.pytorch)
Self-critical training. (Self critical after 25epochs; Suggestion: don't start self critical too late):
att2in 1.12
topdown 1.12

Test split (beam size 5):
cross entropy:
topdown: 1.07

self-critical:
topdown:
Bleu_1: 0.779 Bleu_2: 0.615 Bleu_3: 0.467 Bleu_4: 0.347 METEOR: 0.269 ROUGE_L: 0.561 CIDEr: 1.143
att2in2:
Bleu_1: 0.777 Bleu_2: 0.613 Bleu_3: 0.465 Bleu_4: 0.347 METEOR: 0.267 ROUGE_L: 0.560 CIDEr: 1.156

fc 2018/10/25:
Bleu_1: 0.712 Bleu_2: 0.539 Bleu_3: 0.398 Bleu_4: 0.292 METEOR: 0.239 ROUGE_L: 0.517 CIDEr: 0.892 SPICE: 0.169 loss:  2.42994500277
top_down without bu_data 2018/10/29_
Bleu_1: 0.735 Bleu_2: 0.567 Bleu_3: 0.427 Bleu_4: 0.320 METEOR: 0.252 ROUGE_L: 0.536 CIDEr: 0.986 SPICE: 0.184 loss:  2.35969302292
top_down with bu_data 2018/10/30
Bleu_1: 0.764 Bleu_2: 0.602 Bleu_3: 0.460 Bleu_4: 0.349 METEOR: 0.269 ROUGE_L: 0.559 CIDEr: 1.088 SPICE: 0.201 loss:  0
mcb top_down with bu_data 2018/10/30
Bleu_1: 0.757 Bleu_2: 0.598 Bleu_3: 0.457 Bleu_4: 0.344 METEOR: 0.262 ROUGE_L: 0.556 CIDEr: 1.052 SPICE: 0.195 loss:  0
