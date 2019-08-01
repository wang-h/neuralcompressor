# A Pytorch Implementations of Embedding Quantization (Compress Word Embeddings)

This is a stable pytorch re-implementation of https://github.com/zomux/neuralcompressor.
``Compressing Word Embeddings via Deep Compositional Code Learning`` https://arxiv.org/abs/1711.01068
### Requirements:

numpy and pytorch==1.0

### Tutorial of the code

1. Download the project and prepare the data

```
> git clone https://github.com/zomux/neuralcompressor
> cd neuralcompressor
> bash scripts/download_glove_data.sh
```

2. Train the embedding quantization model

```
> python main.py -M 32 -K 16 --train --use_gpu --model data/model
```

```
...
[epoch165] trian_loss=9.12, train_maxp=0.73, valid_loss=8.97, valid_maxp=0.73,  bps=264 
[epoch191] trian_loss=9.02, train_maxp=0.73, valid_loss=8.85, valid_maxp=0.73,  bps=289 
Training Done
```

3. Evaluate the averaged euclidean distance

```
> python main.py --evaluate --use_gpu --model data/model
```

```
(original tensorflow implementation) Mean euclidean distance: 4.889592628145218
(this     pytorch    implementation) Mean euclidean distance: 4.161795844955444
```

4. Export the word codes and the codebook matrix

```
> python main.py --export --use_gpu --prefix data/model
```

It will generate two files:
- data/mymodel.codes
- data/mymodel.codebook.npy

5. Check the codes

```
> head -100 data/model.codes
```

```
...
president	4 13 6 8 5 0 14 8 6 13 3 12 14 4 2 11 8 8 11 3 13 12 1 0 5 9 7 4 0 10 10 2
only	9 8 15 11 2 14 14 15 3 0 3 12 15 4 8 7 11 9 10 11 3 4 5 5 13 9 9 13 10 4 13 2
state	7 3 9 3 1 3 14 4 10 0 9 8 15 4 4 5 6 6 9 4 13 6 10 10 9 9 3 7 9 3 13 2
million	14 4 3 4 12 3 0 15 8 0 2 12 7 6 6 14 2 5 13 8 9 5 6 15 3 9 0 7 9 12 6 4
could	0 2 2 11 2 14 14 2 11 0 2 12 11 4 1 12 8 9 2 11 13 2 5 5 13 9 12 13 6 4 13 2
us	14 12 14 3 2 10 14 15 8 0 15 12 5 4 15 9 2 1 9 8 13 5 14 15 4 9 7 5 6 4 13 2
most	6 0 13 11 9 15 14 14 9 0 15 0 15 5 8 3 15 2 7 11 10 4 11 5 1 9 1 13 6 3 13 13
against	15 15 2 11 10 4 5 2 14 0 10 13 5 4 6 15 8 1 1 7 13 4 13 3 0 9 13 11 0 4 8 2
u.s.	8 12 14 3 15 11 14 9 10 0 15 5 5 4 15 8 2 1 9 15 13 12 1 14 13 9 13 9 15 4 5 15
...
```
### Sample results
```
man	8 11 15 14 11 2 14 14 13 0 3 10 3 4 13 0 13 7 8 13 0 9 5 15 12 6 1 11 6 8 9 2
king	2 13 3 1 0 11 14 6 8 0 4 2 14 11 0 8 8 15 3 3 15 9 9 12 12 9 13 4 1 9 3 2
woman	8 11 8 12 11 4 14 5 3 0 3 10 3 4 14 0 13 7 15 11 1 9 5 12 12 6 8 11 12 8 12 2
queen	2 13 1 8 0 12 14 6 8 0 3 13 14 11 14 0 8 15 15 0 1 9 9 10 2 7 8 4 13 10 12 2
dog	5 11 4 10 0 2 3 13 13 0 11 12 0 14 13 12 7 13 3 13 0 10 8 13 12 2 7 1 11 4 15 2
dogs	14 0 4 14 11 8 3 2 9 0 11 12 0 14 13 12 12 13 3 5 14 15 8 5 8 2 1 1 10 4 12 2
```