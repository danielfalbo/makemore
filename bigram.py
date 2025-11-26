import torch
import torch.nn.functional as F

with open("names.txt", "r") as f:
    words = f.read().splitlines()

chars = list(sorted(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

# initialize a 27 x 27 tensor
# to count bigram occurrences
#
# N[stoi(c1)][stoi(c2)] will be the number of
# occurrences of c1 followed by c2
#
# N[stoi(c)] will be the tensor of shape (27,) representing
# how many times each possible next character follows c
N = torch.zeros((27, 27), dtype=torch.int32)

# fill N
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1][ix2] += 1

# add 1 to each cell of N so every cell is non-zero
# then cast to float so we can compute divisions
P = (N+1).float()

# compute row-wise sum and divide each cell by its row's sum
#
# >>> t = torch.Tensor([[10,1,4], [9,2,1], [8,8,8]])
# >>> t
# tensor([[10.,  1.,  4.],
#         [ 9.,  2.,  1.],
#         [ 8.,  8.,  8.]])
# >>> t.dtype
# torch.float32
# >>> t.sum(1)
# tensor([15., 12., 24.])
# >>> t.sum(1, keepdims=True)
# tensor([[15.],
#         [12.],
#         [24.]])
# >>> t /= t.sum(1, keepdims=True)
# >>> t
# tensor([[0.6667, 0.0667, 0.2667],
#         [0.7500, 0.1667, 0.0833],
#         [0.3333, 0.3333, 0.3333]])
# >>> t[0], t[0].sum()
# tensor([0.6667, 0.0667, 0.2667]), tensor(1.)
P /= P.sum(1, keepdim=True)

# sample 5 names with manually seeded pseudorandom generator
g = torch.Generator().manual_seed(2147483647)
for _ in range(5):
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1,
                               replacement=True, generator=g).item()
        print(itos[ix], end='')
        if ix == 0: break
    print()

# GOAL:
# maximize likelihood of the data w.r.t. model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihook
# equivalent to minimizing the average negative log likelihood
#
# log(a*b*c) = log(a) + log(b) + log(c)

log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1, ix2 = stoi[ch1], stoi[ch2]
        # prob will be between 0 and 1
        prob = P[ix1, ix2]
        # so logprob will be between -inf and 0
        # https://wolframalpha.com/input?i=log%28x%29
        # since char ix1 is followed by char ix2 in the dataset,
        # a good model would also predict it with high probability,
        # so the higher this probability, the more accurate the model
        logprob = torch.log(prob)
        # we accumulat a sum of all these logprobs a, b, c, ...,
        # which is equivalent to log(a*b*c*...) since
        # log(a*b*c*...) = log(a) + log(b) + log(c) + ...
        log_likelihood += logprob
        n+=1

# since all logprobs will be between -inf and 0, logprob will be a
# negative number. the closer to 0, the better the model.
print(f"{log_likelihood=}")
# we negate it so we can think about it as a loss we want to minimize
# rather than a number we want to maximize
nll = -log_likelihood
print(f"{nll=}")
# we can also compute its average. minimizing this average would be the
# same as minimizing the nll.
print(f"{nll/n=}")

########## NN ##########
print('########## NN ##########')

# create the training set of bigrams.
# one entry (xs[i], ys[i]) will represent an
# instance of xs[i] being followed by ys[i].
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1, ix2 = stoi[ch1], stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

# randomly initialize 27 neurons' weights. each neuron receives 27 inputs.
W = torch.randn((27, 27), requires_grad=True)

# =========== gradient descent ===========
EPOCHS = 100
STEP = 50

# input to the network: one-hot encoding
#
# >>> xs = torch.tensor([4,5,1,2])
# >>> F.one_hot(xs, num_classes=8)   # shape: (4, 8)
# tensor([[0, 0, 0, 0, 1, 0, 0, 0],  # 4
#         [0, 0, 0, 0, 0, 1, 0, 0],  # 5
#         [0, 1, 0, 0, 0, 0, 0, 0],  # 1
#         [0, 0, 1, 0, 0, 0, 0, 0]]) # 2
xenc = F.one_hot(xs,  num_classes=27).float() # shape: (dataset size, 27)

for k in range(EPOCHS):
    # ======= forward pass =======

    # predict log-counts. that is: for each input character in the dataset,
    # compute the log count, in range [-inf, +inf],
    # of each possible following character.
    #
    # matmul shapes: (dataset size, 27) @ (27, 27)
    #
    # result shape (dataset size, 27) where the 'i'th row represents
    # the log count, of each possible following character
    # w.r.t. the 'i'th input character from the dataset.
    #
    # since the input is one-hot encoding,
    # the matmul just returns the relevant row of weights for the given input:
    #
    # >>> W = torch.randn((5, 5), requires_grad=True)
    # >>> W
    # tensor([[-0.2916, -1.1614,  1.9888,  0.0488, -0.1492],
    #         [-1.7746,  0.1434,  0.5204, -0.2836, -0.6686],
    #         [ 0.8811, -1.1309, -1.7453,  0.5903,  1.8205],
    #         [-0.9900, -0.1838,  1.0700,  0.5990,  0.0182],
    #         [-0.6699,  0.2835,  0.1368, -0.5702,  1.5397]],
    #   requires_grad=True)
    # >>> xs = torch.tensor([2])
    # >>> xenc = F.one_hot(xs, num_classes=5).float()
    # >>> xenc
    # tensor([[0., 0., 1., 0., 0.]])
    # >>> xenc @ W
    # tensor([[ 0.8811, -1.1309, -1.7453,  0.5903,  1.8205]],
    #   grad_fn=<MmBackward0>)
    logits = xenc @ W

    # exp the logits gettings counts equivalent to N.
    # logits are log counts in range [-inf, +inf]
    # so counts will live in range [0, +inf]
    # https://wolframalpha.com/input?i=exp%28x%29
    counts = logits.exp()

    # probabilities for next character, obtained by
    # computing row-wise sum and dividing each cell by its row's sum
    #
    # >>> t = torch.Tensor([[10,1,4], [9,2,1], [8,8,8]])
    # >>> t.sum(1, keepdims=True)
    # tensor([[15.],
    #         [12.],
    #         [24.]])
    # >>> t /= t.sum(1, keepdims=True)
    # >>> t
    # tensor([[0.6667, 0.0667, 0.2667],
    #         [0.7500, 0.1667, 0.0833],
    #         [0.3333, 0.3333, 0.3333]])
    # >>> t[0], t[0].sum()
    # tensor([0.6667, 0.0667, 0.2667]), tensor(1.)
    probs = counts / counts.sum(1, keepdims=True)

    # computing loss as mean negative log likelihood
    # by computing log of probabilities of bigrams seen in dataset
    # and adding a normalization term
    #
    # >>> W = torch.randn((5, 5), requires_grad=True)
    # >>> W
    # tensor([[-1.0414, -1.8746,  1.4525,  0.7874,  1.4603],
    #         [-1.8588,  1.9057, -0.9119, -0.7609, -1.2718],
    #         [-0.3349, -0.1505,  0.5478,  0.3559,  0.4125],
    #         [-0.4453,  1.0178,  0.5014, -0.0568, -0.8505],
    #         [ 0.7509,  0.6695,  1.2979, -0.6365,  0.8972]],
    #   requires_grad=True)
    # >>> xs = torch.tensor([4,2,1])
    # >>> xenc = F.one_hot(xs, num_classes=5).float()
    # >>> xenc
    # tensor([[0., 0., 0., 0., 1.],
    #         [0., 0., 1., 0., 0.],
    #         [0., 1., 0., 0., 0.]])
    # >>> logits = xenc @ W
    # >>> logits
    # tensor([[ 0.7509,  0.6695,  1.2979, -0.6365,  0.8972],  # 4
    #         [-0.3349, -0.1505,  0.5478,  0.3559,  0.4125],  # 2
    #         [-1.8588,  1.9057, -0.9119, -0.7609, -1.2718]], # 1
    #   grad_fn=<MmBackward0>)
    # >>> counts = logits.exp()
    # >>> counts
    # tensor([[2.1189, 1.9533, 3.6615, 0.5291, 2.4527],  # 4
    #         [0.7154, 0.8603, 1.7295, 1.4275, 1.5105],  # 2
    #         [0.1559, 6.7238, 0.4018, 0.4672, 0.2803]], # 1
    #   grad_fn=<ExpBackward0>)
    # >>> probs = counts / counts.sum(1, keepdims=True)
    # >>> probs
    # tensor([[0.1977, 0.1823, 0.3417, 0.0494, 0.2289],  # 4
    #         [0.1146, 0.1378, 0.2770, 0.2286, 0.2419],  # 2
    #         [0.0194, 0.8374, 0.0500, 0.0582, 0.0349]], # 1
    #   grad_fn=<DivBackward0>)
    # >>> ys = torch.tensor([1,2,3])
    # >>> arng = torch.arange(xs.shape[0])
    # >>> arng
    # tensor([0, 1, 2])
    # >>> gtprobs = probs[arng, ys]
    # >>> gtprobs
    # tensor([0.1823,  # probability of 1 after 4
    #         0.2770,  # probability of 2 after 2
    #         0.0582], # probability of 3 after 1
    #   grad_fn=<IndexBackward0>)
    # >>> ll = gtprobs.log()
    # >>> ll
    # tensor([-1.7022, -1.2837, -2.8440], grad_fn=<LogBackward0>)
    # >>> mnll = -ll.mean()
    # >>> mnll
    # tensor(1.9433, grad_fn=<NegBackward0>)
    # >>> W**2
    # tensor([[1.0846e+00, 3.5142e+00, 2.1097e+00, 6.2001e-01, 2.1325e+00],
    #         [3.4552e+00, 3.6315e+00, 8.3156e-01, 5.7900e-01, 1.6174e+00],
    #         [1.1215e-01, 2.2653e-02, 3.0011e-01, 1.2668e-01, 1.7012e-01],
    #         [1.9829e-01, 1.0359e+00, 2.5141e-01, 3.2275e-03, 7.2328e-01],
    #         [5.6384e-01, 4.4828e-01, 1.6844e+00, 4.0514e-01, 8.0498e-01]],
    #     grad_fn=<PowBackward0>)
    # >>> lossnorm = 0.01*(W**2).mean()
    # >>> lossnorm
    # tensor(0.0106, grad_fn=<MulBackward0>)
    # >>> loss = mnll + lossnorm
    # >>> loss
    # tensor(1.9538, grad_fn=<AddBackward0>)
    #
    # to compute the noralization term we:
    # 1. square the weights so they are all positive,
    # 2. compute their mean
    # 3. scale it down by a constant,
    #       based on how much smoothing we want to apply
    #
    # with the occurrences matrix method, smoothing was applied via
    # increasing all occurrences by a small amount, like 1, before computing
    # the mean negative log probs of bigrams from the dataset. additionally
    # to smoothing, this also made sure that when evaluated against bigrams
    # never seen in the dataset, the loss was still dicrete rather
    # than +inf, which we would otherwise get when computing negative log of
    # learned likelihood 0 from our occurrences matrix.
    #
    # we want to apply smoothing and take the safety measure with our NN
    # as well. now the weights of the neural network just represent the
    # log count of occurrences, which you can think of like a log
    # occurrences matrix. so we could've added 1 to the counts
    # after exp-ing the model weights and achieve something almost
    # exactly equal to the occurrences matrix normalization. but we can
    # do something more general and more powerful with the neural network.
    # we normalize by adding to the mean nll loss the average of the log counts
    # squared, scaled down based on how much smoothing we want to apply.
    # this is called L2 Regularization or Weight Decay and also acts as a
    # spring or gravity pulling weights toward zero as we follow the gradient.
    loss = (
        -probs[torch.arange(xs.shape[0]), ys].log().mean()
            + 0.01*(W**2).mean()
    )

    # ======= backward pass ========
    W.grad = None # set to zero the gradient
    loss.backward() # backpropagate

    # ======= update ========
    W.data -= STEP * W.grad

print(f"loss: {loss.item():.4f}")

# sample from nn
g = torch.Generator().manual_seed(2147483647)
for _ in range(5):
    ix = 0
    while True:
        # we could use one-hot encoding and do
        # >>> xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        # >>> logits = xenc @ W # predict log-counts
        # >>> counts = logits.exp()
        # >>> p = counts / counts.sum(1, keepdims=True)
        # as usual, but since the one_hot encoding just picks
        # the row of the given index, we can just do
        logits = W[ix]
        counts = logits.exp() # counts, equivalent to N
        p = counts / counts.sum() # probabilities for the next character

        ix = torch.multinomial(p, num_samples=1, replacement=True,
                            generator=g).item()
        print(itos[ix], end='')
        if ix == 0: break
    print()

# # E01: train a trigram language model,
# # i.e. take two characters as an input to predict the 3rd one.
# # Feel free to use either counting or a neural net.
# # Evaluate the loss; Did it improve over a bigram model?
#
# # E02: split up the dataset randomly into
# # 80% train set, 10% dev set, 10% test set.
# # Train the bigram and trigram models only on the training set.
# # Evaluate them on dev and test splits. What can you see?
#
# # E03: use the dev set to tune the strength of smoothing (or regularization)
# # for the trigram model - i.e. try many possibilities and see which one
# # works best based on the dev set loss. What patterns can you see in the
# # train and dev set loss as you tune this strength?
# # Take the best setting of the smoothing and evaluate on the test set
# # once and at the end. How good of a loss do you achieve?
#
# # E05: look up and use F.cross_entropy instead.
# # You should achieve the same result.
# # Can you think of why we'd prefer to use F.cross_entropy instead?
#
# # E06: meta-exercise! Think of a fun/interesting exercise and complete it.
