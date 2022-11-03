# Exposing Outlier Exposure

Here we provide the implementation for the paper *Exposing Outlier Exposure: What Can Be Learned From Few, One, and Zero Outlier Images*.
The implementation is based on PyTorch 1.12.0 and Python 3.10. The code is tested on Linux only.

<img src="data/git_images/toy_example.png?raw=true" height="383" width="738" > 

**Abstract**
Due to the intractability of characterizing everything that looks unlike the normal data,
anomaly detection (AD) is traditionally treated as an unsupervised problem utilizing only
normal samples. However, it has recently been found that unsupervised image AD can be
drastically improved through the utilization of huge corpora of random images to represent
anomalousness; a technique which is known as Outlier Exposure. In this paper we show
that specialized AD learning methods are unnecessary for state-of-the-art performance, and
furthermore one can achieve strong performance with just a small collection of Outlier
Exposure data, contradicting common assumptions in the field of AD. We find that standard
classifiers and semi-supervised one-class methods trained to discern between normal samples
and relatively few random natural images are able to outperform the current state of the
art on an established AD benchmark with ImageNet. Further experiments reveal that even
one well-chosen outlier sample is sufficient to achieve decent performance on this benchmark
(79.3% AUC). We investigate this phenomenon and find that one-class methods are more
robust to the choice of training outliers, indicating that there are scenarios where these
are still more useful than standard classifiers. Additionally, we include experiments that
delineate the scenarios where our results hold. Lastly, no training samples are necessary when
one uses the representations learned by CLIP, a recent foundation model, which achieves
state-of-the-art AD results on CIFAR-10 and ImageNet in a zero-shot setting.


## Work In Progress
The code is currently undergoing a major clean-up and will soon be available. Stay tuned!
