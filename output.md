4
2
0
2

g
u
A
0
3

]
S
A
.
s
s
e
e
[

1
v
2
4
1
7
1
.
8
0
4
2
:
v
i
X
r
a

RECURSIVE ATTENTIVE POOLING FOR EXTRACTING SPEAKER EMBEDDINGS FROM
MULTI-SPEAKER RECORDINGS

Shota Horiguchi

Atsushi Ando

Takafumi Moriya

Takanori Ashihara

Hiroshi Sato Naohiro Tawara Marc Delcroix

NTT Corporation, Japan

ABSTRACT

This paper proposes a method for extracting speaker embedding
for each speaker from a variable-length recording containing multi-
ple speakers. Speaker embeddings are crucial not only for speaker
recognition but also for various multi-speaker speech applications
such as speaker diarization and target-speaker speech processing.
Despite the challenges of obtaining a single speaker’s speech without
pre-registration in multi-speaker scenarios, most studies on speaker
embedding extraction focus on extracting embeddings only from
single-speaker recordings. Some methods have been proposed for
extracting speaker embeddings directly from multi-speaker record-
ings, but they typically require preparing a model for each possible
number of speakers or involve complicated training procedures. The
proposed method computes the embeddings of multiple speakers
by focusing on different parts of the frame-wise embeddings ex-
tracted from the input multi-speaker audio. This is achieved by
recursively computing attention weights for pooling the frame-wise
embeddings. Additionally, we propose using the calculated atten-
tion weights to estimate the number of speakers in the recording,
which allows the same model to be applied to various numbers of
speakers. Experimental evaluations demonstrate the effectiveness of
the proposed method in speaker verification and diarization tasks.

Index Terms— speaker embedding, speaker recognition, speaker

verification, speaker diarization

1. INTRODUCTION

A speaker-discriminative embedding (or speaker embedding) is a
feature extracted from an audio segment that represents the speaker’s
identity or voice characteristics, while excluding the content con-
tained in the segment and other paralinguistic information. The use
of speaker embeddings has become the de facto standard in speaker
recognition tasks, i.e., speaker identification and speaker verification
[1, 2, 3, 4, 5, 6]. These tasks usually assume only a single speaker
per recording, so most conventional methods cannot extract embed-
dings from multi-speaker recordings. At the same time, in recent
years, the use of speaker embeddings has expanded beyond speaker
recognition to include speech applications for multi-speaker audio.
One example of multi-speaker applications in which speaker
embeddings play an important role is speaker diarization. For ex-
ample, in a cascaded speaker diarization system, diarization is per-
formed by dividing detected speech intervals into small segments,
extracting a speaker embedding from each segment, and then clus-
tering them [7]. Since speaker embedding extraction and cluster-
ing are performed by assuming that each segment corresponds to a
single speaker, the functional limitation is that overlapping speech
cannot be handled. Overlaps can be treated to some extent by post-
processing [8, 9, 10], but embeddings extracted from overlaps are

distributed in the middle of each speaker’s embeddings [11], which
can be an obstacle to clustering. There are end-to-end speaker di-
arization methods that naturally deal with overlaps such as EEND
[12, 13] and TS-VAD [14]. However, they still have challenges such
as the need for a large amount of high-quality simulation data for
training [15, 16, 17] and the eventual necessity of extracting speaker
embeddings from multi-speaker speech segments [14, 18, 19, 20].

Another example of a multi-speaker application using speaker
embeddings is to obtain speech processing results for a speaker of
interest, i.e., target speaker. Many models for target-speaker speech
processing have been proposed such as voice activity detection [21,
22], speaker extraction [23, 24, 25], and speech recognition [26, 27,
28]. Target-speaker speech processing usually assumes the avail-
ability of a single-speaker utterance of the target speaker to capture
reliable speaker embeddings. However, extracting reliable speaker
embeddings from multi-speaker audio would allow us to expand the
applications of target-speaker based tasks.

Despite the benefits of extracting speaker embeddings from
multi-speaker audio, few studies have tackled this problem. One
possible approach is to apply speech separation in advance to obtain
single-speaker recordings [29]. However, this is known to produce
artifacts that may negatively affect the later processing stage [30];
that is, separation may damage the original speaker characteristics.
Another approach is to construct a model that directly extracts the
embeddings of each speaker from multi-speaker audio. However,
conventional methods require preparing a model for each number of
speakers [31, 32], which is quite costly, and teacher-student learning
is necessary to achieve reasonable performance [32].

In this paper, we propose a method for extracting a speaker
embedding for each speaker from audio that may include multiple
speakers. Typical speaker embedding extractors have a cascaded
structure consisting of i) an encoder that converts variable-length
input into frame-wise embeddings and ii) a pooling module that ag-
gregates the frame-wise embeddings into a single speaker embed-
ding. The proposed method enables the extraction of embeddings
for each speaker, even from fully overlapped speech, by focusing on
different parts of the frame-wise embeddings. More specifically, the
proposed method recursively calculates the attention weights used
for pooling a variable-length sequence of frame-wise embeddings.
It is notable that this can be achieved only by adding a single linear
layer to the original speaker embedding extractors. We also propose
a method that uses the calculated attention weights in estimating the
number of speakers, which enables the extraction of speaker em-
beddings corresponding to each speaker using the same model even
when the number of speakers in the input is unknown. We demon-
strate the effectiveness of the proposed method in speaker verifica-
tion and diarization tasks, and also provide detailed analyses of the
experimental results to clarify the behavior of the proposed method.

2. RELATED WORK

first expanded with their mean and standard deviation vectors µ and
σ to consider the global context as

Various architectures have been proposed for speaker embedding ex-
traction. As the encoder, previous studies have investigated architec-
tures such as time delay neural network (TDNN) [33, 5], long short-
term memory [34], and convolutional neural network [35, 6]. For
pooling methods, various alternatives to the simple temporal average
pooling [1] have been proposed, such as statistics pooling [1], atten-
tive statistics pooling [36], multi-head attention pooling [37], vector-
based attentive pooling [38], and channel- and context-independent
statistics pooling [5]. Eventually, regardless of the encoder or pool-
ing method, the output is a single embedding, which means that a
single speaker’s recording is assumed as the input.

There are few studies that have investigated speaker embedding
extraction from multi-speaker recordings. MIRNet [31] uses the at-
tention mechanism to extract each embedding of two speakers. It in-
cludes a processor that creates the embedding for the second speaker
by swapping the channels of the first and second halves of the em-
bedding computed for the first speaker, which fixes the number of
speakers to two. In addition, the attention weights are frame-wise
ones, so in principle, it requires single-speaker segments and never
works perfectly for fully overlapped speech. In EEND-vector clus-
tering [18, 19], multiple speakers’ embeddings are extracted on the
basis of estimated overlap-aware diarization results. Here, speaker
embeddings are also calculated as the weighted average of frame-
wise embeddings, and thus, it cannot deal with fully overlapped
speech. Very recent work proposed a method for extracting the
embedding of each speaker from a fully overlapped speech by de-
veloping a multi-headed model [32]. The proposal uses teacher-
student learning, i.e., a model that outputs two embeddings from
two-speaker audio is trained to mimic each output of a well-trained
single-speaker model given single-speaker sources. The authors re-
ported that training the model from scratch did not work well, so us-
ing this method requires two training runs, which is time-consuming.
Also, since the model can only process two-speaker speech, it is nec-
essary first to determine whether the input audio contains one or two
speakers and then use the appropriate model. In contrast, our ap-
proach can use the same model regardless of the number of speakers
and can be trained from scratch.

3. METHOD

Figure 1 shows a schematic diagram of a common single-speaker
speaker embedding extractor (Fig. 1(a)) and how we extend it to
enable the decoding of each of multiple speakers’ embeddings
(Fig. 1(b)). Since our proposal is a method of pooling the output
from the encoder, any encoder architecture that extracts a sequence
of frame-wise embeddings from input recording can be adopted.

3.1. Review of channel- and context-dependent attentive statis-
tics pooling

We first review the channel- and context-dependent attentive statis-
tics pooling [5] on which our proposed method is based. This tech-
nique is widely used in the modern speaker embedding extractors
[39, 40, 41]. It performs weighted averaging of frame-wise embed-
dings using attention over the time axis for each dimension to main-
tain information meaningful in the computation of speaker embed-
ding. In this respect, it is similar to mask-based speech enhancement,
which extracts only those time-frequency bins that contain speech.

Given a T -length sequence of D-dimensional frame-wise em-
beddings H := [h1, . . . , hT ] ∈ RD×T from the encoder, each is

et = ht ⊕ µ ⊕ σ ∈ R3D,

µ =

σ =

1
T

(cid:118)
(cid:117)
(cid:117)
(cid:116)

T
(cid:88)

τ =1

hτ ∈ RD,

1
T

T
(cid:88)

τ =1

hτ ⊙ hτ − µ ⊙ µ ∈ RD.

(1)

(2)

(3)

where ⊕ denotes vector concatenation and ⊙ denotes the Hadamard
product. The attention weights A := [a1, . . . , aT ] to aggregate the
expanded embeddings et is calculated by

A = Softmax ([˜a1, . . . , ˜aT ]) ∈ (0, 1)D×T ,
˜at = W2f (W1et + b1) + b2 ∈ RD,

(5)
where Softmax (·) denotes the row-wise softmax function, W1 ∈
RD′×3D and b1 ∈ RD′
denotes the weight and bias of the first linear
layer, W2 ∈ RD×D′
and b2 ∈ RD are those of the second layer, and
f (·) is the rectified linear unit, respectively. Speaker embedding v
is then computed as

(4)

v = Wo ( ˜µ ⊕ ˜σ) + bo ∈ RE,
(6)
where Wo ∈ RE×2D and bo ∈ RE are the parameters of the last
linear layer to obtain E-dimensional speaker embeddings. ˜µ and ˜σ
are the weighted mean and standard deviation vectors, respectively,
calculated using the attention weights as:

T
(cid:88)

˜µ =

aτ ⊙ hτ ∈ RD,

τ =1
(cid:118)
(cid:117)
(cid:117)
(cid:116)

T
(cid:88)

τ =1

˜σ =

aτ ⊙ hτ ⊙ hτ − ˜µ ⊙ ˜µ ∈ RD.

(7)

(8)

3.2. Recursive embedding extraction and speaker counting

Since conventional pooling methods aggregate a variable-length se-
quence of frame-wise embeddings into a single embedding, single-
In this paper, we extend the
speaker audio is assumed as input.
channel- and context-dependent attentive statistics pooling to extract
as many speaker embeddings as there are speakers in the input au-
dio. If frame-wise embeddings that can assume speaker sparsity are
achieved, we can use the mask-based speech separation approach
to extract an embedding for each speaker. In particular, by calcu-
lating masks recursively as in recurrent selective attention network
(RSAN) [42], it is possible for a single model to extract embed-
dings for an arbitrary number of speakers. RSAN uses the princi-
ple that the sum of masks for each source is constant, but it does
not hold in the case of attention weights in the speaker embedding
extractor. Therefore, we instead introduce the coverage mechanism
[43, 44], which was originally proposed for neural machine trans-
lation to prevent over- or under-translation, to monitor which parts
of the frame-wise embeddings have already been used for decoding
speaker embeddings. Note that we provide a general formulation of
the proposed method for an arbitrary number of speakers in this sec-
tion, while we assume in the experiments that the input is either one-
or two-speaker audio. This is because we can easily remove regions
where there are no speakers using voice activity detection, and in
practice, there are often not more than two speakers speaking at the
same time [45].

In the proposed method, attention weights for the n-th speaker
are calculated from not only the frame-wise embeddings but also

Log mel
spectrogram

Log mel
spectrogram

Encoder

H

Encoder

H

Eqs. (4)–(5)

Attention
calculation

A

Attentive stats pooling

Eqs. (7)–(8)

C (0)

Coverage
matrix

Eq. (6)

Linear

C (1)

Eqs. (9)–(10)

C (2)

Eqs. (9)–(10)

Attention
calculation

p1

Attention
calculation

p2

Attentive stats pooling

A(1)

Attentive stats pooling

A(2)

Eqs. (12)–(13)

Eq. (11)

Eqs. (12)–(13)

Eq. (11)

Linear

Linear

Speaker
embedding

v

Speaker embedding
of the first speaker

v(1)

Speaker embedding
of the second speaker v(2)

(a) Basic speaker embedding extractor.

(b) Proposed speaker embedding extractor for multi-speaker audio using recursive pooling.

Fig. 1: Schematic diagram of the conventional and proposed methods.

the cumulative sum of the previously calculated attention weights.
Instead of (4) and (5), we calculate the attention weights for each
(cid:105)
speaker A(n) :=
that appear in the input audio in
a recursive manner as follows:
˜a(n)
A(n) = Softmax
1

1 , . . . , a(n)
a(n)

∈ (0, 1)D×T ,

. . . , ˜a(n)

(9)

(cid:16)

(cid:17)

(cid:104)

T

T

˜a(n)

t = W2f

(cid:16)

W1et + b1 + Wcc(n)
(cid:105)

t

(cid:104)

(cid:17)

+ b2 ∈ RD.

(10)

T

r=0 a(r)

t = (cid:80)n−1

where c(n)

1 , . . . , c(n)
c(n)

t ∈ RD
Here, C (n) =
≥0
is the coverage matrix defined as the cumulative sum of the previous
attentions, and Wc ∈ RD′×D is the only additional learnable param-
eter needed in the proposed method. Note that a(0)
is defined as a
D-dimensional zero vector. Here, we aim to extract multiple embed-
dings for different speakers from the same input audio by applying
attention to regions that have not been focused on by the (n − 1)-th
speaker when computing the embedding for the n-th speaker. This
requires speaker sparsity in the frame-wise embeddings from the en-
coder, similar to the mask-based speech separation. The proposed
method achieves this by end-to-end optimization of the entire model.
Finally, the embedding of the n-th speakers is extracted as the

t

same manner in (6)–(8) as follows:
˜µ(n) ⊕ ˜σ(n)(cid:17)
(cid:16)

v(n) = Wo

+ bo ∈ RE,

T
(cid:88)

˜µ(n) =

a(n)
τ ⊙ hτ ∈ RD,

(11)

(12)

τ =1
(cid:118)
(cid:117)
(cid:117)
(cid:116)

T
(cid:88)

τ =1

˜σ(n) =

a(n)

τ ⊙ hτ ⊙ hτ − ˜µ(n) ⊙ ˜µ(n) ∈ RD.

(13)

To decide when to stop the recursive attention generation pro-
cess, we monitor whether any parts of the frame-wise embeddings
still attract attention. Since attention is normalized along the time
axis using the softmax function (unlike mask-based source separa-
tion), we use the values before normalization ˜a(n)
for the monitor-
ing. Speaker existence probability pn, which indicates whether the

t

n-th speaker really exists in the input audio, is estimated as

pn =

1 + exp

1
(cid:80)T
t=1 w · ˜a(n)

t + b

(cid:17) ,

(cid:16)

− 1
T

(14)

where w ∈ RD and b ∈ R are the learnable parameters. During
inference, thresholding pn acts as a stopping condition, where the
threshold value is set to 0.5 in this paper.

3.3. Training objective

We assume that the model outputs N speaker embeddings V :=
{v1, . . . , vN }, each of which corresponds to one of the individu-
als in the training speaker set Y := {y1, . . . , yS}. The model also
output existence probabilities p1, . . . , pN +1 for the N + 1 speakers.
The training objective to be minimized is determined as follows:

L = Lspk + αLcnt,

(15)

where α is a weight factor that is fixed to 0.1 in this paper.

The first term, Lspk, in (15) aims to optimize the similarity met-
ric of speaker embeddings. We train the network as a multi-class
classifier using the permutation-free loss

Lspk =

min
(cid:17)

(cid:16)

1 ,...,yϕ
yϕ

N

∈Φ(Y ′)

1
N

N
(cid:88)

n=1

(cid:16)

n, v(n)(cid:17)
yϕ

,

ℓ

(16)

where Y ′ ⊆ Y is the set of speakers in the input audio and Φ (Y ′) is
all permutation of Y ′. ℓ (y, v) is the additive angular margin (AAM)
softmax loss [46] defined as

ℓ (y, v) = − log

es cos(θy +m)

es cos(θy +m) + (cid:80)N

n=1,n̸=y es cos θn

θy = arccos

wy · v
∥wy∥ ∥v∥

,

,

(17)

(18)

where m > 0 denotes the margin, s > 0 is the scaling factor, and
wy ∈ RE is the learnable proxy of identity y.

The second term, Lcnt, in (15) aims to optimize the speaker

counting accuracy, which is defined from cross-entropy as
(cid:33)

Lcnt = −

log (1 − pn) + log pN +1

.

(19)

1
N + 1

(cid:32) N
(cid:88)

n=1

Here, pn for 1 ≤ n ≤ N is optimized to be one, and pN +1 is
optimized to be zero. As mentioned in Sec. 3.2, this paper assumes
that the input contains one or two speakers, so we use the simplified
alternative below to (19) only to check if the second speaker exists:

(cid:40)

Lcnt =

− log p2
− log (1 − p2)

(N = 1),
(N = 2).

3.4. Inference-time length mismatch correction

in a range of [−5, 5] dB, following the separation benchmark [52].
Each model was trained for 80 epochs using the Adam optimizer
[53] with a cyclical learning rate. Each cycle consisted of 20 epochs,
with the first 1,000 iterations taken as warm-up, followed by cosine
annealing for the remaining iterations. The peak learning rate at the
first cycle was set to 0.001 and 0.0005 for the baselines and proposed
method, respectively, and was decayed by a factor of 0.75 with each
cycle. The margin and scaling factor of AAM-softmax were set to
0.2 and 30, respectively.

(20)

4.2. Evaluation

4.2.1. Speaker verification

It is common to align the length of each sample in a single minibatch
during training for efficient batch processing. However, models are
usually required to process variable-length inputs during inference.
The conventional method does not suffer from this mismatch be-
cause ˜at in (5) is calculated independently for each frame. On the
other hand, the proposed method includes c(n)
in (10), which de-
pends on the sequence length for the second and subsequent speakers
because a(n)
is computed with the softmax over the time dimension
as seen in (9). We use the following equation instead of (10) only
during inference so that any mismatch in audio length between train-
ing and inference does not affect the results:

t

t

(cid:18)

˜a(n)

t = W2f

W1et + b1 +

(cid:19)

Wcc(n)
t

Tinfer
Ttrain

+ b2,

(21)

where Ttrain is the sequence length of frame-wise embeddings fixed
during training and Tinfer is their length extracted from inference
speech. This improves inference performance for the second and
subsequent speakers by making the expected value of attention
weights per frame the same during training and inference, which
eliminates the effect of sequence-length mismatch.

4. EXPERIMENTAL SETTINGS

4.1. Training

We used the concatenation of the VoxCeleb1 dev set and VoxCeleb2
dev set [47] for training; the result consisted of 1,240,651 utterances
from 7,205 speakers. The VoxCeleb2 test set was used for validation.
As the network architecture, we tested three different encoders
to show the generality of the proposed method: x-vector encoder
consisting of a five-stacked TDNN (D = 1500) [33], ResNet34
(D = 2560) [48, 49, 32], and ECAPA-TDNN with 1024 channels
(D = 1536) [5]. Channel- and context-aware attentive statistics
pooling and the proposed method were used as the pooling methods
regardless of encoder type. The output dimension of the final linear
layer was set to E = 192. The input to each network was a sequence
of mean-normalized 80-dimensional log mel filterbanks extracted
with a window length of 25 ms and shift of 10 ms. This yielded
100 frame-wise embeddings per second for x-vector and ECAPA-
TDNN, and 12.5 for ResNet34.

During training, each utterance in a mini-batch was 3 s in du-
ration and augmented using noise [50] with a probability of 0.5. It
was further reverberated using simulated room impulse responses
[51] with a probability of 0.5. The mini-batch size was 256 for the
single-speaker baselines and 384 (256 single-speaker audio and 128
two-speaker audio) for the proposed method. Each two-speaker au-
dio was a fully overlapped mixture generated on the fly during train-
ing. Signal-to-interference ratio (SIR) used to generate mixtures lay

Following the previous study [32], speaker verification performance
was evaluated under the three scenarios detailed below:
s vs. s: This is the standard single-speaker audio vs. single-speaker
audio scenario. The evaluation used the VoxCeleb1 test set, a.k.a.
VoxCeleb1-O, consisting of 37,611 trials from 40 speakers.

s vs. m: This is the single-speaker audio vs. two-speaker audio sce-
nario. Each positive sample in s vs. m is a pair in which the
speaker of the single-speaker audio is one of the speakers in the
mixture, while a negative sample is a pair in which the speaker of
the single-speaker audio is not included in the mixture. We used
the same evaluation set used in the conventional study [32], i.e.,
37,611 trials extended from VoxCeleb1-O by mixing interference
speech with random SIR.

m vs. m: This is the two-speaker audio vs. two-speaker audio sce-
nario. A positive (negative) sample in m vs. m is a pair in which
one speaker in one of the mixture is included (not included) in the
other mixture. Note that pairs of mixtures in which both speakers
were the same were not included. In this scenario, two evaluation
protocols were used: any spk and per spk. In the any spk protocol,
only the largest similarity value among all embeddings combina-
tions extracted from each audio was used for evaluation, whereas
in the per spk scenario, the other pair was also used as a negative
pair. Also note that the per spk protocol assumes that two embed-
dings are extracted from any mixture, so it can be evaluated only
if the model is capable of outputting two embeddings and the ora-
cle number of speakers is given. Here too, we used the evaluation
set of 37,611 trials used in the conventional study [32].

As the evaluation metrics, equal error rate (EER) and minimum
detection cost function (minDCF) were used, where the prior proba-
bility for minDCF was set to 0.01 for s vs. s and 0.05 for s vs. m and
m vs. m as in the conventional study [32]. We used cosine similarity
for scoring each trial.

4.2.2. Speaker diarization

Speaker diarization performance was evaluated using the LibriCSS
dataset [45] and AMI Mix-Headset corpus [54]. We used auto-
tuning spectral clustering (SC) [55] and its extension to deal with
overlapping speech (SC-OL) [56] as the baseline methods. Speech
and overlapped segments were given by the oracle voice activity de-
tector and overlap detector. Speaker embeddings for clustering were
extracted with 1.5 s window with 0.75 s shift using the ECAPA-
TDNN-based model. For the proposed method, we extracted one
speaker embedding from single-speaker segments and two speaker
embeddings from overlapped segments, and applied SC with cannot-
link constraints such that a pair of embeddings from the same seg-
ments were never assigned to the same cluster. We used diarization
error rate (DER) without collar tolerance as the evaluation metric.

Table 1: Single- and multi-speaker verification performance evaluated using EERs (%) and minDCF.

#Output

Speaker
counting

s vs. s

s vs. m

m vs. m (any spk) m vs. m (per spk)

EER minDCF

EER minDCF

EER minDCF

EER minDCF

ID

Encoder

Results from the reference papers
R1
R2
R3
R3’

x-vector [57]
ECAPA-TDNN [5]
ResNet34 (teacher) [32]
+ ResNet34 (student) [32]

x-vector
+ Proposed method
+ Oracle # of speakers

Results based on our implementation
S1
S2
S2’
S3
S4
S4’
S5
S6
S6’

ResNet34
+ Proposed method
+ Oracle # of speakers

ECAPA-TDNN
+ Proposed method
+ Oracle # of speakers

1
1
1
1 or 2

1
1 or 2
1 or 2
1
1 or 2
1 or 2
1
1 or 2
1 or 2

-
-
-
Oracle

1.81
0.87
1.06

0.13
0.11
0.16
(Same to R3)

-
Estimated
Oracle
-
Estimated
Oracle
-
Estimated
Oracle

1.65
1.83
1.82
1.09
1.20
1.19
0.88
1.20
1.17

0.16
0.17
0.17
0.11
0.12
0.12
0.09
0.12
0.12

-
-
18.2
9.1

20.72
9.16
8.16
20.85
7.83
7.47
24.51
7.71
6.35

-
-
0.57
0.46

0.61
0.37
0.37
0.57
0.33
0.33
0.59
0.29
0.28

-
-
47.6
15.3

31.48
16.85
15.11
32.01
15.65
15.04
35.26
14.13
11.97

-
-
1.00
0.74

0.87
0.61
0.62
0.83
0.60
0.61
0.84
0.50
0.50

-
-
-
14.1

-
-
10.63
-
-
12.15
-
-
8.34

-
-
-
0.74

-
-
0.54
-
-
0.59
-
-
0.41

5. RESULTS

5.1. Speaker verification

5.1.1. Main results

The experimental results from speaker verification are shown in Ta-
ble 1. The first four rows (R1–R3’) show the values of the conven-
tional methods reported in the cited papers [57, 5, 32]. If we compare
the s vs. s results of R1–R3 with our reimplemented systems (S1,
S3, S5), we can safely say that they have been mostly reproduced,
while some mismatch in training strategies, such as data augmenta-
tion and the number of training iterations, yielded slight differences.
However, since these systems extract only one speaker embedding
from input audio, the verification performance is quite poor in the s
vs. m and m vs. m scenarios as they involve multi-speaker audio. The
proposed method (S2, S4, S6) significantly improved the results for
s vs. m and m vs. m with small degradation in s vs. s. Given the or-
acle number of speakers (S2’, S4’, S6’), the speaker verification
performance is further improved. Even though the proposed method
uses the same model regardless of the number of speakers, it sig-
nificantly outperformed the conventional method that used different
models for each number of speakers [32].

Comparing the encoder types, even though ResNet34 (S4’) and
ECAPA-TDNN (S6’) had similar verification performance on s vs.
s, ResNet34 performed worse in the multi-speaker scenarios, espe-
cially when m vs. m, and was less accurate than as x-vector (S2’).
One possible reason for this is that the encoder output of the x-
vector or ECAPA-TDNN retains the sequence length of the input
log mel spectrogram, whereas that of ResNet34 has lower tempo-
ral resolution due to compressing the sequence length by one-eighth
through the convolutional layers, and thus speaker sparsity cannot
be assumed for each dimension of the embedding sequence output
from the encoder. The following sections provide detailed analyses
of the models based on ECAPA-TDNN (S5,S6,S6’).

5.1.2. Variable length evaluation

Table 2: EERs for various durations without inference-time length
mismatch correction (10) and with correction (21) using S6’.

s vs. s

s vs. m

m vs. m (any spk) m vs. m (per spk)

Duration

1 s
2 s
3 s (matched)
5 s
10 s
Original

16.41
5.18
2.67
1.39
1.20
1.17

(10)

(21)

(10)

28.02
14.70
9.77
6.95
7.07
7.40

27.44
14.61
9.77
6.84
6.33
6.35

36.52
23.43
17.31
12.76
13.34
13.84

(21)

35.64
23.14
17.31
12.61
11.97
11.97

(10)

23.84
15.96
11.92
9.09
9.83
10.30

(21)

23.54
15.83
11.92
8.90
8.35
8.34

audio length during training was 3 s. As in the literature [58], short-
duration utterances significantly degraded the EERs of the s vs. s
scenario, while no disadvantages due to long utterance lengths were
seen. However, in the multi-speaker scenarios, i.e., s vs. m and m vs.
m, degradation of EERs was observed even with increasing utterance
lengths when (10) was used for attention weight calculation. The re-
sults clearly show that introducing inference-time length mismatch
correction in (21) improved the EERs in all cases where there is a
mismatch in utterance length. In addition, the disadvantage of using
longer utterances than used in training almost disappeared.

5.1.3. Detailed analyses

This section provides detailed analyses of the proposed method in-
cluding speaker counting, the effect of SIR, and attention weights.
For the analyses, we used the s vs. s trials and the positive pairs
from the s vs. m trials. Each positive example in the s vs. m trials
consisted of a pair of single-speaker and two-speaker audio, where
one speaker of the two-speaker audio is identical to the speaker of
the single-speaker audio. For the purpose of analyses, this speaker
is considered as the target speaker and the other as the interference
speaker to calculate SIR rSIR. For simplicity, we denote the positive
case of s vs. s by rSIR = ∞ because it contains only the sounds of
the target speaker, and the negative case by rSIR = −∞ because it
contains only the sounds of the inference speaker.

Table 2 shows the EERs when varying the lengths of input utter-
ances using S6’. For the t-second audio evaluation, the score was
calculated using only the first t seconds of each trial pair. Again, the

Table 3 shows the accuracy of speaker counting for each SIR
range. Single-speaker recordings were rarely estimated as having
two speakers, but the speaker counting accuracies of two-speaker

Table 3: SIR-wise speaker counting accuracy (%) of S6. The results
corresponding to the correct prediction are bolded.

s vs. s

s vs. m

|rSIR| (dB)

∞ [0, 5)

[5, 10)

[10, 15) ≥ 15

Predicted as 1 speaker
Predicted as 2 speakers

100.0
0.0

0.1
99.9

2.6
97.4

33.0
67.0

90.6
9.4

All

10.6
89.4

Fig. 3: Visualization of attentions calculated for two speakers in a
single mixture (bottom left) compared with the oracle binary mask
(top left). The patterns commonly seen in the two are also cropped
and shown enlarged (right).

Table 4: DERs (%) on LibriCSS and AMI Mix-Headset.

LibriCSS

AMI

Method

0L

0S OV10 OV20 OV30 OV40 Mix-Headset

SC [55]
1.36 0.32 7.45
1.28 0.18 6.85
SC-OL [56]
SC + Proposed 1.29 0.75 7.10

14.28 19.78 23.36
11.84 16.63 18.06
8.31 11.87 12.71

18.95
17.04
16.93

exhibited the same patterns as the oracle binary masks (Fig. 3 right).
This indicates that the model was able to internally acquire functions
similar to source separation.

5.2. Speaker diarization

Table 4 shows the DERs on the LibriCSS dataset and AMI Mix-
Headset corpus. Focusing on the results on LibriCSS, the proposed
method showed slightly degraded DERs when the overlap ratio was
small but significantly improved DERs when the overlap ratio was
high relative to SC-OL. With the improvements in the case of high
overlap ratios, it can be said that the proposed speaker embedding
extractor enables us to extract two speaker embeddings from one in-
terval, and thus, we can perform overlap-aware diarization by simply
clustering them. The proposed method also outperformed the base-
lines on the AMI Mix-Headset corpus, further supporting its effec-
tiveness. The slight degradations in DER with small overlap ratios
are considered to be due to a marginal degradation in single-speaker
embeddings (cf. s vs. s in Table 1), indicating room for improvement
in the proposed method.

6. CONCLUSION

In this paper, we proposed a method for extracting speaker embed-
dings from multi-speaker audio. The proposed method enabled the
extraction of speaker embedding for each speaker, even from fully
overlapped speech, by recursively calculating the attention weights
for pooling. We also proposed a method for simultaneously estimat-
ing the number of speakers based on the calculated attention weights.
Experimental results showed that the proposed method offers im-
provements in both speaker verification and diarization performance.
Future work will include the combination of the proposed method
with end-to-end diarization framework.

Fig. 2: SIR-wise cosine similarity scores.

recordings depend on the absolute SIR. When the absolute SIR was
below 10 dB, two-speaker estimates were accurate. On the other
hand, in the extreme case of SIR (>15 dB), a single-speaker was
estimated with a probability of more than 90 %.

Figure 2 shows the relationship between SIR and the (maxi-
mum) cosine similarity between embeddings extracted from a pair
of recordings. Note that the oracle number of speakers was given in
this case. When only one embedding was extracted from audio as
in the conventional methods (red boxes in Fig. 2), the interference
speaker became more dominant as SIR became smaller and the co-
sine similarity decreased. Using the proposed method (blue boxes in
Fig. 2), it was possible to extract embedding for each speaker, which
kept high cosine similarity (> 0.4) even when the SIR decreased up
to −10 dB.

Finally, we visualized the ideal binary mask and computed at-
tention weights of the example mixtures from the same and differ-
ent gender pairs in Fig. 3.1 First, it is observed that most of the
time-dimension bins had at most one speaker with high attention
weight since one of the colors is dominant in the visualized attention
weights (Fig. 3 left bottom). This indicates the sparseness of speak-
ers in the embedding sequence, similar to the sparseness of signals in
the time-frequency domain. When focusing on each specific frame,
in some frames, it can be seen that different speakers attracted atten-
tion depending on the dimension. For example, focusing on 1.8 s to
1.9 s in Fig. 3 right, the top and middle figures show the attention in
green, whereas the bottom figure shows the attention in pink. This
indicates that, in contrast to the conventional method [31], the model
can extract information from the same frame for different speakers
depending on the dimension, similar to the oracle binary mask. Next,
when focusing on each specific dimension, the extracted attention

1Note that the order of dimensions of the attention weights was rearranged
for visualization purposes, which is valid because the weights of the speaker
extractor are permutation-free.

−0.20.00.20.40.60.81.0CosinesimilarityrSIR=−∞rSIR<−15−15≤rSIR<−10−10≤rSIR<−5−5≤rSIR<00≤rSIR<55≤rSIR<1010≤rSIR<15rSIR≥15rSIR=∞rSIR(dB)ConventionalProposed800040000Frequency[Hz]Oraclebinarymask012Time[s]153611527683840DimensionindexAttentionweights00.51.01.52.0Time[s]7. REFERENCES

[1] David Snyder, Daniel Garcia-Romero, Daniel Povey, and San-
jeev Khudanpur, “Deep neural network embeddings for text-
independent speaker verification,” in Proc. Interspeech, 2017,
pp. 999–1003.

[2] Ruifang Ji, Xinyuan Cai, and Bo Xu, “An end-to-end text-
independent speaker identification system on short utterances,”
in Proc. Interspeech, 2018, pp. 3628–3632.

[3] Nguyen Nang An, Nguyen Quang Thanh, and Yanbing Liu,
“Deep CNNs with self-attention for speaker identification,”
IEEE Access, vol. 7, pp. 85327–85337, 2019.

[4] Joon Son Chung, Jaesung Huh, Seongkyu Mun, Minjae Lee,
Hee Soo Heo, Soyeon Choe, Chiheon Ham, Sunghwan Jung,
Bong-Jin Lee, and Icksang Han, “In defence of metric learning
for speaker recognition,” in Proc. Interspeech, 2020, pp. 2977–
2981.

[5] Brecht Desplanques, Jenthe Thienpondt, and Kris Demuynck,
“ECAPA-TDNN: Emphasized channel attention, propagation
and aggregation in TDNN based speaker verification,” in Proc.
Interspeech, 2020, pp. 3830–3834.

[6] Tianyan Zhou, Yong Zhao, and Jian Wu,

Res2Net structures for speaker verification,”
2021, pp. 301–307.

“ResNeXt and
in Proc. SLT,

[7] Tae Jin Park, Naoyuki Kanda, Dimitrios Dimitriadis, Kyu J
Han, Shinji Watanabe, and Shrikanth Narayanan, “A review
of speaker diarization: Recent advances with deep learning,”
Computer Speech & Language, vol. 72, no. 7, pp. 101317,
2022.

[8] Mireia Diez, Luk´aˇs Burget, Shuai Wang, Johan Rohdin, and
Jan ˇCernock`y, “Bayesian HMM based x-vector clustering for
speaker diarization,” in Proc. Interspeech, 2019, pp. 346–350.
[9] Latan´e Bullock, Herv´e Bredin, and Leibny Paola Garcia-
“Overlap-aware diarization: Resegmentation using
in Proc.

Perera,
neural end-to-end overlapped speech detection,”
ICASSP, 2020, pp. 7114–7118.

[10] Shota Horiguchi, Paola Garcia, Yusuke Fujita, Shinji Watan-
abe, and Kenji Nagamatsu, “End-to-end speaker diarization as
post-processing,” in Proc. ICASSP, 2021, pp. 7188–7192.
[11] Tobias Cord-Landwehr, Christoph Boeddeker, C˘at˘alin Zoril˘a,
Rama Doddipatla, and Reinhold Haeb-Umbach, “Geodesic in-
terpolation of frame-wise speaker embeddings for the diariza-
tion of meeting scenarios,” in Proc. ICASSP, 2024, pp. 11886–
11890.

[12] Yusuke Fujita, Naoyuki Kanda, Shota Horiguchi, Yawen Xue,
Kenji Nagamatsu, and Shinji Watanabe, “End-to-end neural
speaker diarization with self-attention,” in Proc. ASRU, 2019,
pp. 296–303.

[13] Shota Horiguchi, Yusuke Fujita, Shinji Watanabe, Yawen Xue,
and Paola Garc´ıa, “Encoder-decoder based attractors for end-
IEEE/ACM TASLP, vol. 30, pp.
to-end neural diarization,”
1493–1507, 2022.

[14] Ivan Medennikov, Maxim Korenevsky, Tatiana Prisyach, Yuri
Khokhlov, Mariya Korenevskaya, Ivan Sorokin, Tatiana Timo-
feeva, Anton Mitrofanov, Andrei Andrusenko, Ivan Podluzhny,
Aleksandr Laptev, and Aleksei Romanenko, “Target-speaker
voice activity detection: a novel approach for multi-speaker
diarization in a dinner party scenario,” in Proc. Interspeech,
2020, pp. 274–278.

[15] Natsuo Yamashita, Shota Horiguchi, and Takeshi Homma,
“Improving the naturalness of simulated conversations for end-
to-end neural diarization,” in Proc. Odyssey, 2022, pp. 133–
140.

[16] Federico Landini, Alicia Lozano-Diez, Mireia Diez, and Luk´aˇs
Burget, “From simulated mixtures to simulated conversations
as training data for end-to-end neural diarization,” in Proc.
Interspeech, 2022, pp. 5095–5099.

[17] Federico Landini, Mireia Diez, Alicia Lozano-Diez, and Luk´aˇs
“Multi-speaker and wide-band simulated conversa-
in

Burget,
tions as training data for end-to-end neural diarization,”
Proc. ICASSP, 2023.

[18] Keisuke Kinoshita, Marc Delcroix, and Naohiro Tawara, “In-
tegrating end-to-end neural and clustering-based diarization:
Getting the best of both worlds,” in Proc. ICASSP, 2021, pp.
7198–7202.

[19] Keisuke Kinoshita, Marc Delcroix, and Naohiro Tawara, “Ad-
vances in integration of end-to-end neural and clustering-based
in Proc. Inter-
diarization for real conversational speech,”
speech, 2021, pp. 3565–3569.

[20] Mao-Kui He, Jun Du, Qing-Feng Liu, and Chin-Hui Lee,
“ANSD-MA-MSE: Adaptive neural speaker diarization using
memory-aware multi-speaker embedding,” IEEE/ACM TASLP,
vol. 31, pp. 1561–1573, 2023.

[21] Shaojin Ding, Quan Wang, Shuo-yiin Chang, Li Wan, and Ig-
nacio Lopez Moreno, “Personal VAD: Speaker-conditioned
voice activity detection,” in Proc. Odyssey, 2020, pp. 433–439.

[22] Maokui He, Desh Raj, Zili Huang, Jun Du, Zhuo Chen, and
Shinji Watanabe, “Target-speaker voice activity detection with
improved i-vector estimation for unknown number of speak-
ers,” in Proc. Interspeech, 2021, pp. 3555–3559.

[23] Katerina Zmolikova, Marc Delcroix, Tsubasa Ochiai, Keisuke
Kinoshita, Jan ˇCernock`y, and Dong Yu, “Neural target speech
extraction: An overview,” IEEE Signal Processing Magazine,
vol. 40, no. 3, pp. 8–29, 2023.

[24] Jun Wang, Jie Chen, Dan Su, Lianwu Chen, Meng Yu, Yanmin
Qian, and Dong Yu, “Deep extractor network for target speaker
recovery from single channel speech mixtures,” in Proc. Inter-
speech, 2018, pp. 307–311.

[25] Quan Wang, Hannah Muckenhirn, Kevin Wilson, Prashant
Sridhar, Zelin Wu, John Hershey, Rif A Saurous, Ron J Weiss,
Ye Jia, and Ignacio Lopez Moreno,
“VoiceFilter: Targeted
voice separation by speaker-conditioned spectrogram mask-
ing,” in Proc. Interspeech, 2019, pp. 2728–2732.

[26] Marc Delcroix, Katerina Zmolikova, Keisuke Kinoshita, At-
sunori Ogawa, and Tomohiro Nakatani, “Single channel tar-
get speaker extraction and recognition with speaker beam,” in
Proc. ICASSP, 2018, pp. 5554–5558.

[27] Quan Wang, Ignacio Lopez Moreno, Mert Saglam, Kevin Wil-
son, Alan Chiao, Renjie Liu, Yanzhang He, Wei Li, Jason Pele-
canos, Marily Nika, and Alexander Gruenstein, “VoiceFilter-
Lite: Streaming targeted voice separation for on-device speech
recognition,” in Proc. Interspeech, 2020, pp. 2677–2681.

[28] Takafumi Moriya, Hiroshi Sato, Tsubasa Ochiai, Marc Del-
croix, and Takahiro Shinozaki, “Streaming target-speaker ASR
with neural transducer,” in Proc. Interspeech, 2022, pp. 2673–
2677.

[29] Xiong Xiao, Naoyuki Kanda, Zhuo Chen, Tianyan Zhou,
Takuya Yoshioka, Sanyuan Chen, Yong Zhao, Gang Liu,
Yu Wu, Jian Wu, Shujie Liu, Jinyu Li, and Yifan Gong, “Mi-
crosoft speaker diarization system for the VoxCeleb speaker
in Proc. ICASSP, 2021, pp.
recognition challenge 2020,”
5824–5828.

[30] Hiroshi Sato, Tsubasa Ochiai, Marc Delcroix, Keisuke Ki-
noshita, Naoyuki Kamo, and Takafumi Moriya, “Learning to
enhance or not: Neural network-based switching of enhanced
and observed signals for overlapping speech recognition,” in
Proc. ICASSP, 2022, pp. 6287–6291.

[31] Hyewon Han, Soo-Whan Chung, and Hong-Goo Kang, “MIR-
Net: Learning multiple identities representations in overlapped
speech,” in Proc. Interspeech, 2020, pp. 4303–4307.

[32] Tobias Cord-Landwehr, Christoph Boeddeker, C˘at˘alin Zoril˘a,
Rama Doddipatla, and Reinhold Haeb-Umbach, “A teacher-
student approach for extracting informative speaker embed-
dings from speech mixtures,” in Proc. Interspeech, 2023, pp.
4703–4707.

[33] David Snyder, Daniel Garcia-Romero, Grregory Sell, Daniel
Povey, and Sanjeev Khudanpur, “X-vectors: Robust DNN em-
beddings for speaker recognition,” in Proc. ICASSP, 2018, pp.
5329–5333.

[34] Li Wan, Quan Wang, Alan Papir, and Ignacio Lopez Moreno,
“Generalized end-to-end loss for speaker verification,” in Proc.
ICASSP, 2018, pp. 4879–4883.

[35] Arsha Nagrani, Joon Son Chung, and Andrew Zisserman,
“VoxCeleb: A large-scale speaker identification dataset,” in
Proc. Interspeech, 2017, pp. 2616–2620.

[36] Koji Okabe, Takafumi Koshinaka, and Koichi Shinoda, “At-
tentive statistics pooling for deep speaker embedding,” in Proc.
Interspeech, 2018, pp. 2252–2256.

[37] Miquel India, Pooyan Safari, and Javier Hernando, “Self multi-
head attention for speaker recognition,” in Proc. Interspeech,
2019, pp. 4305–4309.

[38] Yanfeng Wu, Chenkai Guo, Hongcan Gao, Xiaolei Hou, and
Jing Xu, “Vector-based attentive pooling for text-independent
speaker verification.,” in Proc. Interspeech, 2020, pp. 936–940.

[39] Fangyuan Wang, Zhigang Song, Hongchen Jiang, and Bo Xu,
“MACCIF-TDNN: Multi aspect aggregation of channel and
context interdependence features in TDNN-based speaker ver-
ification,” in Proc. ASRU, 2021, pp. 214–219.

[40] Sung Hwan Mun, Jee-weon Jung, Min Hyun Han, and
Nam Soo Kim, “Frequency and multi-scale selective kernel
attention for speaker verification,” in Proc. SLT, 2023, pp. 548–
554.

[41] Zhenduo Zhao, Zhuo Li, Wenchao Wang, and Pengyuan
Zhang, “PCF: ECAPA-TDNN with progressive channel fusion
for speaker verification,” in Proc. ICASSP, 2023.

[42] Keisuke Kinoshita, Lukas Drude, Marc Delcroix, and Tomo-
hiro Nakatani, “Listening to each speaker one by one with
recurrent selective hearing networks,” in Proc. ICASSP, 2018,
pp. 5064–5068.

[43] Zhaopeng Tu, Zhengdong Lu, Yang Liu, Xiaohua Liu, and
Hang Li, “Modeling coverage for neural machine translation,”
in Proc. ACL, 2016, pp. 76–85.

[44] Abigail See, Peter J. Liu, and Christopher D. Manning, “Get
to the point: Summarization with pointer-generator networks,”
in Proc. ACL, 2017, pp. 1073–1083.

[45] Zhuo Chen, Takuya Yoshioka, Liang Lu, Tianyan Zhou, Zhong
Meng, Yi Luo, Jian Wu, Xiong Xiao, and Jinyu Li, “Continu-
ous speech separation: Dataset and analysis,” in Proc. ICASSP,
2020, pp. 7284–7288.

[46] Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kot-
sia, and Stefanos Zafeiriou, “ArcFace: Additive angular mar-
gin loss for deep face recognition,” IEEE TPAMI, vol. 44, no.
10, pp. 5962–5979, 2022.

[47] Arsha Nagrani, Joon Son Chung, Weidi Xie, and Andrew Zis-
serman, “VoxCeleb: Large-scale speaker verification in the
wild,” Computer Speech & Language, vol. 60, pp. 101027,
2020.

[48] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun,
in Proc.

“Deep residual learning for image recognition,”
CVPR, 2016, pp. 770–778.

[49] Shuai Wang, Johan Rohdin, Oldˇrich Plchot, Luk´aˇs Burget, Kai
Yu, and Jan ˇCernock`y,
“Investigation of SpecAugment for
deep speaker embedding learning,” in Proc. ICASSP, 2020,
pp. 7139–7143.

[50] David Snyder, Guoguo Chen, and Daniel Povey, “MUSAN: A
music, speech, and noise corpus,” arXiv:1510.08484, 2015.

[51] Tom Ko, Vijayaditya Peddinti, Daniel Povey, Michael L
Seltzer, and Sanjeev Khudanpur, “A study on data augmen-
tation of reverberant speech for robust speech recognition,” in
Proc. ICASSP, 2017, pp. 5220–5224.

[52] John R. Hershey, Zhuo Chen, Jonathan Le Roux, and Shinji
Watanabe, “Deep clustering: Discriminative embeddings for
segmentation and separation,” in Proc. ICASSP, 2016, pp. 31–
35.

[53] Diederik P. Kingma and Jimmy Ba, “Adam: A method for

stochastic optimization,” in Proc. ICLR, 2015.

[54] Jean Carletta, “Unleashing the killer corpus: experiences in
creating the multi-everything AMI Meeting Corpus,” Lan-
guage Resources and Evaluation, vol. 41, no. 2, pp. 181–190,
2007.

[55] Tae Jin Park, Kyu J. Han, Manoj Kumar, and Shrikanth
Narayanan, “Auto-tuning spectral clustering for speaker di-
arization using normalized maximum eigengap,” IEEE Signal
Processing Letters, vol. 27, pp. 381–385, 2020.

[56] Desh Raj, Zili Huang, and Sanjeev Khudanpur, “Multi-class
spectral clustering with overlaps for speaker diarization,” in
Proc. SLT, 2021, pp. 582–589.

[57] Jee-weon

Jung, Wangyou Zhang,

Jiatong Shi, Za-
karia Aldeneh, Takuya Higuchi, Barry-John Theobald,
Ahmed Hussen Abdelaziz, and Shinji Watanabe,
“ESPnet-
SPK: Full pipeline speaker embedding toolkit with repro-
ducible recipes, self-supervised front-ends, and off-the-shelf
models,” in Proc. Interspeech, 2024.

[58] Jee-weon Jung, Hee-Soo Heo, Hye-Jin Shim, and Ha-Jin
Yu, “Short utterance compensation in speaker verification via
cosine-based teacher-student learning of speaker embeddings,”
in Proc. ASRU, 2019, pp. 335–341.

