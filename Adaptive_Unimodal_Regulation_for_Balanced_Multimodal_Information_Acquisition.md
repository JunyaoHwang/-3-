# Adaptive Unimodal Regulation for Balanced Multimodal Information Acquisition

Chengxiang Huang ${}^{1, \dagger  }$ Yake Wei ${}^{2, \dagger  }$ Zequn Yang ${}^{2}\;$ Di Hu ${}^{2, * }$

${}^{1}$ Beijing University of Posts and Telecommunications ${}^{2}$ Renmin University of China

huangchengxiang2021@bupt.edu.cn, \{yakewei, zqyang, dihu\}@ruc.edu.cn

## Abstract

Sensory training during the early ages is vital for human development. Inspired by this cognitive phenomenon, we observe that the early training stage is also important for the multimodal learning process, where dataset information is rapidly acquired. We refer to this stage as the prime learning window. However, based on our observation, this prime learning window in multimodal learning is often dominated by information-sufficient modalities, which in turn suppresses the information acquisition of information-insufficient modalities. To address this issue, we propose Information Acquisition Regulation (InfoReg), a method designed to balance information acquisition among modalities. Specifically, InfoReg slows down the information acquisition process of information-sufficient modalities during the prime learning window, which could promote information acquisition of information-insufficient modalities. This regulation enables a more balanced learning process and improves the overall performance of the multimodal network. Experiments show that InfoReg outperforms related multimodal imbalanced methods across various datasets, achieving superior model performance. The code is available at https://github.com/GeWu-Lab/InfoReg_CVPR2025.

## 1. Introduction

Learning during early developmental ages in humans and animals is important for skill impairments [16, 22, 44]. Similarly, in deep learning, recent studies have found that models learn in stages, with early learning being especially important for effective information acquisition [2, 19].

The above circumstance motivates us to investigate the process of information acquisition in the multimodal scenario. Due to the presence of multiple modalities, multimodal learning networks are expected to capture sufficient information from each modality $\left\lbrack  {{12},{40}}\right\rbrack$ . To observe the information acquisition of different modalities, we use the trace of the Fisher Information Matrix $\left\lbrack  {2,{10}}\right\rbrack$ to investigate the information amount of each modality in multimodal models. However, based on our observation, the amount of information in multimodal models is not well consistent with intuitive expectations. Firstly, as shown in Figure 1a, the green curve, representing the overall multimodal network, demonstrates a rapid increase in information amount during the early periods. Information acquisition is most rapid in this early stage, which we refer to as the prime learning window, reaching a peak at the end of this period, followed by a subsequent decline. Secondly, for audio modality, its overall trend closely aligns with the overall multimodal model, and shows a high information acquisition amount during the prime learning window. These findings align with our expectation that modality information could be acquired effectively within the prime learning window. However, the video modality, represented by the blue curve, shows a much lower information acquisition amount during the prime learning window. Although the video modality is capable of effective information acquisition in the unimodal scenario in Figure 1b, it fails to do so in the multimodal scenario when trained jointly with the audio modality. These observations suggest that information-insufficient modalities, like video, experience suppressed information acquisition during the prime learning window due to the stronger information acquisition capacity of information-sufficient modalities, such as audio. Moreover, as the multimodal model's capacity for information acquisition diminishes in later stages, the imbalance observed during the prime learning window cannot be compensated for by simply extending the training time, as demonstrated in Table 1.

![bo_d3s8tajef24c73d1aucg_0_926_674_705_315_0.jpg](images/bo_d3s8tajef24c73d1aucg_0_926_674_705_315_0.jpg)

Figure 1. (a). Information amount variation of the audio encoder, video encoder, and multimodal model during the training process on CREMA-D [7]. (b). Information amount variation of the audio and video modalities when trained independently on CREMA-D.

---

${}^{ \dagger  }$ Equal contribution. ${}^{ * }$ Corresponding author.

---

<table><tr><td>$\mathbf{{Method}}$</td><td>Accuracy</td></tr><tr><td>Video in Joint training (50 epochs)</td><td>35.14</td></tr><tr><td>Video in Joint training (100 epochs)</td><td>35.36</td></tr><tr><td>Video in InfoReg (50 epochs)</td><td>49.65</td></tr></table>

Table 1. Comparison of performance for the video modality between Joint training and InfoReg on the CREMA-D dataset. Simply extending the training time (100 epochs) cannot compensate for the suppressed video modality.

Recent studies have investigated the problem of imbal-anced learning across modalities in multimodal learning $\left\lbrack  {{18},{32},{41}}\right\rbrack$ . Although several studies have made progress in addressing this issue $\left\lbrack  {8,{28},{35},{42}}\right\rbrack$ , their methods typically apply adjustments across the entire training process without recognizing the importance of the prime learning window. Consequently, the effectiveness of these methods is significantly constrained. Our research reveals that the prime learning window plays a vital role in multimodal learning. In this window, there is a significant imbalance in information acquisition across modalities. Therefore, effective adjustment within this window is essential for achieving balanced information acquisition across modalities.

Based on the above analysis, information-insufficient modalities experience a significantly reduced information acquisition amount during the prime learning window due to suppression from information-sufficient modalities. To address this imbalance, we slow the information acquisition rate of information-sufficient modalities within this window, thereby allowing information-insufficient modalities to improve their acquisition. Hence, we propose our method, Information Acquisition Regulation (In-foReg). In InfoReg, the process begins by determining whether information-sufficient modalities are within the prime learning window. If so, a unimodal regulation term is applied to regulate the Fisher Information [2, 10], thereby restricting these modalities acquire information. As a result, InfoReg promotes a more balanced information acquisition across modalities, enhancing the overall performance of the model. Our extensive experiments across multiple datasets show that InfoReg achieves superior performance and improves modality balance by helping information-insufficient modalities acquire more information during the prime learning window. Additionally, InfoReg enhances feature quality, supporting prior research on the link between early-stage information acquisition and robust feature representation $\left\lbrack  {1,2}\right\rbrack$ . These results collectively validate the effectiveness of our approach.

Our main contributions are summarized as follows:

- Firstly, we identify the prime learning window in multimodal learning, a critical period where imbalances in information acquisition significantly impact modality balance and overall performance.

- Secondly, We analyze the imbalance of Fisher Information among modalities and propose InfoReg, a method that regulates the information acquisition of information-sufficient modalities during the prime learning window.

- Finally, we validate InfoReg on multiple datasets and settings, demonstrating its considerable improvement while maintaining balanced performance across modalities.

## 2. Related Work

### 2.1. Multimodal imbalance learning

Jointly training multiple modalities is intuitively expected to enhance performance [30]. However, imbalanced learning across modalities poses substantial challenges, hindering the effective training of multimodal networks [3, 17, 36, 37]. Previous studies have investigated the imbalanced learning problem across modalities in multimodal learning networks [32, 41], where certain modalities tend to dominate, limiting the learning of other modalities. This imbalance can degrade overall performance and lead to significant disparities between modalities, sometimes even causing multimodal learning to underperform compared to single-modality scenarios [32]. To address this problem, many methods $\left\lbrack  {{28},{32},{41} - {43}}\right\rbrack$ have been proposed, primarily focusing on balancing the optimization of each modality throughout the training process. BalanceBench [47] further categorizes these methods based on their distinct characteristics, offering a comprehensive framework to evaluate their effectiveness and limitations. Specifically, OGM [32] balances modalities by adjusting the gradients of well-learned modalities. MMPareto [42] leverages an optimized Pareto front to balance the performance across modalities, aiming to improve the generalization of multimodal models. Despite the successes of previous work, they have largely overlooked the impact of different training periods on information acquisition across various modalities. We have recognized this issue and designed a method accordingly. By slowing down the information acquisition rate of information-sufficient modalities, our approach alleviates the suppression of information-insufficient modalities, allowing them to acquire more information in the early stages, effectively balancing the performance of different modalities. This leads to considerable improvements in both overall performance and modality balance.

### 2.2. Early stages in deep learning

The learning process of deep learning models consists of two main phases: an initial phase of information acquisition from the dataset, followed by a phase of gradual information compression or forgetting $\left\lbrack  {1,{19},{21},{48}}\right\rbrack$ . Recent studies highlight the crucial role of early-stage learning in shaping feature representation and overall model performance $\left\lbrack  {{11},{13},{20},{27},{49}}\right\rbrack$ , similar to findings in neuroscience $\left\lbrack  {6,{16},{22},{45}}\right\rbrack$ . Further, the information missed during the early stages cannot be recovered by extending the training duration later on [2]. We define this early important learning period as the prime learning window. However, previous research has overlooked the importance of the prime learning window in multimodal learning. Our findings reveal that balancing information acquisition among modalities during this window leads to considerable performance improvements in multimodal models.

## 3. Method

### 3.1. Multimodal learning framework

For convenience, let the dataset be denoted by $D =$ ${\left\{  \left( {x}_{n},{y}_{n}\right) \right\}  }_{n = 0,1,\ldots , N - 1}$ , where each ${x}_{n}$ contains inputs from $M$ modalities: ${x}_{n} = \left( {{x}_{n}^{1},{x}_{n}^{2},\ldots ,{x}_{n}^{M}}\right)$ . The target label ${y}_{n} \in  \{ 1,2,\ldots \}$ represents the class of sample ${x}_{n}$ . For each modality $m$ , where $m \in  \{ 1,2,\ldots , M\}$ , the input is processed through the corresponding encoder ${\varphi }^{m}\left( {{w}_{m}, \cdot  }\right)$ . Here, ${w}_{m}$ are the weights of encoder $m$ . After feature extraction, the outputs are concatenated and passed to a single-layer linear classifier. Finally, one joint multimodal cross-entropy loss ${\mathcal{L}}_{\text{joint }}\left( w\right)$ is utilized to optimize the model.

### 3.2. Fisher Information in multimodal learning

Following previous work $\left\lbrack  {1,{34}}\right\rbrack$ that uses the Kullback-Leibler (KL) divergence to measure the information contained in the weights of a unimodal network, the information acquisition process for a single modality $m$ can be evaluated by this metric. Given the posterior distribution ${p}_{{w}_{m}}\left( {y \mid  x;D}\right)$ , encoded by the unimodal encoder with weights ${w}_{m}$ and its prior distribution ${q}_{{w}_{m}}\left( {y \mid  x}\right)$ , the mutual information is defined as follows:

$$
{D}_{KL}\left( {{p}_{{w}_{m}}\parallel {q}_{{w}_{m}}}\right)  = \int {p}_{{w}_{m}}\left( {y \mid  x;D}\right) \log \frac{{p}_{{w}_{m}}\left( {y \mid  x;D}\right) }{{q}_{{w}_{m}}\left( {y \mid  x}\right) }{dy}.
$$

(1)

However, quantifying the information a unimodal encoder acquires from the dataset is challenging because the ground truth prior distribution ${q}_{{w}_{m}}\left( {y \mid  x}\right)$ is not accessible. As an alternative, inspired by [2], the rate of information acquisition from the dataset can be estimated by calculating the KL divergence between distributions encoded by weights at successive moments in training. Specifically, given a perturbation ${w}_{m}^{\prime } = {w}_{m} + \delta {w}_{m}$ . The discrepancy between the distributions ${p}_{{w}_{m}}\left( {y \mid  x;D}\right)$ and ${p}_{{w}_{m}^{\prime }}\left( {y \mid  x;D}\right)$ reflects the rate of information acquisition and can be defined as:

$$
{D}_{KL}\left( {{p}_{{w}_{m}}\parallel {p}_{{w}_{m}^{\prime }}}\right)  = \int {p}_{{w}_{m}}\left( {y \mid  x;D}\right) \log \frac{{p}_{{w}_{m}}\left( {y \mid  x;D}\right) }{{p}_{{w}_{m}^{\prime }}\left( {y \mid  x;D}\right) }{dy}.
$$

(2)

Further, the KL divergence can be approximated to second order by applying the Taylor expansion:

$$
{D}_{KL}\left( {{p}_{{w}_{m}}\parallel {p}_{{w}_{m}}^{\prime }}\right)  \approx  \frac{1}{2}\delta {w}_{m}^{T}{F}_{m}\delta {w}_{m}, \tag{3}
$$

where ${F}_{m}$ is the Fisher Information Matrix (FIM) [10], and is defined as:

$$
{F}_{m} = {\mathbb{E}}_{y \sim  {p}_{w}}\left\lbrack  {{\nabla }_{{w}_{m}}\log {p}_{{w}_{m}}\left( {y \mid  x}\right) {\nabla }_{{w}_{m}}\log {p}_{{w}_{m}}{\left( y \mid  x\right) }^{T}}\right\rbrack  .
$$

(4)

The FIM plays a crucial role in quantifying the amount of information captured by the deep neural network $\left\lbrack  {2,{26}}\right\rbrack$ and acts as a local measure, assessing how small perturbations in the model's parameters influence its output [4]. Additionally, the FIM is a semi-definite approximation of the Hessian matrix, providing insights into the curvature of the loss landscape at a given point during training [31, 38].

In multimodal learning, for a unimodal encoder ${\varphi }^{m}$ , the gradient of the encoder can be expressed as :

$$
{g}_{{\varphi }^{m}}\left( {{w}_{m},{x}_{n}^{m}}\right)  = {\nabla }_{{w}_{m}}\log {p}_{{w}_{m}}\left( {{y}_{n} \mid  {x}_{n}^{m}}\right)  = {\nabla }_{{w}_{m}}{\mathcal{L}}_{\text{joint }}\left( {w}_{m}\right) .
$$

(5)

Based on this, the Fisher Information Matrix ${F}_{m}$ can be reformulated as:

$$
{F}_{m} = {\mathbb{E}}_{{x}_{n}^{m} \sim  {X}^{m}}\left\lbrack  {{g}_{{\varphi }^{m}}\left( {{w}_{m},{x}_{n}^{m}}\right) {g}_{{\varphi }^{m}}{\left( {w}_{m},{x}_{n}^{m}\right) }^{T}}\right\rbrack  . \tag{6}
$$

However, computing ${F}_{m}$ directly is computationally expensive. To address this, we use the trace of the Fisher Information Matrix, denoted as $\operatorname{Tr}\left( {F}_{m}\right)$ , to measure the amount of information captured by the deep neural network. This trace can be computed more efficiently and is defined as:

$$
\operatorname{Tr}\left( {F}_{m}\right)  = {\mathbb{E}}_{{x}_{n}^{m} \sim  {X}^{m}}\left\lbrack  {\begin{Vmatrix}{g}_{{\varphi }^{m}}\left( {w}_{m},{x}_{n}^{m}\right) \end{Vmatrix}}^{2}\right\rbrack  . \tag{7}
$$

As shown in Figure 1, $\operatorname{Tr}\left( {F}_{m}\right)$ could effectively measure the amount of information acquired and identify the prime learning window.

As illustrated in Figure 3a, information-sufficient modalities will exhibit significantly larger values of ${g}_{{\varphi }^{m}}$ during the prime learning window. Due to the squared term in Equation 7, these substantial differences in ${g}_{{\varphi }^{m}}$ between modalities are further amplified, thereby making the imbalance in Fisher Information even more pronounced. This indicates that information-sufficient modalities have a clear advantage in the information acquisition during the prime learning window and dominate the overall information acquisition of the multimodal model.

![bo_d3s8tajef24c73d1aucg_3_343_215_1100_565_0.jpg](images/bo_d3s8tajef24c73d1aucg_3_343_215_1100_565_0.jpg)

Figure 2. Overview of InfoReg. This figure shows the main components and workflow of InfoReg. The left side presents our overall framework, while the right side highlights the adaptive unimodal regulation. During the training, InfoReg first identifies the information-sufficient modalities, then evaluates whether they are in the prime learning window, and finally applies adaptive unimodal regulation.

![bo_d3s8tajef24c73d1aucg_3_170_909_698_353_0.jpg](images/bo_d3s8tajef24c73d1aucg_3_170_909_698_353_0.jpg)

Figure 3. (a). The gradient gap between the audio modality and video modality on CREMA-D. (b). The $\operatorname{Tr}\left( {F}_{m}\right)$ gap between the audio modality and video modality on CREMA-D.

### 3.3. Information acquisition regulation

To address the Fisher Information imbalance observed during the prime learning window, our method regulates the Fisher Information of information-sufficient modalities in this important period to slow down their information acquisition, thereby promoting information acquisition in other modalities. Our method consists of the following main components:

- Evaluate the prime learning window for information-sufficient modalities: We first identify the information-sufficient modalities and then evaluate whether they fall within the prime learning window.

- Adaptive unimodal regulation: For information-sufficient modalities, we apply adaptive unimodal regulation to approximately regulate the Fisher Information.

#### 3.3.1 Evaluating the prime learning window.

Inspired by OGM [32], performance scores are used to identify information-sufficient modalities. Afterward, an assessment is made to determine whether these information-sufficient modalities are in the prime learning window. For each iteration, assume that training has reached epoch $t$ and is currently processing batch $b$ , where $b \in  \{ 0,1,\ldots , B - 1\}$ and $B$ represent the total number of batches. The performance score for each modality is given by:

$$
{s}_{m;b}^{t} = {\mathbb{E}}_{{x}_{n}^{m} \sim  {X}^{m}}\left\lbrack  {-\log \left( {\operatorname{softmax}{\left( {\varphi }^{m}\left( {x}_{n}^{m}\right) \right) }_{{y}_{n}}}\right) }\right\rbrack  . \tag{8}
$$

Then, we define its performance gap ${\Delta }_{m}$ relative to other modalities to determine information-sufficient modalities during the prime learning window. Let ${C}_{m}$ represent the number of modalities with performance scores less than ${s}_{m;b}^{t}$ :

$$
{C}_{m} = \left| \left\{  {{m}^{\prime } \in  \left\lbrack  M\right\rbrack  \smallsetminus \{ m\} ;{s}_{{m}^{\prime };b}^{t} < {s}_{m;b}^{t}}\right\}  \right| . \tag{9}
$$

The performance gap ${\Delta }_{m}$ can then be expressed as:

$$
{\Delta }_{m} = \frac{1}{{C}_{m}}\mathop{\sum }\limits_{{{m}^{\prime } \in  \left\lbrack  M\right\rbrack   \smallsetminus  \{ m\} ;{s}_{{m}^{\prime };b}^{t} < {s}_{m;b}^{t}}}\left( {{s}_{m;b}^{t} - {s}_{{m}^{\prime };b}^{t}}\right) , \tag{10}
$$

where ${\Delta }_{m}$ measures the average performance difference between $m$ and all lower or equally performing modalities. This ensures that for the lowest-performing modality, where ${s}_{m;b}^{t}$ is minimal, ${\Delta }_{m}$ will be 0 .

After identifying information-sufficient modalities based on ${s}_{m;b}^{t}$ , the following criterion is used to determine whether these modalities are in the prime learning window:

$$
\frac{\operatorname{Tr}\left( {F}_{m}^{t - 1}\right)  - \operatorname{Tr}\left( {F}_{m}^{t - 2}\right) }{\operatorname{Tr}\left( {F}_{m}^{t - 1}\right) } > K, \tag{11}
$$

where $K$ is a positive hyperparameter that controls the threshold for inclusion. Equation 11 reflects the relative changing rate of $\operatorname{Tr}\left( {F}_{m}\right)$ . A large value of this rate indicates that the information amount in the unimodal encoder is rapidly increasing. Conversely, a small or negative value suggests that information acquisition has slowed down or is even decreasing.

![bo_d3s8tajef24c73d1aucg_4_167_204_704_325_0.jpg](images/bo_d3s8tajef24c73d1aucg_4_167_204_704_325_0.jpg)

Figure 4. The cosine similarities of gradients across different batches within the prime learning window.

Algorithm 1 Pipeline of InfoReg

---

Input: Training dataset $D$ , number of epochs $T$ , number

		of batches $B$ of each batch, hyperparameter $\beta , K$

		for $t = 0,1,\cdots , T - 1$ do

			if $t < 2$ then

				Update model parameters;

				Calculate $\operatorname{Tr}\left( {F}_{m}^{t}\right)$ by Equation 7;

				Continue;

			end if

			Calculate $\frac{\operatorname{Tr}\left( {F}_{m}^{t - 1}\right)  - \operatorname{Tr}\left( {F}_{m}^{t - 2}\right) }{\operatorname{Tr}\left( {F}_{m}^{t - 1}\right) }$ ;

			for $b = 0,1,\cdots , B - 1$ do

				Randomly selects a batch of data from $D$ ;

				Calculate the performance scores ${s}_{m;b}^{t}$ for differ-

				ent modalities by Equation 8;

				Calculate ${\Delta }_{m}$ by Equation 10;

				Decide information-sufficient modalities by ${s}_{m;b}^{t}$ ;

				Calculate $\alpha$ by Equation 16;

				if $\frac{\operatorname{Tr}\left( {F}_{m}^{t - 1}\right)  - \operatorname{Tr}\left( {F}_{m}^{t - 2}\right) }{\operatorname{Tr}\left( {F}_{m}^{t - 1}\right) } > K$ and ${\Delta }_{m} > 0$ then

					Calculate the regulation term by Equation 12;

					Add adaptive unimodal regulation term;

				end if

				Update model parameters;

			end for

			Calculate $\operatorname{Tr}\left( {F}_{m}^{t}\right)$ by Equation 7.

		end for

---

#### 3.3.2 Adaptive unimodal regulation.

During the prime learning window, information-sufficient modalities dominate the model's information acquisition, limiting the ability of information-insufficient modalities to acquire information. Therefore, it becomes essential to regulate information acquisition of information-sufficient modalities in this window. However, directly calculating $\operatorname{Tr}\left( {F}_{m}^{t}\right)$ requires complete gradient information across the entire dataset, making it impractical to regulate effectively. Additionally, calculating $\operatorname{Tr}\left( {F}_{m}^{t}\right)$ over a full epoch is ineffective for adjusting the model at each iteration. To address this, we introduce a regulation term ${P}_{m;b}^{t}$ to approximately regulate the Fisher Information:

$$
{P}_{m;b}^{t} = \frac{\alpha }{2}{\begin{Vmatrix}{w}_{m;b}^{t} - {w}_{m}^{t - 1}\end{Vmatrix}}^{2}, \tag{12}
$$

where $\alpha$ is a parameter that controls the degree of regulation. This regulation term is proportional to $\operatorname{Tr}\left( {F}_{m;b}^{t}\right)$ for each batch. Let the gradient ${g}_{{\varphi }^{m}}$ be denoted as $g,\operatorname{Tr}\left( {F}_{m;b}^{t}\right)$ in each batch can be defined as:

$$
\operatorname{Tr}\left( {F}_{m;b}^{t}\right)  = \frac{1}{b}\mathop{\sum }\limits_{{k = 0}}^{b}{\begin{Vmatrix}{g}_{k}^{t}\end{Vmatrix}}^{2}. \tag{13}
$$

The regulation term ${P}_{m;b}^{t}$ can be written as:

$$
{P}_{m;b}^{t} = \frac{\alpha }{2}{\begin{Vmatrix}{w}_{m;b}^{t} - {w}_{m}^{t - 1}\end{Vmatrix}}^{2}
$$

$$
= \frac{\alpha }{2}{\begin{Vmatrix}-\eta \mathop{\sum }\limits_{{k = 0}}^{b}{g}_{k}^{t}\end{Vmatrix}}^{2}
$$

$$
= \frac{\alpha {\eta }^{2}}{2}\left( {\mathop{\sum }\limits_{{k = 0}}^{b}{\begin{Vmatrix}{g}_{k}^{t}\end{Vmatrix}}^{2} + 2\mathop{\sum }\limits_{{0 \leq  z < k \leq  b}}{g}_{z}^{t}{\left( {g}_{k}^{t}\right) }^{T}}\right) , \tag{14}
$$

where $\eta$ represents the learning rate. The high dimensionality of ${g}_{b}^{t}$ results in the gradients ${g}_{z}^{t}{\left( {g}_{k}^{t}\right) }^{T}$ becoming approximately orthogonal for any two batches $z$ and $k$ . The detailed proof is provided in the Appendix A. As illustrated in Figure 4 , we compare the cosine similarities between gradients of different batches, and the results confirm that they are approximately orthogonal. Then, ${P}_{m;b}^{t}$ can be approximated

as:

$$
{P}_{m;b}^{t} = \frac{\alpha {\eta }^{2}}{2}\mathop{\sum }\limits_{{k = 0}}^{b}{\begin{Vmatrix}{g}_{k}^{t}\end{Vmatrix}}^{2}. \tag{15}
$$

According to the Equation 13 and Equation 15, the unimodal regulation term ${P}_{m;b}^{t}$ regulates $\operatorname{Tr}\left( {F}_{m;b}^{t}\right)$ , thereby limiting the amount of information acquired by the information-sufficient modalities. To modulate the impact of ${P}_{m;b}^{t}$ , we define $\alpha$ as a dynamic parameter:

$$
\alpha  = \exp \left( {\beta  * \tanh \left( {\Delta }_{m}\right) }\right) , \tag{16}
$$

where $\beta$ is a hyperparameter controlling the sensitivity of $\alpha$ to the performance gap ${\Delta }_{m}$ . A higher value of $\alpha$ results in stronger regulation, thereby tightly constraining the information acquisition for information-sufficient modalities. This adaptive regulation ensures balance and prevents any single modality from dominating during the prime learning window. Additional analysis of the regulation term is provided in Appendix B. Overall, our method is shown in Algorithm 1 and illustrated in Figure 2.

<table><tr><td rowspan="2">$\mathbf{{Method}}$</td><td colspan="3">CREMA-D</td><td colspan="3">Kinetics Sounds</td></tr><tr><td>Accuracy</td><td>Acc audio</td><td>Acc video</td><td>Accuracy</td><td>Acc audio</td><td>Acc video</td></tr><tr><td>Joint training</td><td>66.61</td><td>58.99</td><td>35.14</td><td>65.67</td><td>53.13</td><td>36.01</td></tr><tr><td>OGM [32]</td><td>68.70</td><td>56.84</td><td>39.52</td><td>66.63</td><td>53.39</td><td>40.16</td></tr><tr><td>Greedy [46]</td><td>67.82</td><td>59.17</td><td>40.17</td><td>66.54</td><td>53.15</td><td>37.82</td></tr><tr><td>PMR [9]</td><td>66.92</td><td>57.83</td><td>38.91</td><td>66.33</td><td>53.42</td><td>36.17</td></tr><tr><td>AGM [28]</td><td>69.71</td><td>59.32</td><td>43.72</td><td>66.54</td><td>53.12</td><td>37.24</td></tr><tr><td>InfoReg</td><td>71.90</td><td>60.03</td><td>49.65</td><td>69.31</td><td>54.16</td><td>44.73</td></tr></table>

Table 2. Comparison with imbalanced multimodal learning methods. All the methods only use one multimodal loss. Bold and underline represent the best and second best respectively.

<table><tr><td rowspan="2">$\mathbf{{Method}}$</td><td colspan="3">CREMA-D</td><td colspan="3">Kinetics Sounds</td></tr><tr><td>Accuracy</td><td>Acc audio</td><td>Acc video</td><td>Accuracy</td><td>Acc audio</td><td>Acc video</td></tr><tr><td>Joint training</td><td>66.61</td><td>58.99</td><td>35.14</td><td>65.67</td><td>54.13</td><td>36.01</td></tr><tr><td>Joint training*</td><td>70.81</td><td>60.52</td><td>55.23</td><td>68.71</td><td>55.23</td><td>44.18</td></tr><tr><td>G-Blending [41]</td><td>69.11</td><td>60.14</td><td>51.29</td><td>68.33</td><td>54.22</td><td>42.31</td></tr><tr><td>MMPareto [42]</td><td>73.08</td><td>60.83</td><td>58.92</td><td>71.11</td><td>56.47</td><td>53.39</td></tr><tr><td>InfoReg*</td><td>75.71</td><td>61.63</td><td>61.22</td><td>72.03</td><td>57.21</td><td>53.57</td></tr></table>

Table 3. Comparison with imbalanced multimodal learning methods with unimodal loss. Joint training* and InfoReg* denote Joint training with unimodal loss and InfoReg with unimodal loss, repectively. Bold and underline represent the best and second best.

## 4. Experiments

### 4.1. Dataset and experimental settings

CREMA-D [7] is an emotion recognition dataset with recordings of actors expressing six emotions, providing audio-visual samples to examine how auditory and visual cues convey emotion. Kinetics Sounds [5, 23] is designed for human action recognition, featuring 31 action classes from varied video sources, allowing analysis of audio-visual integration in dynamic activity recognition. CMU-MOSI [50] is a sentiment analysis dataset with short video clips including audio, visual, and text modalities, suitable for exploring multimodal sentiment expression.

For our model architecture, we employ ResNet-18 [15] as the backbone for the CREMA-D and Kinetics Sounds, while for the CMU-MOSI, we use a transformer-based model [29]. All models are trained from scratch to ensure that the feature extraction processes are fully optimized for our specific tasks and datasets. Additionally, we implement a late fusion method to integrate uni-modal features from different modalities.

### 4.2. Comparison with related imbalanced methods

To evaluate the effectiveness of InfoReg in addressing information acquisition imbalance across modalities, we compare InfoReg with several imbalanced multimodal learning approaches, including G-Blending [41], OGM [32], Greedy [46], PMR [9], AGM [28], and MMPareto [42]. Of these methods, G-Blending [41] and MMPareto [42] utilize both unimodal and multimodal losses, while OGM [32], Greedy [46], PMR [9], and AGM [28] rely only on unimodal loss. In our evaluation framework, Joint training denotes the widely used baseline for imbalanced multimodal learning, utilizing concatenation fusion with a single multimodal cross-entropy loss function [28, 32]. Meanwhile, Joint training* represents the scenario where both unimodal and multimodal joint losses are applied simultaneously.

As shown in Table 2 and Figure 5a, we conduct experiments on CREMA-D and Kinetics Sounds using In-foReg and several related methods that employ only multimodal loss. Based on the results, we first observe that all imbalanced methods achieve improved performance, indicating the significance of the modality imbalance issue and the need for balancing unimodal learning. Furthermore, InfoReg consistently outperforms other methods, especially with a notable improvement in the video modality. This suggests that regulating information acquisition during the prime learning window effectively balances information gain across modalities, enabling information-sufficient modalities to acquire more information and enhancing overall model performance.

Additionally, we provide experiments with methods incorporating both multimodal loss and unimodal loss. Here, Joint training* and InfoReg* indicate both multimodal loss and unimodal loss are applied. As shown in Table 3, all methods with unimodal loss achieve notable improvements over Joint training. This improvement is attributed to the inclusion of unimodal loss, which facilitates more efficient information retrieval from individual modalities, thereby enhancing overall performance. Moreover, InfoReg* continues to outperform other approaches and achieve better modality balance. This indicates that, with unimodal assistance, InfoReg can still effectively balance information acquisition across modalities, thereby achieving good modality balance and model performance.

![bo_d3s8tajef24c73d1aucg_6_235_209_1319_293_0.jpg](images/bo_d3s8tajef24c73d1aucg_6_235_209_1319_293_0.jpg)

Figure 5. (a). The overall accuracy, audio accuracy, and video accuracy of InfoReg are compared with Joint training. (b). The value of ${Tr}\left( F\right)$ in InfoReg for both modalities. (c). The value of ${Tr}\left( F\right)$ of the video modality in InfoReg compared with that of Joint training. All

experiments are conducted on CREMA-D.

![bo_d3s8tajef24c73d1aucg_6_240_618_1318_315_0.jpg](images/bo_d3s8tajef24c73d1aucg_6_240_618_1318_315_0.jpg)

Figure 6. The representations of the video modality on CREMA-D by t-SNE [39] in InfoReg and Joint training are shown. "Extended Joint training" refers to Joint training that is extended to 100 epochs. Additional t-SNE representations are provided in Appendix C.

### 4.3. Evaluating information acquisition

We conduct experiments on CREMA-D to evaluate the effectiveness of InfoReg in enhancing information acquisition for information-insufficient modalities during the prime learning window. Firstly, As shown in Figure 5b, we observe that both modalities exhibit gradually growing values of $\operatorname{Tr}\left( F\right)$ within the prime learning window, indicating that each modality acquires sufficient information early on. On one hand, the information-sufficient modality, audio, continues to acquire adequate information despite the regulation during the prime learning window, allowing it to maintain good performance throughout training. On the other hand, the information-insufficient modality also acquires sufficient information, as InfoReg alleviates the suppression of its information acquisition by the information-sufficient modality. Secondly, we compare the $\operatorname{Tr}\left( F\right)$ values for the video modality in InfoReg during the prime learning window with those in Joint training. As shown in Figure 5c, the $\operatorname{Tr}\left( F\right)$ values for the video modality in InfoReg are consistently higher than those in Joint training, indicating a considerable improvement in information acquisition for the video modality during the prime learning window. This enhancement enables the multimodal model to learn more comprehensive information during the prime learning window, thereby improving the model's overall performance.

<table><tr><td rowspan="2">$\mathbf{{Method}}$</td><td colspan="3">CREMA-D</td></tr><tr><td>WTP</td><td>$\mathbf{{PLW}}$</td><td>Other periods</td></tr><tr><td>Joint training</td><td>66.61</td><td>-</td><td>-</td></tr><tr><td>OGM [32]</td><td>68.70</td><td>68.76</td><td>67.13</td></tr><tr><td>AGM [28]</td><td>69.71</td><td>69.54</td><td>67.41</td></tr><tr><td>PMR [9]</td><td>66.92</td><td>66.99</td><td>66.57</td></tr><tr><td>InfoReg</td><td>69.03</td><td>71.90</td><td>67.22</td></tr></table>

Table 4. Comparison of performance. "WTP" and "PLW" denotes the whole training process and the prime learning window respectively. "Other periods" indicates adjustments made outside the prime learning window. Bold and underline represent the best and second best.

<table><tr><td rowspan="2">$\mathbf{{Method}}$</td><td colspan="4">Fusion strategies</td></tr><tr><td>Gated</td><td>SUM</td><td>FiLM</td><td>Concat</td></tr><tr><td>Joint training</td><td>65.32</td><td>64.38</td><td>66.67</td><td>66.61</td></tr><tr><td>InfoReg</td><td>69.18</td><td>70.12</td><td>70.23</td><td>71.90</td></tr></table>

Table 5. Comparison of different fusion strategies.

### 4.4. Importance of the prime learning window

Unlike other methods, InfoReg introduces adjustments exclusively during the prime learning window. To evaluate the importance of the prime learning window in addressing the imbalanced multimodal learning problem, we compare the performance of several related methods with adjustments made exclusively during the prime learning window and during other periods. Firstly, all methods show considerable improvements when adjustments are made exclusively during the prime learning window. Specifically, the performance achieved by these methods during the prime learning window is comparable to that achieved by adjusting throughout the whole training process. This highlights the necessity of making adjustments within the prime learning window. Secondly, adjustments made during other periods show significantly lower performance compared to those made during the prime learning window and the whole training process, indicating that late-stage adjustments are unnecessary due to the lack of additional information at this stage. Overall, our method focuses on balancing information acquisition exclusively within the prime learning window, effectively enhancing underrepresented modalities during this period and resulting in strong performance.

<table><tr><td rowspan="2">$\mathbf{{Method}}$</td><td colspan="2">CMU-MOSI</td></tr><tr><td>Accuracy</td><td>Macro F1</td></tr><tr><td>Joint training</td><td>61.09</td><td>60.74</td></tr><tr><td>OGM [32]</td><td>61.88</td><td>61.32</td></tr><tr><td>PMR [9]</td><td>61.47</td><td>60.98</td></tr><tr><td>AGM [28]</td><td>61.39</td><td>60.43</td></tr><tr><td>InfoReg</td><td>62.31</td><td>62.03</td></tr></table>

Table 6. Comparison with imbalanced multimodal learning methods on the CMU-MOSI dataset. Bold and underline represent the best and second best respectively.

![bo_d3s8tajef24c73d1aucg_7_167_586_694_273_0.jpg](images/bo_d3s8tajef24c73d1aucg_7_167_586_694_273_0.jpg)

Figure 7. (a). The overall accuracy of different $\beta$ on CREMA-D. (b). The overall accuracy of different $\beta$ on Kinetics Sounds.

![bo_d3s8tajef24c73d1aucg_7_167_936_694_269_0.jpg](images/bo_d3s8tajef24c73d1aucg_7_167_936_694_269_0.jpg)

Figure 8. (a). The overall accuracy of different $K$ on CREMA-D. (b). The overall accuracy of different $K$ on Kinetics Sounds.

To further evaluate the importance of the prime learning window, we use t-SNE [39] to compare the features learned by the video modality using InfoReg on CREMA-D against those learned through Joint training and extended Joint training. As illustrated in Figure 6, InfoReg yields higher-quality features due to sufficient information acquisition in the prime learning window.This aligns with previous studies, which have concluded that the quality of feature learning is highly correlated with the amount of information acquired early in training [1]. Notably, as shown in Figure 6b and Figure 6c, extending the training period does not compensate for the information loss experienced by the video modality during the prime learning window. Our experiments highlight the critical role of the prime learning window in information acquisition.

#### 4.5.The influence of fusion strategies

We evaluate the performance of InfoReg with four different fusion strategies, including Gated [24], SUM, FiLM [33], and Concat. For the Gated, SUM, and FiLM strategies, we measured the performance of each modality individually by zeroing out the features of other modalities. As shown in Table 5, InfoReg demonstrates consistently robust performance across various fusion strategies. This highlights the adaptability and scalability of InfoReg.

### 4.6. Extension to more complex settings

To validate that our method continues to perform well in more complex settings, we conducted experiments on CMU-MOSI using a Transformer architecture following [29]. As shown in Table 6, InfoReg continues to perform effectively in these more challenging scenarios, outperforming other related methods. This demonstrates the good scal-ability of InfoReg, highlighting its ability to extend to more complex transformer-based architectures and scenarios involving more than two modalities. Further experiments exploring dominant modalities and scenarios requiring inter-modality cooperation are provided in Appendix D.

### 4.7. Hyperparameter sensitivity analysis

The hyperparameter $\beta$ , controlling the regulation term’s strength, is selected from $\{ {0.1},{0.3},{0.5},{0.7},{0.9},{1.0}\}$ . A larger $\beta$ increases the regulation degree of the information-sufficient modalities. Our results (Figure 7) show that $\beta  = {0.9}$ provided the best accuracy. Similarly, the hyper-parameter $K$ , which serves as the threshold for determining whether the model is within the prime learning window, is chosen from $\{ {0.01},{0.02},{0.04},{0.06},{0.08},{0.1}\}$ . Figure 8 indicates that $K = {0.04}$ yields the best performance.

## 5. Conclusion

In this paper, we identify that there is a prime learning window in multimodal learning, and one modality's information acquisition can be suppressed by others during this stage. Then, we propose the Information Acquisition Regulation algorithm. It aims to address the imbalance in information acquisition across modalities by regulating the acquisition rates of information-sufficient modalities during the prime learning window. Our method promotes a more balanced learning process, accordingly enhancing model performance. Experiments across multiple datasets show that our method alleviates imbalanced multimodal learning and then achieves superior performance.

## 6. Acknowledgment

This work was supported through the National Natural Science Foundation of China (Grant No.62106272) and benefited from the research grant program of the CCF-Zhipu.AI Large Model Innovation Fund. References

[1] Alessandro Achille and Stefano Soatto. Emergence of invariance and disentanglement in deep representations. Journal of Machine Learning Research, 19(50):1-34, 2018. 2, 3, 8

[2] Alessandro Achille, Matteo Rovere, and Stefano Soatto. Critical learning periods in deep networks. In International Conference on Learning Representations, 2018. 1, 2, 3

[3] Armen Aghajanyan, Lili Yu, Alexis Conneau, Wei-Ning Hsu, Karen Hambardzumyan, Susan Zhang, Stephen Roller, Naman Goyal, Omer Levy, and Luke Zettlemoyer. Scaling laws for generative mixed-modal language models. In International Conference on Machine Learning, pages 265-279. PMLR, 2023. 2

[4] Shun-ichi Amari and Hiroshi Nagaoka. Methods of information geometry. American Mathematical Soc., 2000. 3

[5] Relja Arandjelovic and Andrew Zisserman. Look, listen and learn. In Proceedings of the IEEE international conference on computer vision, pages 609-617, 2017. 6

[6] Gustavo Arriaga. Of Mice, Birds, and Men: The Mouse Ultrasonic Song System and Vocal Behavior. PhD thesis, Duke University, 2011. 3

[7] Houwei Cao, David G Cooper, Michael K Keutmann, Ruben C Gur, Ani Nenkova, and Ragini Verma. Crema-d: Crowd-sourced emotional multimodal actors dataset. IEEE transactions on affective computing, 5(4):377-390, 2014. 1, 6

[8] Chenzhuang Du, Tingle Li, Yichen Liu, Zixin Wen, Tianyu Hua, Yue Wang, and Hang Zhao. Improving multimodal learning with uni-modal teachers. arXiv preprint arXiv:2106.11059, 2021. 2

[9] Yunfeng Fan, Wenchao Xu, Haozhao Wang, Junxiao Wang, and Song Guo. Pmr: Prototypical modal rebalance for multimodal learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20029- 20038, 2023. 6, 7, 8

[10] Ronald Aylmer Fisher. Theory of statistical estimation. In Mathematical proceedings of the Cambridge philosophical society, pages 700-725. Cambridge University Press, 1925. 1,2,3

[11] Jonathan Frankle, David J Schwab, and Ari S Morcos. The early phase of neural network training. arXiv preprint arXiv:2002.10365, 2020. 3

[12] Ruohan Gao, Tae-Hyun Oh, Kristen Grauman, and Lorenzo Torresani. Listen to look: Action recognition by previewing audio. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10457-10467, 2020. 1

[13] Aditya Sharad Golatkar, Alessandro Achille, and Stefano Soatto. Time matters in regularizing deep networks: Weight decay and data augmentation affect early learning dynamics, matter little near convergence. Advances in Neural Information Processing Systems, 32, 2019. 3

[14] Tal Hassner, Yossi Itcher, and Orit Kliper-Gross. Violent flows: Real-time detection of violent crowd behavior. In 2012 IEEE computer society conference on computer vision and pattern recognition workshops, pages 1-6. IEEE, 2012. 3

[15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770-778, 2016. 6

[16] Takao K Hensch. Critical period regulation. Annu. Rev. Neu-rosci., 27(1):549-579, 2004. 1, 3

[17] Yu Huang, Junyang Lin, Chang Zhou, Hongxia Yang, and Longbo Huang. Modality competition: What makes joint training of multi-modal network fail in deep learning?(provably). In International conference on machine learning, pages 9226-9259. PMLR, 2022. 2

[18] Aya Abdelsalam Ismail, Mahmudul Hasan, and Faisal Ish-tiaq. Improving multimodal accuracy through modality pretraining and attention. arXiv preprint arXiv:2011.06102, 2020. 2

[19] Stanisfaw Jastrzebski, Zachary Kenton, Nicolas Ballas, Asja Fischer, Yoshua Bengio, and Amos Storkey. On the relation between the sharpest directions of dnn loss and the sgd step length. arXiv preprint arXiv:1807.05031, 2018. 1, 3

[20] Stanislaw Jastrzebski, Maciej Szymczak, Stanislav Fort, De-vansh Arpit, Jacek Tabor, Kyunghyun Cho, and Krzysztof Geras. The break-even point on optimization trajectories of deep neural networks. arXiv preprint arXiv:2002.09572, 2020. 3

[21] Stanislaw Jastrzebski, Devansh Arpit, Oliver Astrand, Gian-carlo B Kerg, Huan Wang, Caiming Xiong, Richard Socher, Kyunghyun Cho, and Krzysztof J Geras. Catastrophic fisher explosion: Early phase fisher matrix impacts generalization. In International Conference on Machine Learning, pages 4772-4784. PMLR, 2021. 3

[22] Eric R Kandel, James H Schwartz, Thomas M Jessell, Steven Siegelbaum, A James Hudspeth, Sarah Mack, et al. Principles of neural science. McGraw-hill New York, 2000. 1, 3

[23] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017. 6

[24] Douwe Kiela, Edouard Grave, Armand Joulin, and Tomas Mikolov. Efficient large-scale multi-modal classification. In Proceedings of the AAAI conference on artificial intelligence, 2018. 8

[25] Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringshia, and Davide Testuggine. The hateful memes challenge: Detecting hate speech in multimodal memes. Advances in neural information processing systems, 33:2611-2624, 2020. 3

[26] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521-3526, 2017. 3

[27] Aitor Lewkowycz, Yasaman Bahri, Ethan Dyer, Jascha Sohl-Dickstein, and Guy Gur-Ari. The large learning rate phase of deep learning: the catapult mechanism. arXiv preprint arXiv:2003.02218, 2020. 3

[28] Hong Li, Xingyu Li, Pengbo Hu, Yinuo Lei, Chunxiao Li, and Yi Zhou. Boosting multi-modal model performance with adaptive gradient modulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages ${22214} - {22224},{2023.2},6,7,8$

[29] Paul Pu Liang, Yiwei Lyu, Xiang Fan, Zetian Wu, Yun Cheng, Jason Wu, Leslie Chen, Peter Wu, Michelle A Lee, Yuke Zhu, et al. Multibench: Multiscale benchmarks for multimodal representation learning. Advances in neural information processing systems, 2021(DB1):1, 2021. 6, 8

[30] Paul Pu Liang, Amir Zadeh, and Louis-Philippe Morency. Foundations and trends in multimodal machine learning: Principles, challenges, and open questions. arXiv preprint arXiv:2209.03430, 2022. 2

[31] James Martens. New insights and perspectives on the natural gradient method. Journal of Machine Learning Research, 21 (146):1-76, 2020. 3

[32] Xiaokang Peng, Yake Wei, Andong Deng, Dong Wang, and Di Hu. Balanced multimodal learning via on-the-fly gradient modulation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8238-8247, 2022. 2, 4, 6, 7, 8

[33] Ethan Perez, Florian Strub, Harm De Vries, Vincent Du-moulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence, 2018. 8

[34] Ravid Shwartz-Ziv and Naftali Tishby. Opening the black box of deep neural networks via information. arXiv preprint arXiv:1703.00810, 2017. 3

[35] Ya Sun, Sijie Mai, and Haifeng Hu. Learning to balance the learning rates between various modalities via adaptive tracking factor. IEEE Signal Processing Letters, 28:1650- 1654, 2021. 2

[36] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 2

[37] LCM The, Loïc Barrault, Paul-Ambroise Duquenne, Maha Elbayad, Artyom Kozhevnikov, Belen Alastruey, Pierre Andrews, Mariano Coria, Guillaume Couairon, Marta R Costa-jussà, et al. Large concept models: Language modeling in a sentence representation space. arXiv preprint arXiv:2412.08821, 2024. 2

[38] Valentin Thomas, Fabian Pedregosa, Bart Merriënboer, Pierre-Antoine Manzagol, Yoshua Bengio, and Nicolas Le Roux. On the interplay between noise and curvature and its effect on optimization and generalization. In International Conference on Artificial Intelligence and Statistics, pages 3503-3513. PMLR, 2020. 3

[39] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9 (11), 2008. 7, 8, 2

[40] Dong Wang, Di Hu, Xingjian Li, and Dejing Dou. Temporal relational modeling with self-supervision for action segmentation. In Proceedings of the AAAI conference on artificial intelligence, pages 2729-2737, 2021. 1

[41] Weiyao Wang, Du Tran, and Matt Feiszli. What makes training multi-modal classification networks hard? In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12695-12705, 2020. 2, 6

[42] Yake Wei, Weiran Shen, and Di Hu. Mmpareto: Innocent uni-modal assistance for enhanced multi-modal learning. 2, 6

[43] Yake Wei, Siwei Li, Ruoxuan Feng, and Di Hu. Diagnosing and re-learning for balanced multimodal learning. arXiv preprint arXiv:2407.09705, 2024. 2

[44] Torsten N Wiesel. Postnatal development of the visual cortex and the influence of environment. Nature, 299(5884):583- 591, 1982. 1

[45] Torsten N Wiesel and David H Hubel. Single-cell responses in striate cortex of kittens deprived of vision in one eye. Journal of neurophysiology, 26(6):1003-1017, 1963. 3

[46] Nan Wu, Stanislaw Jastrzebski, Kyunghyun Cho, and Krzysztof J Geras. Characterizing and overcoming the greedy nature of learning in multi-modal deep neural networks. In International Conference on Machine Learning, pages 24043-24055. PMLR, 2022. 6

[47] Shaoxuan Xu, Menglu Cui, Chengxiang Huang, Hongfa Wang, et al. Balancebenchmark: A survey for imbalanced learning. arXiv preprint arXiv:2502.10816, 2025. 2

[48] Gang Yan, Hao Wang, and Jian Li. Critical learning periods in federated learning. arXiv preprint arXiv:2109.05613, 2021. 3

[49] Gang Yan, Hao Wang, Xu Yuan, and Jian Li. Criticalfl: A critical learning periods augmented client selection framework for efficient federated learning. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 2898-2907, 2023. 3

[50] Amir Zadeh, Rowan Zellers, Eli Pincus, and Louis-Philippe Morency. Mosi: Multimodal corpus of sentiment intensity and subjectivity analysis in online opinion videos. arxiv 2016. arXiv preprint arXiv:1606.06259. 6

# Adaptive Unimodal Regulation for Balanced Multimodal Information Acquisition

Supplementary Material

## A. Orthogonality proof

This proof supports the analysis presented in Section 3.3.2, specifically Equation 14, where the regulation term ${P}_{m;b}^{t}$ involves gradients ${\mathbf{g}}_{k}^{t}$ from multiple batches. The analysis assumes that, due to the high dimensionality of the space, the gradients ${\mathbf{g}}_{k}^{t}$ from different batches are nearly orthogonal. Here, we formally prove this assumption by showing that random vectors sampled from the surface of a high-dimensional hypersphere are nearly orthogonal with high probability.

Lemma 1. In high-dimensional spaces, let ${\mathbf{g}}_{z}^{t},{\mathbf{g}}_{k}^{t} \in  {\mathbb{R}}^{n}$ be two random vectors uniformly sampled from the surface of an $n$ -dimensional hypersphere with magnitudes $\begin{Vmatrix}{\mathbf{g}}_{z}^{t}\end{Vmatrix} =$ $a$ and $\begin{Vmatrix}{\mathbf{g}}_{k}^{t}\end{Vmatrix} = b$ . As $n \rightarrow  \infty$ , these vectors are nearly orthogonal with high probability. Specifically, their dot product satisfies:

$$
{\mathbf{g}}_{z}^{t} \cdot  {\mathbf{g}}_{k}^{t} = {ab}\cos \theta  \approx  0. \tag{17}
$$

Proof of Lemma 1. Let ${\mathbf{g}}_{z}^{t}$ and ${\mathbf{g}}_{k}^{t}$ be two random vectors in ${\mathbb{R}}^{n}$ with magnitudes $\begin{Vmatrix}{\mathbf{g}}_{z}^{t}\end{Vmatrix} = a$ and $\begin{Vmatrix}{\mathbf{g}}_{k}^{t}\end{Vmatrix} = b$ . The dot product is given by:

$$
{\mathbf{g}}_{z}^{t} \cdot  {\mathbf{g}}_{k}^{t} = {ab}\cos \theta . \tag{18}
$$

To analyze the distribution of the angle $\theta$ in high-dimensional space, we consider the geometry of the $n$ - dimensional unit hypersphere. Any vector $\mathbf{x} \in  {\mathbb{R}}^{n}$ with unit norm, i.e., $\parallel \mathbf{x}{\parallel }_{2} = 1$ , lies on the surface of the unit hypersphere. It can be parameterized in spherical coordinates as:

$$
\mathbf{x} = \left( {{x}_{1},{x}_{2},\ldots ,{x}_{n}}\right) ,\;\text{ where }{x}_{i} \in  \mathbb{R},\mathop{\sum }\limits_{{i = 1}}^{n}{x}_{i}^{2} = 1. \tag{19}
$$

The components of $\mathbf{x}$ in spherical coordinates are:

$$
{x}_{1} = \cos {\phi }_{1},
$$

$$
{x}_{2} = \sin {\phi }_{1}\cos {\phi }_{2},
$$

$$
{x}_{3} = \sin {\phi }_{1}\sin {\phi }_{2}\cos {\phi }_{3},
$$

$$
\vdots  \tag{20}
$$

$$
{x}_{n} = \mathop{\prod }\limits_{{i = 1}}^{{n - 1}}\sin {\phi }_{i}
$$

where ${\phi }_{1},{\phi }_{2},\ldots ,{\phi }_{n - 2} \in  \left\lbrack  {0,\pi }\right\rbrack$ , and ${\phi }_{n - 1} \in  \left\lbrack  {0,{2\pi }}\right\rbrack$ . The surface element of the hypersphere is:

$$
{dS} = {\left( \sin {\phi }_{1}\right) }^{n - 2}{\left( \sin {\phi }_{2}\right) }^{n - 3}\cdots \sin {\phi }_{n - 2}d{\phi }_{1}d{\phi }_{2}\cdots d{\phi }_{n - 1}.
$$

(21)

Without loss of generality, let one vector ${\mathbf{g}}_{z}^{t}$ be fixed along the ${x}_{1}$ -axis, ${\mathbf{g}}_{z}^{t} = \left( {a,0,\ldots ,0}\right)$ . The second vector ${\mathbf{g}}_{k}^{t}$ can be parameterized using spherical coordinates. The angle $\theta$ between ${\mathbf{g}}_{z}^{t}$ and ${\mathbf{g}}_{k}^{t}$ is the same as ${\phi }_{1}$ , the first coordinate angle, so:

$$
\cos {\phi }_{1} = \cos \theta  \tag{22}
$$

The relevant term in the hypersphere surface element is:

$$
{p}_{n}\left( {\phi }_{1}\right)  \propto  {\left( \sin {\phi }_{1}\right) }^{n - 2}. \tag{23}
$$

This shows that the probability density of ${\phi }_{1}$ (or $\theta$ ) depends on the sine function raised to the power of(n - 2). For large $n,{\left( \sin {\phi }_{1}\right) }^{n - 2}$ is sharply concentrated around ${\phi }_{1} =$ $\pi /2$ because $\sin {\phi }_{1}$ reaches its maximum at $\pi /2$ . As $n \rightarrow$ $\infty$ , this concentration becomes stronger, leading to ${\phi }_{1} \approx  \frac{\pi }{2}$ with high probability. Since ${\phi }_{1} \approx  \pi /2$ , we have:

$$
\cos {\phi }_{1} = \cos \theta  \approx  0. \tag{24}
$$

Thus, in high-dimensional spaces, the angle $\theta$ between two random vectors concentrates around $\pi /2$ , leading to:

$$
{\mathbf{g}}_{z}^{t} \cdot  {\mathbf{g}}_{k}^{t} = {ab}\cos \theta  \approx  0. \tag{25}
$$

This demonstrates that the vectors are nearly orthogonal as $n \rightarrow  \infty$ .

## B. Gradient norm analysis

This section aims to demonstrate that the regulation term ${P}_{m;b}^{t}$ , introduced to regulate the information-sufficient modalities during the prime learning window, does not hinder the convergence of the optimization process. Specifically, we analyze the gradient norm and show that, under proper parameter settings, the convergence rate remains consistent with that of the original optimization objective without the regulation term.

Lemma 2. At training epoch $t$ and batch $b$ , consider the optimization objective:

$$
\mathcal{L}\left( {w}_{m;b}^{t}\right)  = {\mathcal{L}}_{\text{joint }}\left( {w}_{m;b}^{t}\right)  + {P}_{m;b}^{t}, \tag{26}
$$

where ${\mathcal{L}}_{\text{joint }}\left( {w}_{m;b}^{t}\right)$ is the multimodal joint loss function, and the regulation term ${P}_{m;b}^{t}$ is defined as:

$$
{P}_{m;b}^{t} = \frac{\alpha {\eta }^{2}}{2}\mathop{\sum }\limits_{{k = 0}}^{b}{\begin{Vmatrix}{g}_{k}^{t}\end{Vmatrix}}^{2}. \tag{27}
$$

Here, $\alpha  > 0$ is the regularization coefficient, $\eta  > 0$ is the learning rate, and ${g}_{k}^{t}$ denotes the gradient of batch $k$ at epoch $t$ . If $\alpha$ and $\eta$ are sufficiently small, the convergence rate remains of the same order as without the regulation term.

Proof of Lemma 2. During the training, the weight update rule is given by:

$$
{w}_{m;b + 1}^{t} = {w}_{m;b}^{t} - \eta \nabla \mathcal{L}\left( {w}_{m;b}^{t}\right) , \tag{28}
$$

where:

$$
\nabla \mathcal{L}\left( {w}_{m;b}^{t}\right)  = \nabla {\mathcal{L}}_{\text{joint }}\left( {w}_{m;b}^{t}\right)  + \nabla {P}_{m;b}^{t}. \tag{29}
$$

The gradient of the regulation term ${P}_{m;b}^{t}$ is given by:

$$
\nabla {P}_{m;b}^{t} = \alpha {\eta }^{2}\mathop{\sum }\limits_{{k = 0}}^{b}{g}_{k}^{t}. \tag{30}
$$

Assuming that $\mathcal{L}\left( w\right)$ is $L$ -Lipschitz smooth, we have:

$$
\mathcal{L}\left( {w}_{m;b + 1}^{t}\right)  \leq  \mathcal{L}\left( {w}_{m;b}^{t}\right)  + \nabla \mathcal{L}{\left( {w}_{m;b}^{t}\right) }^{T}\left( {{w}_{m;b + 1}^{t} - {w}_{m;b}^{t}}\right)
$$

$$
+ \frac{L}{2}{\begin{Vmatrix}{w}_{m;b + 1}^{t} - {w}_{m;b}^{t}\end{Vmatrix}}^{2}.
$$

(31)

Substituting ${w}_{m;b + 1}^{t} - {w}_{m;b}^{t} =  - \eta \nabla \mathcal{L}\left( {w}_{m;b}^{t}\right)$ , we obtain:

$$
\mathcal{L}\left( {w}_{m;b + 1}^{t}\right)  \leq  \mathcal{L}\left( {w}_{m;b}^{t}\right)  - \eta {\begin{Vmatrix}\nabla \mathcal{L}\left( {w}_{m;b}^{t}\right) \end{Vmatrix}}^{2} + \frac{L{\eta }^{2}}{2}{\begin{Vmatrix}\nabla \mathcal{L}\left( {w}_{m;b}^{t}\right) \end{Vmatrix}}^{2}
$$

(32)

The gradient norm is expressed as:

$$
{\begin{Vmatrix}\nabla \mathcal{L}\left( {w}_{m;b}^{t}\right) \end{Vmatrix}}^{2} = {\begin{Vmatrix}\nabla {\mathcal{L}}_{\text{joint }}\left( {w}_{m;b}^{t}\right) \end{Vmatrix}}^{2}
$$

$$
+ 2\nabla {P}_{m;b}^{t} \cdot  \nabla {\mathcal{L}}_{\text{joint }}\left( {w}_{m;b}^{t}\right)  \tag{33}
$$

$$
+ {\begin{Vmatrix}\nabla {P}_{m;b}^{t}\end{Vmatrix}}^{2}\text{.}
$$

Due to the high dimensionality of the space, as demonstrated in Section A, the regulation term gradient $\nabla {P}_{m;b}^{t}$ and the joint loss gradient $\nabla {\mathcal{L}}_{\text{joint }}\left( {w}_{m;b}^{t}\right)$ are nearly orthogonal. As a result, their dot product can be approximated as:

$$
\nabla {P}_{m;b}^{t} \cdot  \nabla {\mathcal{L}}_{\text{joint }}\left( {w}_{m;b}^{t}\right)  \approx  0. \tag{34}
$$

The gradient of the regulation term is bounded as:

$$
\begin{Vmatrix}{\nabla {P}_{m;b}^{t}}\end{Vmatrix} = \alpha {\eta }^{2}\begin{Vmatrix}{\mathop{\sum }\limits_{{k = 0}}^{b}{g}_{k}^{t}}\end{Vmatrix} \leq  \alpha {\eta }^{2}{bG}, \tag{35}
$$

where $G$ is the upper bound of the gradient norm $\begin{Vmatrix}{g}_{k}^{t}\end{Vmatrix}$ . Thus, the term satisfies:

$$
{\begin{Vmatrix}\nabla \mathcal{L}\left( {w}_{m;b}^{t}\right) \end{Vmatrix}}^{2} \leq  {\begin{Vmatrix}\nabla {\mathcal{L}}_{\text{joint }}\left( {w}_{m;b}^{t}\right) \end{Vmatrix}}^{2} + {\alpha }^{2}{\eta }^{4}{b}^{2}{G}^{2}. \tag{36}
$$

For sufficiently small $\alpha$ and $\eta$ , the additional term ${\alpha }^{2}{\eta }^{4}{b}^{2}{G}^{2}$ becomes negligible. Therefore, the convergence rate remains of the same order as without ${P}_{m;b}^{t}$ .

## C. Supplementary t-SNE analysis

![bo_d3s8tajef24c73d1aucg_11_929_626_702_545_0.jpg](images/bo_d3s8tajef24c73d1aucg_11_929_626_702_545_0.jpg)

Figure 9. The representations of the video modality on CREMA-D by t-SNE [39] across different methods are shown. InfoReg* and Joint training* denote InfoReg and Joint training with unimodal loss respectively. "Extended Joint training*" denotes Joint training* that is extended to 100 epochs.

To provide a more comprehensive evaluation of the proposed InfoReg method, we extend our analysis by incorporating t-SNE visualizations of video modality representations for InfoReg* and Joint training* on the CREMA-D dataset. Here, InfoReg* denotes InfoReg with unimodal loss, and Joint training* denotes Joint training with unimodal loss. As shown in Figure 9, InfoReg* and Joint training* learn better representations than Joint training. This is because the unimodal loss helps the multimodal model acquire more information. Additionally, the features learned by Joint training* and Extended Joint training* are similar, as shown in Figure $9\mathrm{c}$ and Figure $9\mathrm{\;d}$ . This indicates that extending the training time cannot compensate for the lack of information acquired during the prime learning window. Furthermore, InfoReg* learns better representations than both Joint training* and Extended Joint training*. This demonstrates that, with unimodal loss, our method can still help information-insufficient modalities acquire more information in the prime learning window. As a result, InfoReg* learns better representation.

## D. Supplementary experiments

![bo_d3s8tajef24c73d1aucg_12_252_261_525_224_0.jpg](images/bo_d3s8tajef24c73d1aucg_12_252_261_525_224_0.jpg)

Figure 10. Violence Flow dataset example, showcasing video modality dominance.

<table><tr><td>Dataset</td><td>Violence Flow</td><td>$\mathbf{{HatefulMemes}}$</td></tr><tr><td>Joint training</td><td>89.21</td><td>55.00</td></tr><tr><td>InfoReg</td><td>90.56</td><td>56.20</td></tr></table>

Table 7. Accuracy comparison.

To further evaluate the effectiveness of InfoReg under diverse dataset conditions, we conducted experiments on the Violence Flow [14] and Hateful Memes [25] datasets. These datasets present different challenges: Violence Flow emphasizes anomaly detection, where the video modality quickly becomes dominant, while Hateful Memes requires cooperation between modalities due to its complex multimodal nature.

Figure 10 illustrates the information amount during training on the Violence Flow, where the video modality demonstrates dominance during the prime learning window. InfoReg can identify this dominant modality.

The Hateful Memes dataset requires significant cooperation between modalities. As shown in the Table 7, Despite the increased complexity, InfoReg can still improve the per-foremance of the model.