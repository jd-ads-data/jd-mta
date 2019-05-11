# JD-MTA: Causally Driven Incremental Multi Touch Attribution Using a Recurrent Neural Network
The repository contains sample code in **TensorFlow** for a simulation that implements MTA framework proposed in our paper:
>[Causally Driven Incremental Multi Touch Attribution Using a Recurrent Neural Network](https://arxiv.org/abs/1902.00215)
>Ruihuan Du, Yu Zhong, Harikesh Nair, Bo Cui, Ruyang Shou
>Preprint 2019

### Introduction

The code is in the `python_tf/` folder. The simulation dataset is in the `data/` folder. The different python scripts in the repository are below:

| Script Path | Function |
|--- | --- |
| `python_tf/consigures.py` | Configuration file. |
| `python_tf/input.py` | Processes tfrecord files by dataset. |
| `python_tf/model.py` | Expresses the rnn model. |
| `python_tf/shapley_value.py` | Computes the Shapley Value. |
| `python_tf/tfrecord_helper.py` | Defines methods that are used in generating the simulated data. |
| `generate_simulation_data.py` | Generates the simulated data. |
| `main.py` | An example which runs all the scripts above. |

### Simulation Details

User behavior is simulated at fixed parameters values under a model with sequential dependence. The following table lists the parameters used and outlines the data generating process. The code trains the proposed RNN on these data, and computes the corresponding Shapley Values described in the paper.

| Parameter | Value Used in Code | Explanation |
| :--- | :---: | :--- |
| NUM_BRANDS | 2 | Number of brands. |
| NUM_DAYS | 5 | Number of days for the sequence of relevant ad exposures. |
| NUM_POS | 2 | Number of ad-positions. |
| $a$ | 2 | Average number of ad-exposures a user sees at one ad-position for one brand in one day. |
| N | 10000 | Number of simulated users in the simulation data. |

Let $i = 1..N$ be the index of the user, $b=1..B$ be the index of the brand and $t=1..T$ be the index of the day. We denote by $x_{i,t,b}$ the ad-exposure vector for user $i$ and brand $b$ in day $t$, $y_{i,t,b}$ the binary value that denoted whether there is a conversion for user $i$ and brand $b$ at day $t$, $d_i$ the user profile vector (i.e., user time-invariant attributes) and $r_b$ the brand profile for brand $b$ (i.e., brand time-invariant attributes). The data are simulated as follows:

$$ h_{i, 0, b} \sim N(0, 1) $$
$$ d_{i} \sim U(0, 1) $$
$$ r_{b} \sim LogNormal(0, 0.25) $$
$$ x_{i,0,b} = \vec{0} $$
$$ x_{i,t,b} \sim Poisson(a) $$
$$ h_{i, t, b} = 0.5 * sigmoid(1.0 + \alpha_1 x_{i, t-1} + \alpha_2 ) + 0.5 * h_{i, t-1, b}, \alpha_1 \sim N(0,1) $$
$$ u_{i, t, b} = 0.4 * d_i + \beta_1 x_{i, t, b} + 0.01 * h_{i, t, b} + \beta_3 r_b, \beta_1 \sim U(0.1, 0.9) $$
$$ p_{i,t,b} =  sigmoid(u_{i,t,b})$$
$$ y_{i,t,b} \sim Bernoulli(p_{i,t,b} ) $$

### Guide to Run the Example

System requirements:
* TensorFlow 1.11.+
* python3

Type the command to run the example in your terminal: 

```markdown
python3 main.py
```


<script type="text/javascript"
   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
