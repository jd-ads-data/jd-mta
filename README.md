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
| NUM_DAYS | 5 | Number of days for the sequence of relevant ad-exposures. |
| NUM_POS | 2 | Number of ad-positions. |
| a | 2 | Average number of ad-exposures a user sees at one ad-position for one brand in one day. |
| N | 100000 | Number of simulated users in the simulation data. |

Let i = 1..N index users, $b=1..B$ index brands and $t=1..T$ index days. We denote by x_{i,t,b} a vector that contains the exposure to ads by user i for brand b on day t across the various ad-positions; by r_{b,t} the price indices for brand b on day t; by d_i the user's characterastics (i.e., user time-invariant attributes); i's initial conditions with respect to each brand by h_{i, 0, b}; and by y_{i,t,b} a binary value for whether there is a conversion by user i of brand b on day t.

The data are simulated as follows (a more detailed explain of the process can be found in detailed_explain_of_simulation.pdf):

1. Draw a random scalar to simulate user characterastics, d_i.
2. Draw T-length random vector to simulate the brand price indices r_{b,t} in each time period.
3. Draw a T-length random vector for each i, t, b and ad-position to simulate user i's ad-exposures on day t to brand b's ads,  x_{i,t,b}.
4. Simulate user i's initial conditions with respect to each brand, h_{i, 0, b}.
5. Given h_{i, 0, b} and parameters, forward simulate user i's recurrent term for each b in each t, denoted h_{i, t, b}, as a function x_{i, t-1} and h_{i, t-1, b}. The recurrence generates time-dependence in user behavior.
6. Compute u_{i,t,b}, the "attractiveness" of brand b for user i in time t as a function of x_{i, t, b}, d_i, h_{i, t, b}  and r_{b,t}.
7. Transform u_{i,t,b} via a sigmoid function to obtain the probability of conversion by user i of brand b in time t, p_{i,t,b}.
8. Simulate conversion by user i of brand b in time t y_{i,t,b} by drawing a Bernoulli random variate with mean p_{i,t,b}.

### Guide to Run the Example

System requirements:
* TensorFlow 1.11.+
* python3+

Type the command to run the example in your terminal: 

```markdown
python3 main.py
```
