# GN-RD: Training generative networks using random discriminators
Implementation code for the proposed Generative Adversarial Networks (GANs) using randomly generated discriminator, published in the processing of 2019 Data Science Workshop (DSW). [Link to Paper](https://arxiv.org/pdf/1904.09775.pdf)
# Abstract 

In recent years,  Generative  Adversarial  Networks  (GANs) have drawn a lot of attentions for learning the underlying distribution of data in various applications. Despite their wide applicability, training GANs is notoriously difficult. This difficulty is due to the min-max nature of the resulting optimization problem and the lack of proper tools of solving general (non-convex, non-concave) min-max optimization problems. In this paper, we try to alleviate this problem by proposing a new generative network that relies on the use of random discriminators instead of adversarial design. This design helps us to avoid the min-max formulation and leads to an optimization problem that is stable and could be solved efficiently. The performance of the proposed method is evaluated using handwritten digits (MNIST) and Fashion products (Fashion-MNIST) data sets.While the resulting images are not as sharp as adversarial training, the use of random discriminator leads to a much faster algorithm as compared to the adversarial counterpart. This observation, at the minimum, illustrates the potential of the random discriminator approach for warm-start in training GANs.
# Summary of the proposed algorithm
The proposed algorithm does not require any optimization on the discriminator network and only needs randomly generated discriminator to learn the underlying distribution of the data.
<p align="center">
  <img width="400" height="350" src="https://github.com/babakbarazandeh/GN-RD/blob/master/Algorithm.jpg">
</p>
 
# Reuslts 
Following figure shows the performance of the proposed GN-RD algorithm for learning generative network to create samples from MNIST dataset. The generated samples shows that GN-RD quickly converges and generates promising samples.

<p align="center">
  <img width="500" height="165" src="https://github.com/babakbarazandeh/GN-RD/blob/master/Results.jpg">
</p> <br/>

# Getting started
After installing Tensorflow, run Main.py

# Citation 
@article{barazandeh2019training,<br/>
  title={Training generative networks using random discriminators},<br/>
  author={Barazandeh, Babak and Razaviyayn, Meisam and Sanjabi, Maziar},<br/>
