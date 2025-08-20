Flow Matching Policy Gradients

David McAllister^1 ∗Songwei Ge^1 ∗ Brent Yi^1 ∗ Chung Min Kim^1
Ethan Weber^1 Hongsuk Choi^1 Haiwen Feng^1 ,^2 Angjoo Kanazawa^1

(^1) UC Berkeley (^2) Max Planck Institute for Intelligent Systems
Abstract

Flow-based generative models, including diffusion models, excel at modeling con-
tinuous distributions in high-dimensional spaces. In this work, we introduce Flow
Policy Optimization (FPO), a simple on-policy reinforcement learning algorithm
that brings flow matching into the policy gradient framework. FPO casts policy
optimization as maximizing an advantage-weighted ratio computed from the con-
ditional flow matching loss, in a manner compatible with the popular PPO-clip
framework. It sidesteps the need for exact likelihood computation while preserving
the generative capabilities of flow-based models. Unlike prior approaches for
diffusion-based reinforcement learning that bind training to a specific sampling
method, FPO is agnostic to the choice of diffusion or flow integration at both train-
ing and inference time. We show that FPO can train diffusion-style policies from
scratch in a variety of continuous control tasks. We find that flow-based models
can capture multimodal action distributions and achieve higher performance than
Gaussian policies, particularly in under-conditioned settings. For an overview of
FPO’s key ideas, see our accompanying blog post: flowreinforce.github.io

1 Introduction

Flow-based generative models—particularly diffusion models—have emerged as powerful tools
for generative modeling across the domains of images [ 1 – 3 ], videos [ 4 – 6 ], speech [ 7 ], audio [ 8 ],
robotics [ 9 ], and molecular dynamics [ 10 ]. In parallel, reinforcement learning (RL) has proven to be
effective for optimizing neural networks with non-differentiable objectives, and is widely used as a
post-training strategy for aligning foundation models with task-specific goals [11, 12].
In this work, we introduce Flow Policy Optimization (FPO), a policy gradient algorithm for optimizing
flow-based generative models. FPO reframes policy optimization as maximizing an advantage-
weighted ratio computed from the conditional flow matching (CFM) objective [ 13 ]. Intuitively,
FPO shapes probability flow to transform Gaussian noise into high-reward actions by reinforcing its
experience using flow matching. The method is simple to implement and can be readily integrated into
standard techniques for stochastic policy optimization. We use a PPO-inspired surrogate objective for
our experiments, which trains stably and serves as a drop-in replacement for Gaussian policies.
FPO offers several key advantages. It sidesteps the complex likelihood calculations typically associ-
ated with flow-based models, instead using the flow matching loss as a surrogate for log-likelihood in
the policy gradient. This aligns the objective directly with increasing the evidence lower bound of
high-reward actions. Unlike previous methods that reframe the denoising process as an MDP, binding
the training to specific sampling methods and extending the credit-assignment horizon, FPO treats
the sampling procedure as a black box during rollouts. This distinction allows for flexible integration
with any sampling approach—whether deterministic or stochastic, first- or higher-order, and with any
number of integration steps during training or inference.

∗Equal contribution.

Preprint.

arXiv:2507.21053v2 [cs.LG] 1 Aug 2025

We theoretically analyze FPO’s correctness and empirically validate its performance across a diverse
set of tasks. These include a GridWorld environment, 10 continuous control tasks from MuJoCo
Playground [ 14 ], and high-dimensional humanoid control—all trained from scratch. FPO demon-
strates robustness across tasks, enabling effective training of flow-based policies in high-dimensional
domains. We probe flow policies learned in the toy GridWorld environment and find that on states
with multiple possible optimal actions, it learns multimodal action distributions, unlike Gaussian
policies. On humanoid control tasks, we show that the expressivity of flow matching enables single-
stage training of under-conditioned control policies, where only root-level commands are provided.
In contrast, standard Gaussian policies struggle to learn viable walking behaviors in such cases. This
highlights the practical benefits of the more powerful distribution modeling enabled by FPO. Finally,
we discuss limitations and future work.
2 Related Work

Policy Gradients.We study on-policy reinforcement learning, where a parameterized policy is
optimized to maximize cumulative reward in a provided environment. This is commonly solved with
policy gradient techniques, which bypass the need for differentiable environment rewards by weight-
ing action log-probabilities with observed rewards or advantages [ 15 – 23 ]. Policy gradient methods
are central in learning policies for general continuous control tasks [ 24 , 25 ], robot locomotion [ 26 – 29 ]
and manipulation [ 30 – 33 ]. They have also been adopted increasingly for searching through and
refining prior distributions in pretrained generative models. This has proven effective for alignment
with human preferences [34, 35] and improving reasoning using verifiable rewards [36, 37].

In this work, we propose a simple algorithm for training flow-based generative policies, such as
diffusion models, under the policy gradient framework. By leveraging recent insights from flow
matching [ 13 ], we train policies that can represent richer distributions than the diagonal Gaussians
that are most frequently used for reinforcement learning for continuous control [ 26 – 29 , 32 , 33 ], while
remaining compatible with standard actor-critic training techniques.

Diffusion Models. Diffusion models are powerful tools for modeling complex continuous dis-
tributions and have achieved remarkable success across a wide range of domains. These models
have become the predominant approach for generating images [ 38 – 41 ], videos [ 42 – 44 , 4 ], au-
dio [ 7 , 45 , 46 , 8 ], and more recently, robot actions [ 9 , 47 , 48 ]. In these applications, diffusion models
aim to sample from a data distribution of interest, whether scraped from the internet or collected
through human teleoperation.

Flow matching [ 13 ] simplifies and generalizes the diffusion model framework. It learns a vector
field that transports samples from a tractable prior distribution to the target data distribution. The
conditional flow matching (CFM) objective trains the model to denoise data that has been perturbed
with Gaussian noise. Given dataxand noiseε∈N(0,I), the CFM objective can be expressed as:

LCFM,θ=Eτ,q(x),pτ(xτ|x)∥ˆvθ(xτ,τ)−u(xτ,τ|x)∥^22 , (1)

wherexτ=ατx+στεrepresents the partially noised sample at flow stepτ, an interpolation of noise
and data with a schedule defined by hyperparametersατandστ.ˆvθ(xτ,τ)is the model’s estimate of
the velocity to the original data, andu(xτ,τ|x)is the conditional flowx−ε. The model can also
estimate the denoised samplexor noise componentεas the optimization target instead of velocity.
The learned velocity field is a continuous mapping that transports samples from a simple, tractable
distribution (e.g. Gaussian noise) to the training data distribution through ODE integration.

Optimizing likelihoods directly through flow models is possible, but requires divergence estima-
tion [ 49 ] and is computationally prohibitive. Instead, flow matching optimizes variational lower
bounds of the likelihood with the simple denoising loss above. In this work, we leverage flow
matching directly within the policy gradient formulation. This approach trains diffusion models from
rewards without prohibitively expensive likelihood computations.

Diffusion Policies.Diffusion-based policies have shown promising results in robotics and decision-
making applications [ 50 , 51 , 47 ]. Most existing approaches train these models via behavior
cloning [ 52 , 9 ], where the policy is supervised to imitate expert trajectories without using reward
feedback. Motivated by the strong generative capabilities of diffusion and flow-based models, several
works have explored using reinforcement learning to fine-tune diffusion models, particularly in
domains like text-to-image generation [53–55].

Recent work by Psenka et al. [ 56 ] explores off-policy training of diffusion policies via Q-score
matching. While off-policy reinforcement learning continues to make progress [ 57 , 58 ], on-policy
methods dominate practical applications today. Methods like DDPO [ 54 ], DPPO [ 59 ], and Flow-
GRPO [ 55 ] adopt on-policy policy gradient methods by treating initial noise values as observations
from the environment, framing the denoising process as a Markov decision process, and training each
step as a Gaussian policy using PPO. Our approach differs by directly integrating the conditional
flow matching (CFM) objective into a PPO-like framework, maintaining the structure of the standard
diffusion forward and reverse processes. Since FPO integrates flow matching as its fundamental
primitive, it is agnostic to the choice of sampling method during both training and inference, just like
flow matching for behavior cloning.
3 Flow Matching Policy Gradients

3.1 Policy Gradients and PPO

The goal of reinforcement learning is to learn a policyπθthat maximizes expected return in a provided
environment. At each iteration of online reinforcement learning, the policy is rolled out to collect
batches of observation, action, and reward tuples(ot,at,rt)for each environment timestept. These
rollouts can used in the policy gradient objective [ 15 ] to increase likelihood of actions that result in
higher rewards:

max
θ

Eat∼πθ(at|ot)

h
logπθ(at|ot)Aˆt

i
, (2)

whereAˆtis an advantage estimated from the rollout’s rewardsrtand a learned value function [60].

The vanilla policy gradient is valid only locally around the current policy parameters. Large updates
can lead to policy collapse or unstable learning. To address this, PPO [ 20 ] incorporates a trust region
by clipping the likelihood ratio:

max
θ

Eat∼πθold(at|ot)

h
min



r(θ)Aˆt,clip(r(θ), 1 −εclip,1 +εclip)Aˆt

i
, (3)

whereεclipis a tunable threshold andr(θ)is the ratio between current and old action likelihoods:

r(θ) =

πθ(at|ot)
πold(at|ot)

. (4)

PPO is popular choice for on-policy reinforcement learning because of its stability, simplicity,
and performance. Like the standard policy gradient, however, it requires exact likelihoods for
sampled actions. These quantities are tractable for simple Gaussian or categorical action spaces, but
computationally prohibitive to estimate for flow matching and diffusion models.

3.2 Flow Policy Optimization

We introduce Flow Policy Optimization (FPO), an online reinforcement learning algorithm for
policies represented as flow modelsˆvθ. There are two key differences in practice from Gaussian
PPO. During rollouts, a flow model transforms random noise into actions via a sequence of learned
transformations [ 13 ], enabling much more expressive policies than those used in standard PPO. Also,
to update the policy, the Gaussian likelihoods are replaced with a transformed flow matching loss.

Instead of updating exact likelihoods, we propose a proxyˆrFPOfor the log-likelihood ratio. FPO’s
overall objective is the same as Equation 3, but with the ratio substituted:

max
θ

Eat∼πθ(at|ot)

h
min



rˆFPO(θ)Aˆt,clip(ˆrFPO(θ), 1 −εclip,1 +εclip)Aˆt

i

. (5)

Intuitively, FPO’s goal is to steer the policy’s probability flow toward high-return behavior. Instead of
computing likelihoods, we construct a simple ratio estimate using standard flow matching losses:

ˆrFPO(θ) = exp(LˆCFM,θold(at;ot)−LˆCFM,θ(at;ot)), (6)

which, as we will discuss, can be derived from optimizing the evidence lower bound.

For a given action and observation pair,LˆCFM,θ(at;ot)is an estimate of the per-sample conditional
flow matching lossLCFM,θ(at;ot):

LˆCFM,θ(at;ot) =^1
Nmc

XNmc

i

ℓθ(τi,εi) (7)

ℓθ(τi,εi) =||ˆvθ(aτti,τi;ot)−(at−εi)||^22 (8)
aτti=ατiat+στiεi, (9)

where we denote flow timesteps withτand environment timesteps witht. We include both timesteps
inaτt, which represents an action at rollout timetwith noise levelτfollowing Equation 1. We use
the sameεi∼N(0,I)andτi∈[0,1]samples betweenLˆCFM,θoldandLˆCFM,θ.

Properties.FPO’s ratio estimate in Equation 6 serves as a drop-in replacement for the PPO likelihood
ratio. FPO therefore inherits compatibility with advantage estimation methods like GAE [ 60 ] and
GRPO [ 23 ]. Without loss of generality, it is also compatible with flow and diffusion implementations
based on estimating noiseε[ 38 ] or clean actionat[ 1 ], which can be reweighted for mathematical
equivalence toLθ,CFM[61, 13]. We leverage this property in our FPO ratio derivation below.

3.3 FPO Surrogate Objective

Exact likelihood is computationally expensive even to estimate in flow-based models. Instead, it is
common to optimize the evidence lower bound (ELBO) as a proxy for log-likelihood:

ELBOθ(at|ot) = logπθ(at|ot)−DKLθ , (10)

whereDKLθ is the KL gap between the ELBO and true log-likelihood andπθis the distribution
captured by sampling from the flow model. Both flow matching and diffusion models optimize the
ELBO using a conditional flow matching loss, a simple MSE denoising objective [ 62 , 13 ]. The
FPO ratio (Equation 11) leverages the fact that flow models can be trained via ELBO objectives.
Specifically, we compute the ratio of ELBOs under the current and old policies:

rFPO(θ) =

exp(ELBOθ(at|ot))
exp(ELBOθold(at|ot))

. (11)

Decomposing this ratio reveals a scaled variant of the true likelihood ratio (Equation 4):

rFPO(θ) =

πθ(at|ot)
πθold(at|ot)
| {z }
Likelihood

exp(DKLθold)
exp(DθKL)
| {z }
Inv. KL Gap

. (12)

Here, the ratio decomposes into the standard likelihood ratio and an inverse correction term involving
the KL gap. Maximizing this ratio therefore increases the modeled likelihood while reducing the KL
gap—both of which are beneficial for policy optimization. The former encourages the policy to favor
actions with positive advantage, while the latter tightens the approximation to the true log-likelihood.

3.4 Estimating the FPO Ratio with Flow Matching

We estimate the FPO ratio using the flow matching objective directly, which follows from the
relationship between the weighted denoising lossLwθand the ELBO established by Kingma and
Gao [ 63 ]. Lwθ is a more general form of the flow matching and denoising diffusion loss that
parameterizes the model as predictingˆεθ, an estimate of the true noiseεpresent in the model input.

The weighted denoising lossLwθ for a clean actionattakes the form:

Lwθ(at) =

1
2

Eτ∼U(0,1),ε∼N(0,I)



w(λτ)·


−

dλ
dτ



·∥ˆεθ(aτt;λτ)−ε∥^22


, (13)

wherewis a choice of weighting andλτrepresents the log-SNR at noise levelτ. We estimate this
value with Monte Carlo draws of timestepτand noiseε:

ℓwθ(τ,ε) =

1
2

w(λτ)·


−

dλ
dτ



·∥ˆεθ(aτt;λτ)−ε∥^22. (14)

The choice of weightingwincorporates the conditional flow matching loss and standard diffusion
loss as specific cases of a more general familyLwθ(at).

We focus here on the constant weight casew(λτ) = 1(diffusion schedule), which yields the simplest
theoretical connection. Similar results hold for many popular schedules, including optimal transport
and variance preserving schedules [13]. Please see the supplementary material for details.

For the diffusion schedule, [63] proves that:

Lwθ(at) =−ELBOθ(at) +c, (15)

wherecis a constant w.r.tθ. Geometrically, minimizingLwθ(at)points the flow more towardat.
MinimizingLwθalso maximizes the ELBO (Eq. 10) and thus the likelihood ofat, so flowing toward a
specific action makes it more likely. This intuition aligns naturally with the policy gradient objective:
we want to increase the probability of high-advantage actions. By redirecting flow toward such
actions (i.e., minimizing their diffusion loss), we make them more likely under the learned policy.

Using this relationship, we express the FPO ratio (Eq. 11) in terms of the flow matching objective:

rFPOθ =

exp(ELBOθ(at|ot))
exp(ELBOθold(at|ot))

= exp(Lwθold(at)−Lwθ(at)), (16)

whereLwθ, as per Equation 7, can be estimated by averaging overNmcdraws of (τ,ε). We find the
sample countNmcto be a useful hyperparameter for controlling learning efficiency. This estimator
recovers the exact FPO ratio in the limit, although we use only a few draws in practice.

One possible concern with smallerNmcvalues is bias. A ratio estimated from only one (τ,ε) pair,

ˆrFPOθ (τ,ε) = exp(ℓwθold(τ,ε)−ℓwθ(τ,ε)), (17)

is in expectation only an upper-bound of the true ratio. This can be shown by Jensen’s inequality:

Eτ,ε[ˆrθFPO(τ,ε)]≥rFPOθ. (18)

To understand the upward bias, we can use the log-derivative trick to decompose the FPO gradient:

∇θˆrFPOθ (τ,ε) =−rˆFPOθ (τ,ε)∇θℓwθ(τ,ε). (19)

Since the gradient operator commutes with expectation, the gradient term on the right side is unbiased:

Eτ,ε[−∇θℓwθ(τ,ε)] =−∇θLwθ(at) =∇θELBOθ(at). (20)

In other words, gradient estimates are directionally unbiased even with worst-case overestimation of
ratios. Our experiments are consistent with this result: while additional samples help, we observe
empirically in Section 4.2 that FPO can be trained to outperform Gaussian PPO even withNmc= 1.

Algorithm 1 details FPO’s practical implementation using this mathematical framework.

3.5 Denoising MDP Comparison

Existing algorithms [ 54 , 59 , 55 ] for on-policy reinforcement learning with diffusion models refor-
mulate the denoising process itself as a Markov Decision Process (MDP). These approaches bypass
flow model likelihoods by instead treating every step in the sampling chain as its own action, each
parameterized as a Gaussian policy step. This has a few limitations that FPO addresses.

First, denoising MDPs multiply the horizon length by the number of denoising steps (typically 10-50),
which increases the difficulty of credit assignment. Second, these MDPs do not consider the initial
noise sample during likelihood computation. Instead, these noise values are treated as observations
from the environment [ 59 ]—this significantly increases the dimensionality of the learning problem.
Finally, denoising MDP methods are limited to stochastic sampling procedures by construction.
Instead, since FPO employs flow matching, it inherits the flexibility of sampler choices from standard
flow/diffusion models. These include fast deterministic samplers, higher-order integration, and
choosing any number of sampling steps. Perhaps most importantly, FPO is simpler to implement
because it does not require a custom sampler or the notion of extra environment steps.

Algorithm 1Flow Policy Optimization (FPO)

Require:Policy parametersθ, value function parametersφ, clip parameterε, MC samplesNmc
1:whilenot convergeddo
2: Collect trajectories using any flow model sampler and compute advantagesAˆt
3: For each action, storeNmctimestep-noise pairs{(τi,εi)}and computeℓθ(τi,εi)
4: θold←θ
5: foreach optimization epochdo
6: Sample mini-batch from collected trajectories
7: foreach state-action pair(ot,at)and corresponding MC samples{(τi,εi)}do
8: Computeℓθ(τi,εi)using stored(τi,εi)
9: ˆrθ←exp


−N^1 mc

PNmc
i=1(ℓθ(τi,εi)−ℓθold(τi,εi))



10: LFPO(θ)←min(ˆrθAˆt,clip(ˆrθ, 1 ±ε)Aˆt)
11: end for
12: θ←Optimizer(θ,∇θ
P

LFPO(θ))
13: end for
14: Update value function parametersφlike standard PPO
15: end while

Learned Flow and Target Action Distribution at

Denoising steps
Goal Agent

Gridworld with 2 Goals Sampled Trajectories

start end

Figure 1:Grid World. (Left) 25×25 GridWorld with green goal cells. Each arrow shows a denoised action
sampled from the FPO-trained policy, conditioned on a different latent noise vector. (Center) At the saddle-point
state (⋆) shown on the left, we visualize three denoising stepsτas the initial Gaussian gradually transforms into
the target distribution through the learned flow, illustrated by the deformation of the coordinate grid. (Right)
Sampled trajectories from the same starting states reach different goals, illustrating the multimodal behavior
captured by FPO.
4 Experiments

We assess FPO’s effectiveness by evaluating it in multiple domains. Our experiments include: (1) an
illustrative GridWorld environment using Gymnasium [ 64 , 65 ], (2) continuous control tasks with
MuJoCo Playground [ 14 , 66 ], and (3) physics-based humanoid control in Isaac Gym [ 67 ]. These
tasks vary in dimensionality, reward sparsity, horizon length, and simulation environments.

4.1 GridWorld

We first test FPO on a 25×25 GridWorld environment designed to probe the policy’s ability to capture
multimodal action distributions. As shown in Figure 1 left, the environment consists of two high
reward regions located as the top and bottom of the map (green cells). The reward is sparse: agents
receive a single reward upon reaching a goal or a penalty, with no intermediate rewards. This setup
creates saddle points where multiple distinct actions can lead to equally successful outcomes, offering
a natural opportunity to model diverse behaviors.

We train a diffusion policy from scratch using FPO by modifying a standard implementation [ 68 ]
of PPO. The policy is parameterized as a two-layer MLP modelingp(at|s,aτt), whereat∈R^2 is
the action,s∈R^2 is the grid state, andaτt∈R^2 is the latent noise vector at noise levelτ, initialized
fromN(0,I)atτ= 0. FPO consistently maximizes the return in this environment. The arrows in
Figure 1 left shows denoised actions at each grid location, computed by conditioning on a random
aτt∼ N(0,I)and running 10 steps of Euler integration. In Figure 1 center, we probe the learned

0 15304560

0

200

400

600

Eval Reward

BallInCup

0 15 304560

0

200

400

600

800

CartpoleBalance

0 1530 4560

0

200

400

600

800

CheetahRun

0 15304560

0

150

300

450

600

FingerSpin

0 15 304560

0

200

400

600

800

FingerTurnEasy

0 15304560

0

200

400

600

Eval Reward

FingerTurnHard

0 15 304560

0

150

300

450

600

FishSwim

0 1530 4560

0

200

400

600

800

PointMass

0 15304560

0

200

400

600

800

ReacherEasy

0 15 304560

0

200

400

600

800

ReacherHard

FPO Gaussian PPO

Figure 2:Comparison between FPO and Gaussian PPO [ 20 ] on DM Control Suite tasks.Results show
evaluation reward mean and standard error (y-axis) over 60M environment steps (x-axis). We run 5 seeds for
each task; the curve with the highest terminal evaluation reward is bolded.

0 15304560

0

200

400

600

Eval Reward

BallInCup

0 15 304560

0

200

400

600

800

CartpoleBalance

0 1530 4560

0

200

400

600

800

CheetahRun

0 15304560

0

150

300

450

600

FingerSpin

0 15 304560

0

200

400

600

800

FingerTurnEasy

0 15304560

0

200

400

600

Eval Reward

FingerTurnHard

0 15 304560

0

150

300

450

600

FishSwim

0 1530 4560

0

200

400

600

800

PointMass

0 15304560

0

200

400

600

800

ReacherEasy

0 15 304560

0

200

400

600

800

ReacherHard

FPO DPPO

Figure 3:Comparison between FPO and DPPO [ 59 ] on DM Control Suite tasks.Results show evaluation
reward mean and standard error (y-axis) over 60M environment steps (x-axis). We run 5 seeds for each task; the
curve with the highest terminal evaluation reward is bolded.

policy by visualizing the flow over its denoising steps at the saddle point. The initial Gaussian evolves
into a bimodal distribution, demonstrating that the policy captures the multi-modality of the solution
at this location. Figure 1 right shows multiple trajectories sampled from the policy, initialized from
various fixed starting positions. The agent exhibits multimodal behavior, with trajectories from the
same starting state reaching different goals. Even when heading toward the same goal, the paths vary
significantly, reflecting the policy’s ability to model diverse action sequences.

We also train a Gaussian policy using PPO, which successfully reaches the goal regions. Compared to
FPO, it exhibits more deterministic behavior, consistently favoring the nearest goal with less variation
in trajectory patterns. Results are included in the supplemental material (Appendix A.2).

4.2 MuJoCo Playground

Next, we evaluate FPO for continuous control using MuJoCo Playground [ 14 ]. We compare three
policy learning algorithms: (i) a Gaussian policy trained using PPO, (ii) a diffusion policy trained
using FPO, and (iii) a diffusion policy trained using DPPO [ 59 ]. We evaluate these algorithms on 5
seeds for each of 10 environments adapted from the DeepMind Control Suite [ 69 , 70 ]. Results are
reported in Figures 2 and 3.

Policy implementations.For the Gaussian policy baseline, we run the Brax [ 71 ]-based implementa-
tion used by MuJoCo Playground [ 14 ]’s PPO training scripts. We also use Brax PPO as a starting

Methods Goal conditioning Success rate (↑) Alive duration (↑) MPJPE (↓)
Gaussian PPO All joints 98. 7 % 200. 46 31. 62
FPO All joints 96 .4% 198. 00 41. 98
Gaussian PPO Root + Hands 46 .5% 142. 50 97. 65
FPO Root + Hands 70. 6 % 171. 32 62. 91
Gaussian PPO Root 29 .8% 114. 06 123. 70
FPO Root 54. 3 % 152. 90 73. 55

Table 2:Humanoid Control Quantitative Metrics.We compare FPO with Gaussian PPO with different
conditioning goals, and report the success rate, alive duration, and MPJPE averaged over all motion sequences.

point for implementing both FPO and DPPO. Following Section 3.2, only small changes are required
for FPO: noisy action and timestep inputs are included as input to the policy network, Gaussian
sampling is replaced with flow sampling, and the PPO loss’s likelihood ratio is replaced with the FPO
ratio approximation. For DPPO, we make the same policy network modification, but apply stochastic
sampling [ 55 ] during rollouts. We also augment each action in the experience buffer with the exact
sampling path that was taken to reach it. Following the two-layer MDP formulation in DPPO [ 59 ],
we then replace intractable action likelihoods with noise-conditioned sampling path likelihoods.

Hyperparameters.We match hyperparameters in Gaussian PPO, FPO, and DPPO training when-
ever possible: following the provided configurations in Playground [ 14 ], all experiments use
ADAM [ 72 ], 60M total environment steps, batch size 1024, and 16 updates per batch. For FPO
and DPPO, we use 10 sampling steps, set learning rates to 3e-4, and swept clipping epsilon
εclip∈ { 0. 01 , 0. 05 , 0. 1 , 0. 2 , 0. 3 }. For DPPO, we perturb each denoising step with Gaussian noise
with standard deviationσt, which we sweep∈{ 0. 01 , 0. 05 , 0. 1 }. We found thatεclip= 0. 05 produces
the best FPO results andεclip= 0. 2 ,σt= 0. 05 produced the best DPPO results; we use these values
for all experiments. For fairness, we also tuned learning rates and clipping epsilons for Gaussian
PPO. We provide more details about hyperparameters and baseline tuning in Appendix A.3.

Method Reward
Gaussian PPO 667.8±66.
Gaussian PPO† 577.2±74.
DPPO 652.5±83.
FPO‡ 759.3±45.
FPO, 1(τ,ε) 691.6±50.
FPO, 4(τ,ε) 731.2±58.
FPO,u-MSE 664.6±48.
FPO,εclip=0.1 623.3±76.
FPO,εclip=0.2 526.4±76.

Table 1: FPO variant comparison.
We report averages and standard errors
across MuJoCo tasks.†Using default
hyperparameters from MuJoCo Play-
ground.‡FPO results use 8(τ,ε)pairs,
ε-MSE,εclip= 0. 05.

Results.We observe in Figures 2 and 3 that FPO-optimized
policies outperform both Gaussian PPO and DPPO on the Play-
ground tasks. It outperforms both baselines in 8 of 10 tasks.

Analysis.In Table 1, we present average evaluation rewards
for baselines, FPO, and several variations of FPO. We observe:
(1)(τ,ε)sampling is important.Decreasing the number of
sampled pairs generally decreases evaluation rewards. More
samples can improve learning without requiring more expensive
environment steps.(2)ε-MSE is preferable overu-MSE in
Playground.ε-MSE refers to computing flow matching losses
by first converting velocity estimates toεnoise values;u-MSE
refers to MSE directly on velocity estimates. In Playground, we
found that the former produces higher average rewards. We hy-
pothesize that this is becauseεscale is invariant to action scale,
which results in better generalization forεclipchoices. For fair-
ness, we also performed learning rate and clipping ratio sweeps
for theu-MSE ablation.(3) Clipping.Like Gaussian PPO, the
choice ofεclipin FPO significantly impacts performance.

4.3 Humanoid Control

Physics-aware humanoid control is higher-dimensional than standard MuJoCo benchmarks, making
it a stringent test of FPO’s generality. We therefore train a humanoid policy to track motion-capture
(MoCap) trajectories in the PHC setting [73], using the open-source Puffer-PHC implementation as
our baseline^2. This experiment follows the goal-conditioned imitation-learning paradigm pioneered
by DeepMimic [ 74 ], in which simulated characters learn to reproduce reference motions. Depending
on the deployment needs, these reference signals (goals) can be as rich as full-body joint information

(^2) https://github.com/kywch/puffer-phc

(a) Episode return along training. (b) Root+hand conditioning. (c) Rough terrain locomotion.

Figure 4:Physics-based Humanoid Control.(a) The curves show that FPO performance is close to that of
Gaussian-PPO when conditioning on all joints and surpasses it when goals are reduced to the root or root+hands,
indicating stronger robustness to sparse conditioning. (b) In the root+hands goal setting, FPO (blue) tracks the
reference motion (grey) while Gaussian-PPO (orange) falls. (c) Trained with terrain randomization, FPO walks
stably across procedurally generated rough ground.

or as sparse as root joint (pelvis) commands, providing the flexibility required for reliable sim-to-real
transfer [ 29 ]. The problem with sparse goals is under-conditioned and significantly more challenging,
requiring the policy to fill in the missing joint specification in a manner that is physically plausible.

Implementation details.Our simulated agent is an SMPL-based humanoid with 24 actuated joints,
each offering six degrees of freedom and organized in a kinematic tree rooted at the pelvis, simulated
in Isaac Gym [ 67 ]. The policy receives both proprioceptive observations and goal information
computed from the motion-capture reference. A single policy is trained to track AMASS [ 75 ]
motions following PHC [ 73 ]. We use the root height, joint positions, rotations, velocity, and angular
velocity in a local coordinate frame as the robot state. For goal conditioning, we compute the
difference between the tracking joint information (positions, rotations, velocity, and angular velocity)
and the current robot’s joint information, as well as the tracking joint locations and rotations. We
explore both full conditioning,i.e.,conditioning on all joint targets, and under conditioning,i.e.,
conditioning only on the root or the root and hands targets. The latter matches the target signals
typically provided by a joystick or VR controller. Please note that the same imitation reward based
on all joints is used for both conditioning experiments. The per-joint tracking reward is computed as
in DeepMimic [74].

Evaluation.For evaluation, we compute the success rate, considering an imitation unsuccessful if
the average distance between the body joints and the reference motion exceeds 0.5 meters at any
point during the sequence. We also report the average duration the agent stays alive till it completes
the tracking or falls. Finally, we compute the global mean per-joint position error (MPJPE) on the
conditioned goals.

Results.Figure 4a shows that we successfully train FPO from scratch on this high-dimensional
control task. With full joint conditioning, FPO performance is close to Gaussian PPO. However,
when the model is under-conditioned—e.g., conditioned only on the root or the root and hands—FPO
outperforms Gaussian PPO, highlighting the advantage of flow-based policies. While prior methods
can also achieve sparse-goal control, they often rely on training a teacher policy that conditions on
full joint reference first and then distilling the knowledge to sparse conditioned policies [ 76 , 29 , 77 ]
or training a separate encoder observing sparse references [78, 79].

Figure 4b visualizes the behaviors in the root+hands setting (left-to-right: reference motion, FPO,
Gaussian-PPO); FPO tracks the target closely, whereas the Gaussian policy drifts. Table 2 quantifies
these trends, with FPO achieving much higher success rates in the under-conditioned scenarios.
Finally, as illustrated in Fig. 4c, FPO trained with terrain randomization enables the humanoid to
traverse rough terrain, showing potential for sim-to-real transfer. Please see the supplemental video
for more qualitative results.
5 Discussion and Limitations

We introduce Flow Policy Optimization (FPO), an algorithm for training flow-based generative models
using policy gradients. FPO reformulates policy optimization as minimizing an advantage-weighted

conditional flow matching (CFM) objective, enabling stable training without requiring explicit
likelihood computation. It integrates easily with PPO-style algorithms, and crucially, preserves the
flow-based structure of the policy—allowing the resulting model to be used with standard flow-based
mechanisms such as sampling, distillation, and fine-tuning. We demonstrate FPO across a range of
control tasks, including a challenging humanoid setting where it enables training from scratch under
sparse goal conditioning, where Gaussian policies fail to learn.
The training and deployment of flow-based policies is generally more computationally intensive than
for corresponding Gaussian policies. FPO also lacks established machinery such as KL divergence
estimation for adaptive learning rates and entropy regularization.
We also explored applying FPO to fine-tune a pre-trained image diffusion model using reinforcement
learning. While promising in principle, we found this setting to be unstable in practice—likely due to
the issue of fine-tuning diffusion models on its own output multiple times as noted in recent works [ 80 –
82 ]. In particular, we observed sensitivity to classifier-free guidance (CFG) that compounds with
self-generated data, even outside of the RL framework. This suggests that the instability is not
a limitation of FPO itself, but a broader challenge in applying reinforcement learning to image
generation. Please see the supplementary material for more detail.
Despite these limitations, FPO offers a simple and flexible bridge between flow-based models and
online reinforcement learning. We are particularly excited to see future work apply FPO in settings
where flow-based policies are already pretrained—such as behavior-cloned diffusion policies in
robotics—where FPO’s compatibility and simplicity may offer practical benefits for fine-tuning with
task reward.

Acknowledgments

We thank Qiyang (Colin) Li, Oleg Rybkin, Lily Goli and Michael Psenka for helpful discussions
and feedback on the manuscript. We thank Arthur Allshire, Tero Karras, Miika Aittala, Kevin
Zakka and Seohong Park for insightful input and feedback on implementation details and the broader
context of this work. This project was funded in part by NSF:CNS-2235013, IARPA DOI/IBC No.
140D0423C0035, and Bakar fellows. CK and BY are supported by NSF fellowship. SG is supported
by the NVIDIA Graduate Fellowship
References

[1]Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical
text-conditional image generation with clip latents.arXiv preprint arXiv:2204.06125, 2022.

[2]Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar
Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic
text-to-image diffusion models with deep language understanding. 2022.

[3]Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko,
Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High
definition video generation with diffusion models.arXiv preprint arXiv:2210.02303, 2022.

[4]Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr,
Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh.
Video generation models as world simulators. 2024. URLhttps://openai.com/research/
video-generation-models-as-world-simulators.

[5]Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv
Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media
foundation models.arXiv preprint arXiv:2410.13720, 2024.

[6]Veo-Team, :, Agrim Gupta, Ali Razavi, Andeep Toor, Ankush Gupta, Dumitru Erhan, Eleni
Shaw, Eric Lau, Frank Belletti, Gabe Barth-Maron, Gregory Shaw, Hakan Erdogan, Hakim
Sidahmed, Henna Nandwani, Hernan Moraldo, Hyunjik Kim, Irina Blok, Jeff Donahue, José
Lezama, Kory Mathewson, Kurtis David, Matthieu Kim Lorrain, Marc van Zee, Medhini
Narasimhan, Miaosen Wang, Mohammad Babaeizadeh, Nelly Papalampidi, Nick Pezzotti,
Nilpa Jha, Parker Barnes, Pieter-Jan Kindermans, Rachel Hornung, Ruben Villegas, Ryan
Poplin, Salah Zaiem, Sander Dieleman, Sayna Ebrahimi, Scott Wisdom, Serena Zhang, Shlomi
Fruchter, Signe Nørly, Weizhe Hua, Xinchen Yan, Yuqing Du, and Yutian Chen. Veo 2. 2024.
URLhttps://deepmind.google/technologies/veo/veo-2/.

[7] Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, and
Mark D. Plumbley. Audioldm: Text-to-audio generation with latent diffusion models, 2023.
URLhttps://arxiv.org/abs/2301.12503.

[8]Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile
diffusion model for audio synthesis, 2021. URLhttps://arxiv.org/abs/2009.09761.

[9]Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ
Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion.
The International Journal of Robotics Research, 2024.

[10]Sanjeev Raja, Martin Šípka, Michael Psenka, Tobias Kreiman, Michal Pavelka, and Aditi S
Krishnapriyan. Action-minimization meets generative modeling: Efficient transition path
sampling with the onsager-machlup functional.arXiv preprint arXiv:2504.18506, 2025.

[11]Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans,
Quoc V Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of
foundation model post-training.arXiv preprint arXiv:2501.17161, 2025.

[12]Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr,
Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient
mixture-of-experts language model.arXiv preprint arXiv:2405.04434, 2024.

[13]Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow
matching for generative modeling, 2023. URLhttps://arxiv.org/abs/2210.02747.

[14]Kevin Zakka, Baruch Tabanpour, Qiayuan Liao, Mustafa Haiderbhai, Samuel Holt, Jing Yuan
Luo, Arthur Allshire, Erik Frey, Koushil Sreenath, Lueder A Kahrs, et al. Mujoco playground.
arXiv preprint arXiv:2502.08844, 2025.

[15]Richard S. Sutton, David McAllester, Satinder P. Singh, and Yishay Mansour. Policy gradient
methods for reinforcement learning with function approximation. InProceedings of the 12th
International Conference on Neural Information Processing Systems (NeurIPS), pages 1057–
1063, 1999.

[16]Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforce-
ment learning.Machine learning, 1992.

[17]Sham M. Kakade. A natural policy gradient. InProceedings of the 14th International Conference
on Neural Information Processing Systems (NeurIPS), pages 1531–1538, 2002.

[18]Jan Peters and Stefan Schaal. Natural actor–critic.Neurocomputing, 71(7–9):1180–1190, 2008.

[19]John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust
region policy optimization. InInternational conference on machine learning, pages 1889–1897.
PMLR, 2015.

[20]John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal
policy optimization algorithms.arXiv preprint arXiv:1707.06347, 2017.

[21]Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Tim Harley, Timothy
Lillicrap, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement
learning. InProceedings of the 33rd International Conference on Machine Learning (ICML),
pages 1928–1937, 2016.

[22]Ziyu Wang, Tom Schaul, Matteo Hessel, Hado Hasselt, Marc Lanctot, and Nando de Freitas.
Sample efficient actor–critic with experience replay. InProceedings of the 30th International
Conference on Neural Information Processing Systems (NeurIPS), pages 1061–1071, 2016.

[23]Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical
reasoning in open language models.arXiv preprint arXiv:2402.03300, 2024.

[24]Yan Duan, Xi Chen, Rein Houthooft, John Schulman, and Pieter Abbeel. Benchmarking deep
reinforcement learning for continuous control. InInternational conference on machine learning,
pages 1329–1338. PMLR, 2016.

[25]Shengyi Huang, Quentin Gallouédec, Florian Felten, Antonin Raffin, Rousslan Fernand Julien
Dossa, Yanxiao Zhao, Ryan Sullivan, Viktor Makoviychuk, Denys Makoviichuk, Mohamad H
Danesh, et al. Open rl benchmark: Comprehensive tracked experiments for reinforcement
learning.arXiv preprint arXiv:2402.03046, 2024.

[26]Nikita Rudin, David Hoeller, Philipp Reist, and Marco Hutter. Learning to walk in minutes
using massively parallel deep reinforcement learning. InProceedings of the 5th Conference
on Robot Learning, volume 164 ofProceedings of Machine Learning Research, pages 91–100.
PMLR, 2022. URLhttps://proceedings.mlr.press/v164/rudin22a.html.

[27]Clemens Schwarke, Victor Klemm, Matthijs van der Boon, Marko Bjelonic, and Marco Hutter.
Curiosity-driven learning of joint locomotion and manipulation tasks. InProceedings of
The 7th Conference on Robot Learning, volume 229 ofProceedings of Machine Learning
Research, pages 2594–2610. PMLR, 2023. URLhttps://proceedings.mlr.press/v229/
schwarke23a.html.

[28]Mayank Mittal, Nikita Rudin, Victor Klemm, Arthur Allshire, and Marco Hutter. Sym-
metry considerations for learning task symmetric robot policies. In2024 IEEE Interna-
tional Conference on Robotics and Automation (ICRA), pages 7433–7439, 2024. doi:
10.1109/ICRA57147.2024.10611493.

[29]Arthur Allshire, Hongsuk Choi, Junyi Zhang, David McAllister, Anthony Zhang, Chung Min
Kim, Trevor Darrell, Pieter Abbeel, Jitendra Malik, and Angjoo Kanazawa. Visual imitation
enables contextual humanoid control.arXiv preprint arXiv:2505.03729, 2025.

[30]Ilge Akkaya, Marcin Andrychowicz, Maciek Chociej, Mateusz Litwin, Bob McGrew, Arthur
Petron, Alex Paino, Matthias Plappert, Glenn Powell, Raphael Ribas, et al. Solving rubik’s cube
with a robot hand.arXiv preprint arXiv:1910.07113, 2019.

[31]Tao Chen, Jie Xu, and Pulkit Agrawal. A system for general in-hand object re-orientation.
Conference on Robot Learning, 2021.

[32]Haozhi Qi, Brent Yi, Sudharshan Suresh, Mike Lambeta, Yi Ma, Roberto Calandra, and Jitendra
Malik. General in-hand object rotation with vision and touch. InConference on Robot Learning,
pages 2549–2564. PMLR, 2023.

[33]Haozhi Qi, Brent Yi, Mike Lambeta, Yi Ma, Roberto Calandra, and Jitendra Malik. From simple
to complex skills: The case of in-hand object reorientation.arXiv preprint arXiv:2501.05439,

[34]Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin,
Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to
follow instructions with human feedback.Advances in neural information processing systems,

[35]Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei.
Deep reinforcement learning from human preferences, 2023. URLhttps://arxiv.org/abs/
1706.03741.

[36]DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin
Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu,
Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan
Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang,
Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli
Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng
Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li,
Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian
Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean
Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan
Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian,
Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong
Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan
Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting
Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun,
T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu,
Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao
Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su,
Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang
Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X.
Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao
Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang
Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He,
Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong
Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha,
Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan
Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu,
Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang,
and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement
learning, 2025. URLhttps://arxiv.org/abs/2501.12948.

[37]Mistral-AI, :, Abhinav Rastogi, Albert Q. Jiang, Andy Lo, Gabrielle Berrada, Guillaume
Lample, Jason Rute, Joep Barmentlo, Karmesh Yadav, Kartik Khandelwal, Khyathi Raghavi
Chandu, Léonard Blier, Lucile Saulnier, Matthieu Dinot, Maxime Darrin, Neha Gupta, Roman
Soletskyi, Sagar Vaze, Teven Le Scao, Yihan Wang, Adam Yang, Alexander H. Liu, Alexan-
dre Sablayrolles, Amélie Héliou, Amélie Martin, Andy Ehrenberg, Anmol Agarwal, Antoine
Roux, Arthur Darcet, Arthur Mensch, Baptiste Bout, Baptiste Rozière, Baudouin De Monicault,

Chris Bamford, Christian Wallenwein, Christophe Renaudin, Clémence Lanfranchi, Darius
Dabert, Devon Mizelle, Diego de las Casas, Elliot Chane-Sane, Emilien Fugier, Emma Bou
Hanna, Gauthier Delerce, Gauthier Guinet, Georgii Novikov, Guillaume Martin, Himanshu
Jaju, Jan Ludziejewski, Jean-Hadrien Chabran, Jean-Malo Delignon, Joachim Studnia, Jonas
Amar, Josselin Somerville Roberts, Julien Denize, Karan Saxena, Kush Jain, Lingxiao Zhao,
Louis Martin, Luyu Gao, Lélio Renard Lavaud, Marie Pellat, Mathilde Guillaumin, Mathis
Felardos, Maximilian Augustin, Mickaël Seznec, Nikhil Raghuraman, Olivier Duchenne, Patri-
cia Wang, Patrick von Platen, Patryk Saffer, Paul Jacob, Paul Wambergue, Paula Kurylowicz,
Pavankumar Reddy Muddireddy, Philomène Chagniot, Pierre Stock, Pravesh Agrawal, Romain
Sauvestre, Rémi Delacourt, Sanchit Gandhi, Sandeep Subramanian, Shashwat Dalal, Siddharth
Gandhi, Soham Ghosh, Srijan Mishra, Sumukh Aithal, Szymon Antoniak, Thibault Schueller,
Thibaut Lavril, Thomas Robert, Thomas Wang, Timothée Lacroix, Valeriia Nemychnikova,
Victor Paltz, Virgile Richard, Wen-Ding Li, William Marshall, Xuanyu Zhang, and Yunhao
Tang. Magistral, 2025. URLhttps://arxiv.org/abs/2506.10910.

[38]Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models.Advances
in neural information processing systems, 2020.

[39] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models, 2022.
URLhttps://arxiv.org/abs/2010.02502.

[40]Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-
resolution image synthesis with latent diffusion models, 2022. URLhttps://arxiv.org/
abs/2112.10752.

[41]Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data
distribution, 2020. URLhttps://arxiv.org/abs/1907.05600.

[42]Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and
David J. Fleet. Video diffusion models, 2022. URLhttps://arxiv.org/abs/2204.03458.

[43]Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry
Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video:
Text-to-video generation without text-video data, 2022. URLhttps://arxiv.org/abs/
2209.14792.

[44]Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko,
Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans.
Imagen video: High definition video generation with diffusion models, 2022. URLhttps:
//arxiv.org/abs/2210.02303.

[45]Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, and Mikhail Kudinov. Grad-
tts: A diffusion probabilistic model for text-to-speech, 2021. URLhttps://arxiv.org/abs/
2105.06337.

[46]Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, Najim Dehak, and
William Chan. Wavegrad 2: Iterative refinement for text-to-speech synthesis, 2021. URL
https://arxiv.org/abs/2106.09660.

[47]Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo
Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming
Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang
Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky.π 0 : A
vision-language-action flow model for general robot control, 2024. URLhttps://arxiv.
org/abs/2410.24164.

[48]NVIDIA, :, Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding,
Linxi "Jim" Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang,
Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu,
Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed,
You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie,
Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao,
Ruijie Zheng, and Yuke Zhu. Gr00t n1: An open foundation model for generalist humanoid
robots, 2025. URLhttps://arxiv.org/abs/2503.14734.

[49]Marta Skreta, Lazar Atanackovic, Avishek Joey Bose, Alexander Tong, and Kirill Neklyudov.
The superposition of diffusion models using the itô density estimator, 2025. URLhttps:
//arxiv.org/abs/2412.17762.

[50]Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ
Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion.
The International Journal of Robotics Research, 2024.

[51]Anurag Ajay, Yilun Du, Abhi Gupta, Joshua B. Tenenbaum, Tommi S. Jaakkola, and Pulkit
Agrawal. Is conditional generative modeling all you need for decision making? InThe Eleventh
International Conference on Learning Representations, 2023.

[52]Michael Janner, Yilun Du, Joshua B Tenenbaum, and Sergey Levine. Planning with diffusion
for flexible behavior synthesis.arXiv preprint arXiv:2205.09991, 2022.

[53]Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter
Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning text-to-image models
using human feedback.arXiv preprint arXiv:2302.12192, 2023.

[54]Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion
models with reinforcement learning.arXiv preprint arXiv:2305.13301, 2023.

[55]Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan,
Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl.arXiv
preprint arXiv:2505.05470, 2025.

[56]Michael Psenka, Alejandro Escontrela, Pieter Abbeel, and Yi Ma. Learning a diffusion model
policy from rewards via q-score matching.arXiv preprint arXiv:2312.11752, 2023.

[57]Younggyo Seo, Carmelo Sferrazza, Haoran Geng, Michal Nauman, Zhao-Heng Yin, and Pieter
Abbeel. Fasttd3: Simple, fast, and capable reinforcement learning for humanoid control, 2025.
URLhttps://arxiv.org/abs/2505.22642.

[58]Scott Fujimoto, Herke van Hoof, and David Meger. Addressing function approximation error in
actor-critic methods, 2018. URLhttps://arxiv.org/abs/1802.09477.

[59]Allen Z Ren, Justin Lidard, Lars L Ankile, Anthony Simeonov, Pulkit Agrawal, Anirudha
Majumdar, Benjamin Burchfiel, Hongkai Dai, and Max Simchowitz. Diffusion policy policy
optimization.arXiv preprint arXiv:2409.00588, 2024.

[60]John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-
dimensional continuous control using generalized advantage estimation. arXiv preprint
arXiv:1506.02438, 2015.

[61]Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of
diffusion-based generative models.Advances in neural information processing systems, 35:
26565–26577, 2022.

[62]Diederik P. Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models,

    URLhttps://arxiv.org/abs/2107.00630.

[63]Diederik P. Kingma and Ruiqi Gao. Understanding diffusion objectives as the elbo with simple
data augmentation, 2023. URLhttps://arxiv.org/abs/2303.00848.

[64]Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang,
and Wojciech Zaremba. Openai gym, 2016.

[65]Mark Towers, Ariel Kwiatkowski, Jordan Terry, John U Balis, Gianluca De Cola, Tristan Deleu,
Manuel Goulão, Andreas Kallinteris, Markus Krimmel, Arjun KG, et al. Gymnasium: A
standard interface for reinforcement learning environments.arXiv preprint arXiv:2407.17032,

[66]Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based
control. In2012 IEEE/RSJ international conference on intelligent robots and systems, 2012.

[67]Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles
Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, et al. Isaac gym: High
performance gpu-based physics simulation for robot learning.arXiv preprint arXiv:2108.10470,

[68]Eric Yang Yu. Ppo-for-beginners: A simple, well-styled ppo implementation in pytorch.
https://github.com/ericyangyu/PPO-for-Beginners, 2020. GitHub repository.

[69]Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David
Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, et al. Deepmind control suite.
arXiv preprint arXiv:1801.00690, 2018.

[70]Saran Tunyasuvunakool, Alistair Muldal, Yotam Doron, Siqi Liu, Steven Bohez, Josh Merel,
Tom Erez, Timothy Lillicrap, Nicolas Heess, and Yuval Tassa. dm_control: Software and tasks
for continuous control.Software Impacts, 6:100022, 2020.

[71]C Daniel Freeman, Erik Frey, Anton Raichuk, Sertan Girgin, Igor Mordatch, and Olivier
Bachem. Brax–a differentiable physics engine for large scale rigid body simulation.arXiv
preprint arXiv:2106.13281, 2021.

[72]Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint
arXiv:1412.6980, 2014.

[73]Zhengyi Luo, Jinkun Cao, Kris Kitani, Weipeng Xu, et al. Perpetual humanoid control for
real-time simulated avatars. InProceedings of the IEEE/CVF International Conference on
Computer Vision, pages 10895–10904, 2023.

[74]Xue Bin Peng, Pieter Abbeel, Sergey Levine, and Michiel Van de Panne. Deepmimic: Example-
guided deep reinforcement learning of physics-based character skills.ACM Transactions On
Graphics (TOG), 37(4):1–14, 2018.

[75]Naureen Mahmood, Nima Ghorbani, Nikolaus F Troje, Gerard Pons-Moll, and Michael J
Black. Amass: Archive of motion capture as surface shapes. InProceedings of the IEEE/CVF
international conference on computer vision, pages 5442–5451, 2019.

[76]Chen Tessler, Yunrong Guo, Ofir Nabati, Gal Chechik, and Xue Bin Peng. Maskedmimic:
Unified physics-based character control through masked motion inpainting.ACM Transactions
on Graphics (TOG), 43(6):1–21, 2024.

[77]Yixuan Li, Yutang Lin, Jieming Cui, Tengyu Liu, Wei Liang, Yixin Zhu, and Siyuan Huang.
Clone: Closed-loop whole-body humanoid teleoperation for long-horizon tasks.arXiv preprint
arXiv:2506.08931, 2025.

[78]Zhengyi Luo, Jinkun Cao, Josh Merel, Alexander Winkler, Jing Huang, Kris Kitani, and
Weipeng Xu. Universal humanoid motion representations for physics-based control.arXiv
preprint arXiv:2310.04582, 2023.

[79]Zhengyi Luo, Jinkun Cao, Sammy Christen, Alexander Winkler, Kris Kitani, and Weipeng
Xu. Omnigrasp: Grasping diverse objects with simulated humanoids. InAdvances in Neural
Information Processing Systems, volume 37, pages 2161–2184, 2024.

[80]Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, and Yarin
Gal. Ai models collapse when trained on recursively generated data. Nature, 631(8022):
755–759, 2024.

[81]Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Yarin Gal, Nicolas Papernot, and Ross Ander-
son. The curse of recursion: Training on generated data makes models forget.arXiv preprint
arXiv:2305.17493, 2023.

[82]Sina Alemohammad, Josue Casco-Rodriguez, Lorenzo Luzi, Ahmed Imtiaz Humayun, Hossein
Babaei, Daniel LeJeune, Ali Siahkoohi, and Richard G Baraniuk. Self-consuming generative
models go mad. International Conference on Learning Representations (ICLR), 2024.

[83]Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models.
arXiv preprint arXiv:2202.00512, 2022.

[84]Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance, 2022. URLhttps://
arxiv.org/abs/2207.12598.
Flow Matching Policy Gradients
Supplementary Material

In this supplementary material, we discuss the deferred proofs of technical results, elaborate on the
details of our experiments, and present additional visual results for the grid world, humanoid control,
and image finetuning experiments.
A.1 FPO Derivation

The mathematical details presented in this section provide expanded derivations and additional context
for the theoretical results outlined in Section 3 of the main text. Specifically, we elaborate on the
connection between the conditional flow matching objective and the evidence lower bound (ELBO)
first mentioned in Section 3.4, and provide complete derivations for the FPO ratio introduced in
Section 3.3. These details are included for completeness and to situate our work within the theoretical
framework established by Kingma et al. [ 63 ], but are not necessary for understanding the core FPO
algorithm or implementing it in practice.

First, we detail the different popular loss weightings used when training flow matching models laid
out by Kingma et al. [ 63 ]. These weightings, denoted asw(λt), determine how losses at different
noise levels contribute to the overall objective and lead to different theoretical interpretations of Flow
Policy Optimization.

Then, we show the more general result, which is that FPO optimizes the advantage-weighted expected
ELBO of the noise-perturbed data. Specifically, for any monotonic weighting function (including
Optimal Transport CFM schedules [13]), we can express the weighted loss as:

Lwθ(at) =−Epw(τ),q(aτt|at)[ELBOτ(aτt)] +c 1 , (21)

wherepw(τ)is the distribution over timesteps induced by the weighting function, andELBOτ(aτt)is
the evidence lower bound at noise levelτfor the perturbed actionaτt.

This means that FPO increases the likelihood of high-reward samples and the intermediate noisy

samplesaτtfrom the sample path. By weighting this objective with advantagesAˆτ, we guide the
policy to direct probability flow toward action neighborhoods that produce higher reward.

For diffusion schedules with uniform weightingw(λτ) = 1, we show a somewhat stronger theoretical
result. In this special case, the weighted loss directly corresponds to maximizing the ELBO of clean
actions:

−ELBO(at) =

1
2

Eτ∼U(0,1),ε∼N(0,I)


−

dλ
dτ

·∥ˆεθ(aτt;λτ)−ε∥^22



+c 2 , (22)

which is a more direct connection to maximum likelihood estimation.

A.1.1 Loss Weighting Choices

Most popular instantiations of flow-based and diffusion models can be reparameterized in the weighted
loss scheme proposed by Kingma et al. [ 63 ]. This unified framework expresses each version as an
instance of a weighted denoising loss:

Lwθ(x) =

1
2

Eτ∼U(0,1),ε∼N(0,I)[w(λτ)·−

dλ
dτ

·∥ˆεθ(aτt;λτ)−ε∥^22 ], (23)

wherew(λτ)is a time-dependent function that determines the relative importance of different noise
levels.

For those with a loss weight that varies monotonically with noise timestepτ, the aforementioned
relationship between the weighted loss and expected ELBO holds. Specifically, whenw(λτ)is
monotonically increasing withτ, Kingma et al. prove:

Lwθ(at) =−Epw(τ),q(aτt|at)[ELBOτ(aτt)] +c 1 , (24)

wherec 1 is a constant, and does not vary with model parameters.

These monotonic weightings include several popular schedules: (1) standard diffusion with uniform
weightingw(λτ) = 1[ 38 ], (2) optimal transport linear interpolation schedule [ 13 ], which yields
w(λτ) =e−λ/^2 , and (3) velocity prediction (v-prediction) with cosine schedule [ 83 ], which also

yieldsw(λτ) =e−λ/^2.

A.1.2 Flow Matching as Expected ELBO Optimization

To derive FPO in the more general flow matching case, we begin with the standard policy gradient
objective, but replace direct likelihood maximization with maximization of the ELBO for noise-
perturbed data:

max
θ

Eat∼πθ(at|ot)

h
Epw(τ),q(aτt|at)[ELBOτ(aτt)]·Aˆt

i
, (25)

wheretis temporal rollout time andτis diffusion/flow noise timestep.

This formulation directly leverages the result from Kingma et al. [ 63 ] that for monotonic weightings,
the weighted denoising loss equals the negative expected ELBO of noise-perturbed data plus a
constant:

Lwθ(at) =−Epw(τ),q(aτt|at)[ELBOτ(aτt)] +c 1. (26)

To apply this within a trust region approach similar to PPO, we need to define a ratio between the
current and old policies. Since we are working with expected ELBOs, the appropriate ratio becomes:

rFPO(θ) =

exp(Epw(τ),q(aτt|at)[ELBOτ(aτt)]θ)
exp(Epw(τ),q(aτt|at)[ELBOτ(aτt)]θ,old)

(27)

This ratio represents the relative likelihood of actions and their noisy versions under the current policy
compared to the old policy.

It is important to note that the constantc 1 in the ELBO equivalence depends only on the noise
schedule endpointsλminandλmax, the data distribution, and the forward process, but not on the
model parameterθ. This is critical for our derivation. It ensures that within a single trust region data
collection and training episode, this constant remains identical between the old policyθoldand the
updated policyθ. Consequently, when forming the ratiorFPO(θ), these constants cancel out:

rFPO(θ) =

exp(Epw(τ),q(aτt|at)[ELBOτ(aτt)]θ+c 1 )
exp(Epw(τ),q(aτt|at)[ELBOτ(aτt)]θ,old+c 1 )

=

exp(Epw(τ),q(aτt|at)[ELBOτ(aτt)]θ)
exp(Epw(τ),q(aτt|at)[ELBOτ(aτt)]θ,old)
(28)

We estimate this ratio through Monte Carlo sampling of timestepsτand noiseε:

rˆFPO(τ,ε) = exp(−ℓθ(τ,ε) +ℓθ,old(τ,ε)), (29)

whereℓθ(τ,ε) =^12 [−λ ̇(τ)]∥ˆεθ(aτt;λτ)−ε∥^2 is the reparameterized conditional flow matching loss
for a single draw of random variablesεandτ.

As discussed in the main text,ˆrFPOoverestimates the scale but unbiasedly estimates the direction of
the gradient. We can reduce or eliminate the scale bias by drawing more samples ofτandε.

A.1.3 FPO with Diffusion Schedules

For the special case of standard diffusion schedules with uniform weightingw(λt) = 1, we can
derive a stronger theoretical result connecting our optimization objective directly to the ELBO of
clean (non-noised) data.

Goal

Gridworld with 2 Goals Sampled Trajectories

start end

Gaussian Policy noise = 0.0 noise = 0.1 noise = 0.

Figure A.1:GridWorld with Gaussian Policy.Left) 25 × 25 GridWorld with green goal cells. Each arrow
shows an action predicted by the Gaussian policy. Right) Four rollouts under test-time noise perturbations
(σ= 0. 0 , 0. 1 , 0. 5 ). While the Gaussian policy achieves the goal, its trajectories lack diversity and hit the same
goal consistently when given the same initialization point.

As shown by Kingma et al. [ 63 ], when using uniform weighting, the weighted loss directly corre-
sponds to the negative ELBO of the clean data plus a constant:

−ELBO(at) =

1
2

Eτ∼U(0,1),ε∼N(0,I)


−

dλ
dτ

·∥ˆεθ(aτt;λτ)−ε∥^22



+c 2 , (30)

wherec 2 is a different constant thanc 1 that also does not depend on model parameterθ.

This means that minimizing the unweighted loss (w(λτ) = 1) is equivalent to maximizing the
ELBO of the clean actionat, providing a more direct connection to traditional maximum likelihood
estimation.

In the context of FPO, we can therefore express our advantage-weighted objective as:

max
θ

Eat∼πθ(at|ot)

h
ELBOθ(at)·Aˆt

i
(31)

In this case, the objective direct increases a lower bound of the log-likelihood of clean actionsat
weighted by their advantages, rather than over noise-perturbed actions.

The FPO ratio in this case becomes:

rFPO(θ) =

exp(ELBOθ(at))
exp(ELBOθ,old(at))

(32)

This specific case highlights the close relationship between FPO and traditional maximum likelihood
methods common for PPO [20]. FPO still retains the computational advantages of avoiding explicit
likelihood computations.

As in the general case, our Monte Carlo estimator exhibits upward bias of gradient scale. We can use
the same PPO clipping mechanism to control the magnitude of parameter changes.

A.1.4 Advantage-Weighed Flow Matching Discussion

Advantage estimates are typically zero-centered to reduce variance in estimating the policy gradient.
Flow matching, however, learns probability flows which must be nonnegative by construction. Since
advantages function as loss weights in this context, they should remain positive for mathematical
consistency. A constant shift does not affect policy gradient optimization, which follows from the
same baseline-invariance property that justifies using advantages in the first place. We find that both
processed and unprocessed advantages work empirically.
A.2 GridWorld

Figure A.1 shows results from the Gaussian policy on the same Grid World trained using PPO. While
the Gaussian policy can learn optimal behaviors, the trajectories resulting from it are not as diverse as

Learning Rate Clipping Epsilon (εclip)
0.3 0.2 0.1 0.05 0.03 0.01
0.0001 589.5 648.5 646.6 608.6 500.5 458.5
0.001 556.0 646.1 654.6 636.2 562.6 471.8
0.003 548.9 603.1 586.4 535.7 480.8 400.8
0.0003 567.0 631.8 667.8 650.9 570.4 492.0
0.0005 544.8 586.8 629.5 559.7 505.6 406.5

Table A.1:Hyperparameter sweep for Gaussian PPO on the subset of Playground tasks that we evaluate
on.All quantities are average rewards across 10 tasks, with 5 seeds per task. The default configuration in
Playground [ 14 ] (before tuning) uses learning rate 1e-3 and clipping epsilon 0.3; the tuned variant we use for
results in the main paper body sets learning rate to 3e-4 and clipping epsilon to 0.1.

0 15304560

0

150

300

450

Eval Reward

BallInCup

0 15 304560

0

250

500

750

1000

CartpoleBalance

0 1530 4560

0

150

300

450

600

CheetahRun

0 15304560

0

150

300

450

600

FingerSpin

0 15 304560

0

200

400

600

800

FingerTurnEasy

0 15304560

0

200

400

600

Eval Reward

FingerTurnHard

0 15 304560

0

150

300

450

FishSwim

0 1530 4560

0

200

400

600

800

PointMass

0 15304560

0

200

400

600

800

ReacherEasy

0 15 304560

0

200

400

600

800

ReacherHard

Gaussian PPO (tuned) Gaussian PPO (before tuning)

Figure A.2:Gaussian PPO baseline results before and after tuning.We tune clipping epsilon and learning
rate to maximize average performance across tasks. Results show evaluation reward mean and standard error
(y-axis) over 60M environment steps (x-axis). We run 5 seeds for each task; the curve with the highest terminal
evaluation reward is bolded.

those of the diffusion policy. We visualize 4 samples from the Gaussian policy with 0.0, 0.1, and 0.5
random noise perturbations at test time (Fig. A.1, right). Note that despite being initialized at the
midpoint of the environment, all shown positions lead to asinglegoal mode, never both.
A.3 MuJoCo Playground

Table A.2 shows hyperparameters used for PPO training in the MuJoCo Playground environment.
These are imported directly from the configurations provided by MuJoCo Playground [ 14 ], but after
sweeping hyperparameters to tune learning rate and clipping coefficients (Table A.1). We visualize
improvements from this sweep in Figure A.2. Our flow matching and diffusion-based policies
use the same hyperparameters, but adjust the clipping coefficient, turn off the entropy coefficient,
and for DPPO [ 59 ], introduce a stochastic sampling variance to account for the change in policy
representation.
A.4 Humanoid Control

In Table A.3, we report the detailed hyperparameters that we used for training both the Gaussian
policy with PPO and the Diffusion policy with FPO in the humanoid control experiment. Note
that we use the same set of hyperparameters for both policies. In our project webpage, we also
provide videos showing qualitative comparisons between the Gaussian policy and ours on tracking an
under-conditioned reference, and visual results of FPO on different terrains.

Hyperparameter Value
Discount factor (γ) 0.995 (most environments)
0.95 (BallInCup, FingerSpin)
GAEλ 0.95
Value loss coefficient 0.25
Entropy coefficient 0.01
Reward scaling 10.0
Normalize advantage True
Normalize observations True
Action repeat 1
Unroll length 30
Batch size 1024
Number of minibatches 32
Number of updates per batch 16
Number of environments 2048
Number of evaluations 10
Number of timesteps 60M
Policy network MLP (4 hidden layers, 32 units)
Value network MLP (5 hidden layers, 256 units)
Optimizer Adam

Table A.2:PPO hyperparameters imported from MuJoCo playground [14].

Hyperparameter Value Hyperparameter Value
Policy Settings
Hidden size 512 Solver step size 0.1
Action perturbation std 0.05 Target KL divergence None
Number of environments 4096 Normalize advantage True
Training Settings

Batch size 131072 Minibatch size 32768
Learning rate 0.0001 LR annealing False
LR decay rate 1.5e-4 LR decay floor 0.2
Update epochs 4 L2 regularization coef. 0.0
GAE lambda 0.2 Discount factor (γ) 0.98
Clipping coefficient 0.01 Value function coefficient 1.2
Clip value loss True Value loss clip coefficient 0.2
Max gradient norm 10.0 Entropy coefficient 0.0
Discriminator coefficient 5.0 Bound coefficient 10.0

Table A.3:Policy training hyperparameters for humanoid control.

A.5 Image Reward Fine-tuning

We explore fine-tuning a pre-trained image diffusion model on a non-differentiable task using the
JPEG image compression gym proposed in DDPO [ 54 ]. We report this experiment as a negative
result for FPO, due to the difficulty of fine-tuning diffusion models on their own output. Specifically,
we find that repeatedly generating samples from a text-to-image diffusion model and training on them
is highly unstable, even with manually-specified uniform advantages. We believe that this is related
to classifier-free guidance (CFG) [ 84 ]. CFG is necessary to generate realistic images, however it
is sensitive to hyperparameters, where too much or too little guidance introduces artifacts such as
blur or oversaturation that do not reflect the original training data. Sometimes these artifacts are not
visible to human eyes. These artifacts are further amplified over successive iterations of RL epochs,
ultimately dominating the training signal.

Figure A.3:Image Generation at Different Training Steps.We generate images using Stable Diffusion 1.5
finetuned with FPO as training progresses. We manually set all advantages to 1 to eliminate the reward signal
and investigate the dynamics of sampling from a text-to-image diffusion model then training on the results in
a loop. In the top row, we display images from a training run using a classifier-free guidance (CFG) scale of

    In the bottom row, we display images from a training run using a CFG scale of 2. Low CFG scales tend to
    encourage bluriness while high CFG scales encourage saturation and sharp geometric artifacts. Both diverge
    after a few hundred epochs even with tuned hyperparameters.

This phenomenon aligns with challenges previously identified in the literature on fine-tuning genera-
tive models on their own outputs [ 80 – 82 ]. To illustrate this, we fine-tune Stable Diffusion with all
advantages set to 1 to eliminate the reward signal. This is equivalent to fine-tuning on self-generation
data in an online manner. We explore CFG scales of 2 and 4 in Figure A.3. We find that both CFG
scales induce quality regression. Specifically, the CFG scale of 2 makes the generation more blurry,
while the scale of 2 causes the generated images to feature high saturation and geometry patterns.
Both eventually diverge to abstract geometric patterns.

