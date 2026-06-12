# How Much Policy Does a VLA Need?

**CS295 project by Eric and Sean**

Vision-language-action models combine perception, language understanding, and
robot control. This project asks a narrower question:

> When pretrained components already provide visual and language
> representations, how much capacity is still needed in the policy that turns
> those representations into robot actions?

We studied this question with two model families.

## Experiments

### LAPA Backbone Replacement

We exported LAPA's learned embeddings and latent-action head, froze them, and
replaced its 7B multimodal backbone with Pythia models connected through
trainable projection layers.

| Backbone | Best latent-action accuracy |
| --- | ---: |
| Pythia 160M | 47.1% |
| Pythia 410M | **47.8%** |
| Pythia 1B | 46.9% |

All three models performed well above the 12.5% uniform baseline, but the
differences between sizes were small. Because Pythia is text-pretrained, this
experiment mixed policy capacity with the difficulty of interpreting LAPA's
multimodal representation.

Code: [`LAPA_torch/LAPA`](LAPA_torch/LAPA)

### VLA Foundry Policy Scaling

For the cleaner comparison, we froze the same Foundry VLM for every run and
varied only the depth of the flow-transformer action policy:

- 77M preset: 6 layers
- 205M preset: 12 layers
- 410M preset: 24 layers

All models used a 40,000-sample training budget on one NVIDIA RTX A6000.

| Task | 6 layers | 12 layers | 24 layers |
| --- | ---: | ---: | ---: |
| PickAndPlaceBox MSE | 0.3003 | 0.2802 | **0.2703** |
| PutOrangeOnSaucer MSE | 0.3037 | 0.3017 | **0.2840** |
| PushBox MSE | 0.3089 | 0.3005 | **0.2903** |

Deeper policies consistently improved held-out action prediction. Closed-loop
performance remained poor:

| Task | 6 layers | 12 layers | 24 layers |
| --- | ---: | ---: | ---: |
| PickAndPlaceBox | 2/50 | 3/50 | 4/50 |
| PutOrangeOnSaucer | 0/50 | 0/50 | 0/50 |

Code: [`vla_foundry`](vla_foundry)

Sweep scripts: [`vla_foundry/scripts/sweep`](vla_foundry/scripts/sweep)

## Conclusion

Increasing policy depth improved offline action prediction, but it did not
produce reliable closed-loop control under our training budget. The results
suggest that policy capacity was not the only bottleneck: limited training,
narrow data coverage, and compounding control errors were at least as
important.

We therefore do not conclude that a larger policy is always better. The more
useful question is when representation quality, data coverage, and training
are sufficient for additional policy capacity to become the limiting factor.

Large datasets, environments, and training checkpoints are intentionally
excluded from this repository.
