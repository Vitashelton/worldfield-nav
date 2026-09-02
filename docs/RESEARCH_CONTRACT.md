# Research Contract

## Working title

**WorldFlow: Action-Conditioned Persistent Latent World-Field Evolution for
Indoor Mobile Robot Navigation**

## Scientific question

Can an indoor mobile robot maintain a persistent metric latent world state and
imagine how that world would evolve under hypothetical future robot actions?

## State, input, and output

`Phi_t = [geometry, visibility, memory, latent representation]`

Input: the current persistent world field and a hypothetical future robot
action sequence. Output: the future persistent world-field rollout
`Phi_(t+1:t+H)`.

## Core method

1. Frozen DINOv3 visual representation.
2. Geometry-aware metric lifting.
3. Persistent latent world field.
4. Deterministic ego-motion transport.
5. Learned action-conditioned field update/revelation.
6. Autoregressive future rollout.

## Not the paper

- Scalar trajectory scoring.
- Human trajectory forecasting or autonomous driving.
- Pure occupancy forecasting or visual video generation.
- Habitat-GS itself.

## Downstream task and final evidence

The downstream task is indoor mobile-robot predictive navigation. Final paper
evidence must cover a controlled Habitat-GS benchmark, unseen-scene prediction,
counterfactual world imagination, closed-loop navigation, and Ranger Mini
real-robot validation.
