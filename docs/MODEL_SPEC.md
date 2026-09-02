# WorldFlow Model Specification

## Status

This is the target model contract, not current implementation authorization.
Learned-model work begins only when the active execution plan permits it.

## World state

The physical metric field is world-aligned on the X-Z plane. Its initial
channels are `O` occupancy, `H` height, `V` visibility/observation state, and
`A` information age. The initial resolution is 128x128 covering 10m x 10m
(0.078125m/cell). A learned visual latent `Z` may be appended later.

`A` is a deterministic memory state: it is updated from observation/visibility
events and elapsed time, not a primary neural prediction target. A model may
consume `A` as context, while evaluation focuses learned prediction capacity on
physical geometry, visibility/revelation, and `Z` when authorized.

## Computation graph

`RGB -> frozen DINOv3 dense features`

`Depth + pose -> geometry-aware lifting -> persistent metric latent field Phi_t`

`hypothetical u_(t:t+H) = [(v, omega, dt)] -> kinematic integration -> future robot poses`

`Phi_t + integrated future poses -> deterministic SE(2) world-aligned crop transport -> transported state`

`transported state + current context + actions -> learned field update/revelation`

`-> Phi_(t+1) -> autoregressive Phi_(t+1:t+H)`

The hypothetical action is a continuous `(v, omega, dt)` sequence, not a
discrete turn label. Its kinematic integration yields the future SE(2) robot
poses that drive field/crop transport. In a globally world-aligned tensor, ego
rotation affects the integrated trajectory; the regridded crop uses the induced
world displacement rather than an artificial tensor rotation.

## Non-negotiable properties

- The output is a future world-state rollout, not a scalar score.
- Future RGB/depth cannot be used to construct a prediction at inference time.
- DINOv3 stays frozen by default and is not the claimed contribution.
- The learned term is evaluated against M0-Transport, M1-Direct, and M2-RSSM.
