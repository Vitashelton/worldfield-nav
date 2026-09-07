# AGENTS.md — ExecNav Research Harness

Read `docs/RESEARCH_CONTRACT.md`, `docs/MODEL_SPEC.md`,
`docs/BENCHMARK_SPEC.md`, then the single plan under
`docs/exec-plans/active/`. The active paper direction is ExecNav: VLM task
parsing plus a learned executability critic for image-goal navigation.

WorldFlow, C1 and GeoAnchor/G1/G2 are preliminary infrastructure; do not delete
or reinterpret their assets. The only learned component authorized now is the
small critic trained from privileged Habitat outcomes. VLM is a replaceable,
cached or remote task parser and cannot emit coordinates or low-level actions.

Use Habitat ImageNav/NavMesh as the unchanged executor. Keep simulator teacher
labels separate from online inputs. Compare semantic-free, VLM-only,
handcrafted-feasibility, and critic+recovery methods under identical episodes.
Dynamic avatars are a labelled stress subset. Do not start real robot, ROS2,
LLM API integration, new datasets, or new research directions.

Raw outputs belong in `outputs/formal/ExecNav/`; only paper-ready tables,
figures and videos belong in `paper_assets/`. Every formal run records config,
seed, metrics and failure cases. Do not claim novelty beyond the measured
executability-calibration contribution.
