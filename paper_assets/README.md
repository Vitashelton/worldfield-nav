# Paper assets

This is the headless-server handoff directory. Only presentation-ready
artifacts belong here:

- `tables/`: final CSV tables with one row per method/setting;
- `figures/`: paper-quality PNG/SVG figures;
- `videos/`: short group-meeting or paper videos.

Raw caches, checkpoints, manifests, and per-episode logs remain under
`outputs/formal/` and must not be deleted. Preliminary S0--S2, B1, C1, and G1
assets are retained here with their original names. Formal GeoAnchor/G3
artifacts should use the `metricanchor_` or `g3_` prefix and be referenced by
the result markdown, so a remote-only user can download this directory without
sorting through intermediate files.
