# Image-Goal Localization Protocol

The primary task is locating and navigating to the physical place shown in a
user-provided goal image. This is distinct from estimating the robot's current
pose from its current RGB image.

## Information boundaries

- Offline training and reference-database construction may use RGB-D, depth,
  absolute pose, and LIO geometry.
- Query-time retrieval uses only the goal RGB descriptor in the primary result.
  It must not read the goal image's hidden global pose or global LIO pose.
- Hidden goal pose is evaluation-only.
- After retrieval, Nav2 may use the robot's current LIO pose to execute the
  retrieved goal pose.
- Logs must keep `robot_pose_for_control`, `retrieved_goal_pose`, and
  `hidden_goal_pose_for_eval_only` separate.

## Main comparison

Frozen DINOv3, Vanilla Adapter, MetricAnchor, and MetricAnchor-Full use the
same reference index, keyframe density, candidate set, and navigation planner.
Primary metrics are RGB-only retrieval and downstream navigation. VLM is an
optional semantic filter and must receive identical outputs across methods; it
cannot provide or infer world coordinates and is excluded from primary metrics.

The paper claim is geometry-assisted persistent visual place memory, not a
replacement for mature visual-inertial/SLAM relocalization.
