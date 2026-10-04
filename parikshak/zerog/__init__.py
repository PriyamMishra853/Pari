"""Zero-G rack-centric perception: real 3D pose, AprilTag rack frame, body mesh,
inclination, interaction graph and temporal activity recognition.

Everything this package reports is measured from the camera frame it is given.
When a measurement cannot be made (no tags, no person, no model) it says so -
it never fills the gap with a plausible-looking constant.
"""
