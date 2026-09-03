"""
Our V5 - Vision-Action Episode Representation

Episode-level embedding that fuses:
- observation.images.top (top camera)
- observation.images.wrist (wrist camera)
- action trajectory

Output: single episode-level embedding vector.
"""