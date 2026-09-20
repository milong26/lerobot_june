# Preliminary PPT summary

Valid matched-camera checkpoints used: **24**
Rows excluded because an older quick run used a mismatched camera: **25**

## Slide 1

**Title:** Few-Episode Task Success Is Not a Reliable Checkpoint Ranking Signal

Across the currently valid 24 checkpoints, 5-episode task success shows weak agreement with the historical 200-episode task success (Pearson r = -0.096, Spearman rho = 0.118).

The five-episode estimate changes in 20-percentage-point increments and is highly sensitive to the sampled initial states, so it can mis-rank checkpoints.

**Takeaway:** Raw 5-episode task success is insufficient as a checkpoint filter.

## Slide 2

**Title:** Intermediate Grasp Success Provides a More Informative Early Signal

Using the same matched-camera checkpoints, quick grasp success shows stronger agreement with full-evaluation grasp success (Pearson r = 0.730, Spearman rho = 0.748).

For reference, task-success rank correlation is rho = 0.118, whereas grasp-success rank correlation is rho = 0.748.

**Takeaway:** A useful low-cost surrogate should use intermediate task competence rather than only final success from a few episodes.

## Current status counts

- checkpoint_missing: 22
- not_run_budget: 31
- ok: 49
- timeout: 2
