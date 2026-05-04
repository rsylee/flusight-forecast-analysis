# FluSight Forecast Analysis

Forecast evaluation and analysis for CDC FluSight 2025 flu and RSV hospitalization predictions (UM-DeepOutbreak model).

## Overview

This repo contains scripts for evaluating and visualizing forecast submissions against CDC ground truth. It also includes two post-processing methods — epimodulation damping (applied per horizon via grid search) and ensemble learning (simple averaging, weighted averaging by inverse error, and LR/SVR stacking) — to improve forecast accuracy.

## Requirements

```bash
pip install pandas numpy matplotlib epiweeks
```
