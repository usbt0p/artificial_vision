# Computer Vision

## Overview

These are all the codes for a subject called "Visión Artificial" (Computer Vision).

- Session [01](../session01/): `Image Transformations and Camera Calibration`

- Session [02](../session02/): `Filtering and Edge Detection`

- Session [03](../session03/): `CNNs and Residual Networks`

- Session [04](../session04/): `Dense Prediction and U-Net`

- Session [05](../session05/): `Fully Convolutional Networks`

- Session [06](../session06/): `R-CNN Object Detection`

- Session [07](../session07/): `Object Tracking`

- Session [08](../session08/): `Video Understanding and Temporal Action Recognition`

- Session [09](../session09/): `Advanced Video Analysis`

- Session [10](../session10/): `Depth Estimation`

- Session [11](../session11/): `3D Scene Understanding`

- Session [12](../session12/): `Visual RL`

- Session [13](../session13/): `Visual Policy Learning`

## Project

Twenty percent of the final grade was determined by a project based on one of the sessions. We chose [session 04](../session04/labCode/) and researched different implementations of the U-Net architecture. We did this in a team of three and had to hand it in by Sunday, the 25th of January of 2026. We were graded 8.5 out of 10. Here is the explanation for the grade:

| <img width="1000"><br><p align="center">Section | <img width="1000"><br><p align="center">Grade |
| :---------------------------------------------- | --------------------------------------------: |
| Delivery and communication                      |                                         14/15 |
| Framing and conceptual rigor                    |                                         13/15 |
| Methods and reproducibility                     |                                         22/25 |
| Results and critical analysis                   |                                         23/30 |
| Complexity/ambition and execution               |                                         13/15 |
| **Total**                                       |                                    **85/100** |

### Project description (what they did)

They studied the role of _skip connections_ in U-Net through ablations (concat/add/attention/none) and capacity variants (“bigger”), analyzing computational cost, stability, and performance (IoU/loss/gradient).

### Overall assessment of the execution

A well-designed and technically consistent project, with strong experimental work; it loses some points due to limited statistical closure and incomplete operational reproducibility in the report.

### Pros

The report is well structured and argues with solid technical maturity why stability, cost, and performance change depending on the type of `skip connection`; the result graphs support these conclusions well.

There is also good methodological ambition for a lab project: they compare several variants, two network depths, and multiple seeds, instead of limiting themselves to a single execution.

### Cons

Reproducibility could be more complete because some more “operational” execution details are missing (exact environment, single reproduction commands, and a more explicit train/val/test separation in the text), although the code itself is fairly complete.

The quantitative analysis would be stronger with final summary tables per variant (mean ± standard deviation across seeds and statistical comparison), since several conclusions rely mainly on visual inspection of curves.
