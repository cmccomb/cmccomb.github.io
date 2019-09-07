---
type: journal
layout: post
title: "Design Repository Effectiveness for 3D Convolutional Neural Networks: Application to Additive Manufacturing"
authors: ["G. Williams", "N. A. Meisel", "T. W. Simpson", "C. McComb"]
venue: Journal of Mechanical Design
year: 2019
accepted: true
asme: 10.1115/1.4044199
---

Machine learning can be used to automate common or time-consuming engineering tasks for which sufficient data already exists. For instance, design repositories can be used to train deep learning algorithms to assess component manufacturability; however, methods to determine the suitability of a design repository for use with machine learning do not exist. We provide an initial investigation towards identifying such a method by using “artificial” design repositories to experimentally test the extent to which altering properties of the dataset impacts the assessment precision and generalizability of neural networks trained on the data. For this experiment, we use a 3D convolutional neural network to estimate quantitative manufacturing metrics directly from voxel-based component geometries. Additive manufacturing (AM) is used as a case study because of the recent growth of AM-focused design repositories such as GrabCAD and Thingiverse that are readily accessible online. In this study, we focus only on material extrusion, the dominant consumer AM process, and investigate three AM build metrics: (1) part mass, (2) support material mass, and (3) build time. Additionally, we compare the convolutional neural network accuracy to that of a baseline multiple linear regression model. Our results suggest that training on design repositories with less standardized orientation and position resulted in more accurate trained neural networks and that orientation-dependent metrics were harder to estimate than orientation-independent metrics. Furthermore, the convolutional neural network was more accurate than the baseline linear regression model for all build metrics.
