---
type: other
style: "Technical Presentation"
layout: pub
title: "Design Repository Effectiveness for 3D Convolutional Neural Networks: Application to Additive Manufacturing"
authors: ["G. Williams", "N. A. Meisel", "T. W. Simpson", "C. McComb"]
venue: 30th Annual International Solid Freeform Fabrication Symposium
year: 2019
accepted: true
---
Designing for additive manufacturing (AM) challenges traditional design understanding by introducing new freedoms of complexity. Unfortunately, capitalizing on these complexities can involve time-consuming steps, such as analyzing build metrics during iterative modeling. Machine learning can leverage design repositories, such as GrabCAD and Thingiverse, to automate some of these tedious tasks. However, determining the suitability of an AM design repository for use with machine learning is challenging. We provide an initial investigation towards a solution by using artificial design repositories to test how altering dataset properties impacts trained neural network precision. For this experiment, we use a 3D convolutional neural network to estimate build metrics directly from voxel-based geometries. We focus on material extrusion AM and investigate three AM build metrics: part mass, support material mass, and build time. Our results suggest that training on repositories with less standardized positioning increased neural network accuracy and that estimating orientation-dependent metrics was hardest.
