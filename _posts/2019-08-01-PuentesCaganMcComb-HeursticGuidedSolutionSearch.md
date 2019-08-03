---
layout: post
type: journal
title: "Heuristic-Guided Solution Search through a Two-Tiered Design Grammar"
authors: ["L. Puentes", "J. Cagan", "C. McComb"]
venue: Journal of Computing and Information Science in Engineering
year: 2019
accepted: true
---
Grammar-based design is typically a gradual process; incremental design changes are performed until a problem statement has been satisfied. While they offer an effective means for searching a design space, standard grammars risk being computationally costly because of the iteration required, and the larger a given grammar the broader the search required. This paper proposes a two-tiered design grammar that enhances computational design generation with generalized heuristics in order to provide a way to more efficiently search a design space. Specifically, this two-tiered grammar captures a combination of heuristic-based strategic actions (often observed in human designers) and smaller-scale modifications (common in traditional grammars). Rules in the higher-tier are abstract and applicable across multiple design domains. Through associated guiding heuristics, these macro rules are translated down into a sequence of domain-specific, lower-tier micro rules. This grammar is evaluated through an implementation within an agent-based simulated annealing team algorithm in which agents iteratively select actions from either the higher-tier or lower-tier. This algorithm is used in two applications: truss generation, which is commonly used for testing engineering design methods, and wave energy converter design generation, which is currently a relevant research area in sustainable energy production. Comparisons are made between designs generated using only lower-tier rules and those generated using only higher-tier rules. Further tests demonstrate the efficacy of applying a combination of both lower-tier and higher-tier rules.
