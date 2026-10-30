# Publication topic audit — October 30, 2026

## Scope and method

This review covers the **live** `cmccomb.com` publication map snapshot built at
`2026-10-29T23:45:30Z`: 267 records from `ccm/publications` commit
`0359eb8a8a1ca9dae4e40d43057e63c1b5f60cdb`. The live JSON matched the
committed `assets/json/pubs.json` at the start of this audit. I read all 267
titles and the abstracts for the reinforcement-learning examples. Counts below
are descriptive checks of that snapshot, not a validated topic-classification
score.

A displayed **topic** is the name of an *exclusive cluster*: each paper receives
one `cluster_id`, and an automatically selected phrase names the whole cluster.
It is not an independently verified tag on every paper. The phrase selector
requires source-phrase support from only 20% of the cluster's papers (at least
two). Thus a valid extractive phrase can still misdescribe most members. The
map's label is placed near a cluster centroid; papers from other clusters can
surround it.

“Near 10” below counts how many of the ten papers closest to the stored
cluster centroid in the **unadjusted two-dimensional projection** belong to
that cluster. It audits what a label is likely to appear near, rather than
semantic accuracy. The browser's collision layout can move circles and labels,
so the rendered result also needs visual inspection.

| Displayed topic | Papers | Near 10 | Representative concern from title review |
| --- | ---: | ---: | --- |
| First-year engineering | 17 | 10 | Mostly coherent; “Engineering Futures” and “Research and Practice Group Methodology” are broader education work. |
| Human behavior | 14 | 10 | Includes “Toward a Time-Aware Assessment Framework for the Carbon Cost of AI-Enabled Decarbonization” and a manufacturing robot RL paper. |
| Artificial intelligence | 21 | 10 | Broad AI/human-teaming region; raw student truss and cooling-system design datasets do not obviously warrant this topic. |
| Large language models | 18 | 8 | Two hidden-Markov/neuroimaging design-ideation papers are far from the rest and are not LLM studies. |
| Reinforcement learning | 9 | **0** | Four of nine papers lack an RL method or object of study; details below. |
| Super-resolution | 16 | 10 | Includes “GARCH-Informed Neural Networks for Volatility Prediction in Financial Markets” and the KeepDelta software paper. |
| Strain fields | 12 | 10 | Label is narrower than the region: several papers concern additive-manufacturing thermal modeling or process security, without strain-field analysis. |
| Pump designs | 17 | **4** | Only four titles concern pumps; other members cover Arrow's theorem, VR/CAD evaluation, composite decks, construction, ethics, and supply chains. |
| Additive manufacturing | 29 | 10 | Mostly coherent, but AircraftVerse and MSEval material-selection datasets do not appear to be additive-manufacturing papers. |
| Permafrost thaw | 13 | 10 | Mostly Arctic/permafrost work, but includes a wave-energy-converter study and a financial-market simulation. |
| Trait empathy | 20 | 8 | Many are general prototyping/team papers; the manufacturing-fixation paper is especially far from the centroid. |
| Design teams | 33 | 10 | Mostly coherent across empirical and simulated teams; some methods papers are broader than the displayed phrase. |
| Cognitive assistants | 20 | 8 | Only a small subset directly concerns cognitive assistants; others cover NLP, startup landscape, AI adoption, and design-computing taxonomy. |
| Human designers | 28 | 10 | Coherent design-agent core, but includes a generative-AI special issue, LLM design work, and other papers that do not focus on human designers. |

The strongest overall label problems are **reinforcement learning** and **pump
designs**. Narrower names such as **trait empathy**, **strain fields**, and
**cognitive assistants** also risk implying that every member studies the named
subtopic. Proximity alone cannot establish topical relevance: “strain fields”
has all ten nearest papers from its own cluster, yet several do not study
strain fields.

## Reinforcement learning: paper-level check

Five of its nine members have clear RL evidence in title or abstract:

| Scholar publication ID suffix | Evidence |
| --- | --- |
| `AvfA0Oy_GE0C` | “Reinforcement Learning for Efficient Design Space Exploration…” |
| `_Re3VWB3Y0AC` | Abstract studies visualizations of RL agents' search trajectories. |
| `ClCfbGk0d_YC` | “Symmetry Heuristics for Stable Reinforcement Learning Design Agents.” |
| `sNmaIFBj_lkC` | Abstract describes a multi-fidelity RL policy-alignment framework. |
| `8xutWZnSdmoC` | A later publication version of the policy-alignment RL work. |

Four members have no RL evidence in title or abstract and describe different
work:

| Scholar publication ID suffix | Title / actual subject |
| --- | --- |
| `OU6Ihb5iCvQC` | “Stochastic Stackelberg Games for Agent-Driven Robust Design” — game-theoretic simulated-annealing design. |
| `BrmTIyaxlBUC` | “An Adversarial Agent-Based Design Method Using Stochastic Stackelberg Game Conditions” — game-theoretic adversarial design. |
| `tzM49s52ZIMC` | “Assessing Machine Learnability Of Image And Graph Representations For Drone Performance Prediction” — GNN/CNN performance prediction. |
| `6ZxmRoH8BuwC` | “HUVER: The HyForm Uncrewed Vehicle Engineering Repository” — a multimodal UAV design dataset. |

The two Stackelberg papers occupy a separate island around `(0.6, 10.9)`, while
the principal RL papers lie around `y = -8`. The RL cluster spans `y = -8.1`
to `10.9`; its mean paper-to-centroid distance is 7.5 map units, versus about
1.2–3.2 for most other clusters. Its ten closest papers to the centroid are
**all from other clusters**, mainly large-language-model work. The label's
position is therefore misleading even before browser collision adjustment.

Four other papers explicitly mention RL in title or abstract but are assigned
elsewhere:

| Scholar publication ID suffix | Current topic | RL evidence |
| --- | --- | --- |
| `t6usbXjVLHcC` | Human behavior | “A Multi-Agent Reinforcement Learning Framework for Intelligent Manufacturing with Autonomous Mobile Robots”; abstract reports RL use. |
| `xtoqd-5pKcoC` | Human behavior | Abstract proposes hierarchical imitation and reinforcement learning for human-behavior modeling. |
| `Ug5p-4gJ2f0C` | Human behavior | Abstract examines multi-agent RL control frameworks for manufacturing robots. |
| `HIFyuExEbWQC` | Human designers | “Human-Machine Collaboration with Reinforcement Learning for Explaining Manual Layup Strategies…” |

These omissions show why simply deleting the RL label or suppressing outlying
papers would lose legitimate RL work. A paper can reasonably belong to more
than one research subject; exclusive cluster names cannot express that.

## Cause and correction criteria

The builder seeds ten clusters in embedding PCA space, then splits a parent
only when it exceeds `floor(267 / 8) = 33` papers. A 25% projection-distance
term applies **only during those size-driven splits**. The nine-paper RL child
is below the cap, so its disconnected visual islands are not examined again.
The phrase scorer can then choose “reinforcement learning” from the five
RL-supported members and the browser displays that phrase for all nine.

Add a second split trigger based on a cluster's *separation in the displayed
projection*, with minimum child support and a check against original-embedding
semantic coherence. Evaluate whether each resulting group has a name supported
by its papers; do not split solely to improve centroid appearance. For precise
subject filtering, derive paper-level, evidence-backed tags separately from
the exclusive visual clusters. An RL tag should include the four relevant
papers now assigned elsewhere and exclude the four unrelated members above.

After rebuilding, review all labels and their member titles, the ten-paper
centroid neighborhoods, any very small or scattered groups, and the rendered
map at desktop and narrow widths. These are release checks for the next
snapshot, not claims that a two-dimensional projection can certify topical
similarity.

## Response in the proposed 0.2.18 snapshot

The proximity check splits the two Stackelberg works into an automatically
named **stochastic stackelberg game** neighborhood. All 267 publication IDs,
metadata, and projected coordinates remain identical. The RL neighborhood
shrinks from nine to seven papers; five of those seven have clear RL evidence,
while the two drone-design works remain. Its centroid's nearest ten papers now
include four members (and one other RL paper assigned to **human designers**),
up from zero members in the live 0.2.17 snapshot. This is a substantial label
placement improvement, not complete subject classification.

The browser now describes a paper's assigned label as its **map neighborhood**
and does not insert that label into the paper's search index. A search for
“reinforcement learning” therefore uses publication text and excludes the
Stackelberg and drone works while finding RL evidence in other neighborhoods.
The remaining mixed labels in the table above are still review findings; this
release does not claim to correct every individual subject assignment.

Global semantic diagnostics move slightly downward: original-space cosine
silhouette changes from 0.1180 to 0.1164 and nearest-ten same-neighborhood
agreement from 61.57% to 61.39%. Projection trustworthiness and paper positions
are unchanged. The split is justified by the specific label-placement and
membership correction, not by a claim of general classification improvement.
