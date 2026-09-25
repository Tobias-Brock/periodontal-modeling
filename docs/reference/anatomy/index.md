# periomod.anatomy Overview

The `periomod.anatomy` module holds the anatomical structure of the periodontal chart. It defines which teeth are adjacent within a dental arch, which teeth occlude and how the sites of adjacent teeth meet at an interproximal contact. Both `periomod.graph` and `periomod.bayes` build their relations from these functions, so the message passing of the graph neural network and the spatial priors of the Bayesian models share one definition of adjacency.

## Available Components

| Component              | Description                                                         |
|------------------------|---------------------------------------------------------------------|
| [get_arch_neighbors](get_arch_neighbors.md)         | Adjacent teeth per tooth within an arch.          |
| [get_arch_pairs](get_arch_pairs.md)                 | Pairs of adjacent teeth of both arches.           |
| [get_occlusal_pairs](get_occlusal_pairs.md)         | Pairs of occluding teeth.                         |
| [is_mesial_neighbor](is_mesial_neighbor.md)         | Orientation of a neighboring tooth.               |
| [is_midline_pair](is_midline_pair.md)               | Central incisors meeting across the midline.      |
| [get_interproximal_pairs](get_interproximal_pairs.md) | Interproximal site pairs of adjacent teeth.     |
