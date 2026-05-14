# Example session

An illustrative transcript of the HNOCA organoid agent (`python -m agent.agent`,
or `make demo`). The user asks in plain English; the agent maps the request to
a concrete (protocol, age) pair, calls a tool that answers from **real HNOCA
cells**, and interprets the result. All numbers come from
`data/hnoca_dt_subset.h5ad` — the LLM only narrates them.

> Setup for this run: 4,430 cells, Dorsal-telencephalon subset of HNOCA
> (Zenodo 14161275); 15 protocols spanning days 15–300.

---

**You:** What cells dominate a Velasco day-100 cortical organoid?

**Agent:** I'll look at real Velasco cells in the day 80–120 window.

```
  -> calling query_composition({"protocol": "Velasco", "age_min": 80, "age_max": 120})
```

A Velasco day-80–120 cortical organoid in HNOCA is **overwhelmingly dorsal
telencephalic neurons** (≈91% by coarse label):

| Coarse class | Fraction |
|---|---|
| **Dorsal Telencephalic Neuron** | **90.7%** |
| Dorsal Telencephalic NPC | 7.7% |
| Dorsal Telencephalic IP | 1.7% |

Inside the neuronal population, three cortical pyramidal sub-identities make up
the bulk:

| Cell type (fine label) | Fraction |
|---|---|
| cerebral cortex pyramidal neuron | **43.3%** |
| extratelencephalic-projecting glutamatergic cortical neuron | **23.1%** |
| pyramidal neuron | 18.9% |
| neuroblast | 10.8% |
| radial glial cell | 3.9% |

This is the expected outcome for a mid-stage Velasco-protocol cortical organoid
— most progenitors have exited the cell cycle, deep-layer (extratelencephalic /
subcerebral-projecting) cortical neurons are well represented, and only a
residual ~4% radial glia pool remains. The dataset has 889 cells across 42
distinct organoids in this window, so the composition is reasonably stable.

---

**You:** How does NEUROD6 change with organoid age?

**Agent:** NEUROD6 is a transient cortical-neurogenesis marker — let me check
how it tracks age across protocols.

```
  -> calling gene_expression_timecourse({"gene": "NEUROD6"})
```

Across the whole subset (4,430 cells, all protocols pooled), **NEUROD6 mean
log1p expression** by age bin:

| Age bin | NEUROD6 mean log1p | n cells |
|---|---|---|
| 15–50 d | 0.41 | 1,191 |
| 50–86 d | 0.95 | 863 |
| **86–121 d** | **1.63** | 1,169 |
| 121–157 d | 0.95 | 480 |
| 157–193 d | 0.97 | 631 |
| 264–300 d | 0.33 | 96 |

NEUROD6 peaks around days 86–121 (the heart of the cortical neurogenesis
window) and falls in older organoids as those neurons mature and switch to
post-neurogenic markers. This is the classic transient bHLH expression pattern
you'd expect.

---

**You:** Compare what you'd predict for a Velasco day-60 vs day-180 organoid.

```
  -> calling predict_composition_at_age({"protocol": "Velasco", "target_age_days": 60})
  -> calling predict_composition_at_age({"protocol": "Velasco", "target_age_days": 180})
```

| | Day 60 (predicted) | Day 180 (predicted) |
|---|---|---|
| Dorsal Telencephalic NPC | high | low |
| Dorsal Telencephalic Neuron | rising | dominant |
| cerebral cortex pyramidal neuron | minor | majority |

By day 60 the population is still progenitor-rich; by day 180 it's mostly
post-mitotic cortical pyramidal neurons. Both predictions come from real
HNOCA Velasco cells at the nearest available ages, so they are interpolations,
not extrapolations.

---

Want to see it live? `make demo` runs the agent on the first example above.
Other things to try:

- *Which protocols look most like a Velasco day-100 organoid in the atlas?*
- *Compare DCX expression in Velasco vs Lancaster organoids.*
- *Where in the atlas does a Pasca day-30 organoid sit?*
