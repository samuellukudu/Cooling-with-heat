# Heat-Driven Cooling Materials Screening

This project is searching for materials that can use heat to create cooling. The most practical first mechanism is an adsorption cooling cycle: a porous solid adsorbs a natural refrigerant such as water at low temperature, then waste heat regenerates the material by desorbing the refrigerant.

## Application Profiles

The same material will not be optimal for every cooling target.

| Application | Heat source | Cooling target | Main material priority |
| --- | --- | --- | --- |
| CPU / electronics | Warm liquid loop, roughly 60-80 C | Cold plate or rack assist | Very high specific cooling power and thermal conductivity |
| Human comfort / HVAC | Solar thermal or waste heat, roughly 70-90 C | 7-12 C chilled water or dehumidification | COP, stability, non-toxicity, low cost |
| Vehicles | Exhaust/coolant waste heat, roughly 90-150 C | Cabin or battery cooling | Compactness, fast cycling, vibration/thermal stability |
| Data centers | Warm water loop, roughly 45-70 C | Facility chilled water or elevated-temperature liquid cooling | Low regeneration temperature, COP, continuous durability |

## What Materials Project Can Screen

Materials Project is useful for early filtering with physics-informed proxy criteria:

- chemical system
- density
- formation energy
- energy above hull
- crystal volume and atoms per cell
- stable inorganic framework candidates

These are not enough to prove adsorption cooling performance. They are first-pass descriptors for stability, openness, and chemistry.

## What Must Be Added Later

To turn the screen into a defensible materials discovery pipeline, add:

- pore limiting diameter, accessible pore volume, and surface area from Zeo++ or similar tools
- water adsorption isotherms from GCMC or experiments
- thermal conductivity from experiment, simulation, or a trained surrogate model
- binder/coating compatibility with aluminum or copper heat exchangers
- cyclic hydrothermal stability tests

## Current Script

Run:

```bash
python heat_cooling_screen.py --top 10
```

With Materials Project, put this in `.env`:

```bash
MP_API_KEY="your_key_here"
```

Then run:

```bash
.venv/bin/python heat_cooling_screen.py --apps datacenter human --top 20
```

The script ranks candidates separately for CPU/electronics, human comfort, vehicle cooling, and data centers. It first runs a Pareto target sweep over hypothetical `q_sat` and `Q_st` values for the application operating temperatures. Then it uses proxy estimates for adsorption capacity, adsorption heat, and thermal conductivity to score real Materials Project candidates against that Pareto-derived target.

By default, `heat_cooling_screen.py` derives its Materials Project search space from each application profile. It generates chemical systems from required/allowed elements, then filters by density and energy-above-hull windows appropriate to that application. The old hand-picked chemistry list is available only as a fallback:

```bash
.venv/bin/python heat_cooling_screen.py --use-fallback-chemsys
```

For a quick smoke test of the generated search, cap the number of generated chemical systems:

```bash
.venv/bin/python heat_cooling_screen.py --apps datacenter --max-generated-chemsys 3 --limit-per-system 5
```

Treat the output as a prioritization list for deeper simulation, not as a final claim that a material works.

`Pscore` in the output is the Pareto-closeness score. A low value means the candidate may be stable and chemically relevant, but its estimated adsorption capacity and adsorption heat are far from the application's preferred physics window.

The output also includes a materials Pareto frontier. This is a postprocessing step after Materials Project retrieval. It keeps candidates that are not dominated across real-material objectives:

- Pareto closeness to the application physics target
- thermodynamic stability
- thermal-conductivity proxy
- open-framework proxy
- low-density score
- non-toxic chemistry score

The weighted ranking gives one prioritized order. The materials Pareto frontier preserves tradeoff candidates that may be scientifically interesting even when they are not the top weighted result.
