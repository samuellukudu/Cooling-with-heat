# diffheat — Application Scenarios

Tracked applications for future simulation with diffheat. Each entry includes the PDE system, boundary conditions, parameter ranges, and target optimization problem.

The library itself remains general-purpose; these are specific compositions of its operators and solvers.

---

## 1. Natural Convection (Boussinesq)

**PDE system:**
- ∂T/∂t + **u**·∇T = α ∇²T (energy)
- ∂**u**/∂t + **u**·∇**u** = -∇p + ν ∇²**u** + β g (T - T_ref) **k** (momentum)
- ∇·**u** = 0 (continuity)

**Parameters:** Rayleigh number (Ra), Prandtl number (Pr), aspect ratio

**Boundary conditions:** Heated bottom plate, cooled top plate, insulated side walls, no-slip velocity on all walls

**Optimization target:** Maximize heat transfer (Nusselt number) for given Ra; optimize heating pattern for uniform cooling

---

## 2. Thermoelectric Cooling (Peltier/Seebeck)

**PDE system:**
- ρ c_p ∂T/∂t = ∇·(k ∇T) + J²/σ - τ J·∇T (heat + Joule + Thomson)
- ∇·(σ ∇V) = -∇·(σ S ∇T) (electric potential with Seebeck source)

**Parameters:** Seebeck coefficient (S), electrical conductivity (σ), thermal conductivity (k), Thomson coefficient (τ)

**Boundary conditions:** Fixed voltage at contacts, convective heat transfer at boundaries

**Optimization target:** Maximize cooling ΔT for given input current; minimize power consumption for target cooling

---

## 3. Absorption Chiller Cycle

**Model type:** Lumped ODE system (not PDE — would require `diffheat` ODE support)

**4-component cycle:** Generator → Condenser → Evaporator → Absorber

**Parameters:** Heat input (Q_gen), cooling output (Q_evap), solution concentrations, mass flow rates

**Optimization target:** Maximize COP (Q_evap / Q_gen); optimize cycle temperatures for given heat source

---

## 4. Thermal Cloak Optimization

**PDE system:** ∂T/∂t = ∇·(κ(x,y) ∇T) (spatially-varying conductivity)

**Goal:** Optimize κ(x,y) field so that a protected interior region experiences minimal temperature gradient while external temperature field appears undisturbed

**Parameters:** Conductivity range [κ_min, κ_max], cloak geometry

**Optimization target:** Minimize |∇T| inside protected region while matching far-field temperature profile

---

## 5. Forced Convection Cooling

**PDE system:**
- ∂T/∂t + **u**·∇T = α ∇²T (advection-diffusion)
- **u**(x,y) prescribed (not solved — one-way coupling)

**Parameters:** Péclet number (Pe = UL/α), channel geometry

**Boundary conditions:** Inlet temperature, convective outlet, heated component at center

**Optimization target:** Optimize inlet velocity profile or channel geometry to minimize component temperature

---

## 6. Irregular Domains

**Status:** Not yet supported. Requires unstructured mesh support (`GridUnstructured`) and finite volume / finite element operator assembly.

**Blockers:**
- Mesh generation (triangulation)
- Unstructured operator stencils (neighbor lookups)
- Boundary condition application on curved edges
