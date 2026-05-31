from . import pso, dbo, coa, koa, cpo, sboa

ALGORITHMS = {
    "PSO":  (pso.optimize,  1995, "Particle Swarm Optimization (baseline)"),
    "DBO":  (dbo.optimize,  2023, "Dung Beetle Optimizer"),
    "COA":  (coa.optimize,  2023, "Crayfish Optimization Algorithm"),
    "KOA":  (koa.optimize,  2023, "Kepler Optimization Algorithm"),
    "CPO":  (cpo.optimize,  2024, "Crested Porcupine Optimizer"),
    "SBOA": (sboa.optimize, 2024, "Secretary Bird Optimization Algorithm"),
}
