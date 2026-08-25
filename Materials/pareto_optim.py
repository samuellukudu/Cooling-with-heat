from cooling_physics import find_pareto_frontier, pareto_target_window, simulate_adsorption_cycle


def run_virtual_pareto_screen(
    t_evap_c: float = 7.0,
    t_cond_c: float = 35.0,
    t_des_c: float = 80.0,
    cycle_time_sec: float = 600.0,
    q_sat_steps: int = 25,
    q_st_steps: int = 25,
) -> list[dict]:
    points = []
    for i in range(q_sat_steps):
        q_sat = 0.08 + (0.90 - 0.08) * i / (q_sat_steps - 1)
        for j in range(q_st_steps):
            q_st = 2.30e6 + (4.10e6 - 2.30e6) * j / (q_st_steps - 1)
            cycle = simulate_adsorption_cycle(
                q_sat=q_sat,
                q_st=q_st,
                t_evap_c=t_evap_c,
                t_cond_c=t_cond_c,
                t_des_c=t_des_c,
                cycle_time_sec=cycle_time_sec,
            )
            points.append({"q_sat": q_sat, "Q_st": q_st, **cycle})

    return sorted(find_pareto_frontier(points), key=lambda row: row["COP"])


if __name__ == "__main__":
    frontier = run_virtual_pareto_screen()
    target = pareto_target_window(
        t_evap_c=7.0,
        t_cond_c=35.0,
        t_des_c=80.0,
        cycle_time_sec=600.0,
        cop_weight=0.40,
        scp_weight=0.20,
    )

    print(f"Screening complete. Identified {len(frontier)} Pareto-optimal virtual materials.")
    print(
        "Recommended target window: "
        f"q_sat={target['q_sat_min']:.3f}-{target['q_sat_max']:.3f} kg/kg, "
        f"Q_st={target['Q_st_min'] / 1e6:.3f}-{target['Q_st_max'] / 1e6:.3f} MJ/kg "
        f"from {target['window_points']} target-window points"
    )

    print("\n=========================================================================")
    print("PARETO FRONTIER: STRUCTURAL CRITERIA TO SYSTEM PERFORMANCE")
    print("=========================================================================")
    print(
        f"{'Rank':<5} | {'q_sat (kg/kg)':<14} | {'Q_st (MJ/kg)':<13} | "
        f"{'delta_q (kg/kg)':<15} | {'COP':<6} | {'SCP (W/kg)'}"
    )
    print("-" * 75)

    for rank, row in enumerate(frontier, start=1):
        print(
            f"{rank:<5} | {row['q_sat']:<14.4f} | {row['Q_st'] / 1e6:<13.3f} | "
            f"{row['delta_q']:<15.4f} | {row['COP']:<6.3f} | {row['SCP_W_kg']:.2f}"
        )
