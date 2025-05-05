from genetico import GeneticAlgorithm
from data_loader import load_data
from collections import defaultdict
from datetime import datetime, timedelta
from config_loader import CFG

STEP_MIN = CFG["minutes_per_block"]          
FMT = "%H:%M"

def _add_minutes(hhmm: str, minutes: int) -> str:
    dt = datetime.strptime(hhmm, FMT) + timedelta(minutes=minutes)
    return dt.strftime(FMT)

def pretty_print_schedule(schedule):
    week = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
    agenda = {d: [] for d in week}
    for (d, h), act in schedule.items():
        if act:
            agenda[d].append((h, act))

    for day in week:
        bloques = sorted(agenda[day], key=lambda x: x[0])
        if not bloques:
            continue

        print(f"\n{day.capitalize():=^40}")  # ======= Lunes =======

        # Para llevar cuenta del tiempo por actividad
        time_by_activity = defaultdict(int)

        i = 0
        while i < len(bloques):
            start, act = bloques[i]
            end = _add_minutes(start, STEP_MIN)
            duration = STEP_MIN

            while (
                i + 1 < len(bloques)
                and bloques[i + 1][0] == end
                and bloques[i + 1][1] == act
            ):
                end = _add_minutes(end, STEP_MIN)
                duration += STEP_MIN
                i += 1

            print(f"  {start} – {end}  |  {act}")
            time_by_activity[act] += duration
            i += 1

        # Mostrar resumen por actividad
        print("\n  -- Resumen del día --")
        for act, total in sorted(time_by_activity.items(), key=lambda x: -x[1]):
            horas = total // 60
            minutos = total % 60
            dur = f"{horas}h" if minutos == 0 else f"{horas}:{minutos}" if horas else f"{minutos}min"
            print(f"  {act:<20} → {dur}")



def main():
    tasks, events, slots, prefs = load_data("data")

    pop_size = 200  # más diversidad
    generations = 1000  # más tiempo de evolución
    cx_rate = 0.95  # más recombinación para explorar mejor
    mut_rate = 0.05  # mantenelo moderado (o subir a 0.1 si hay estancamiento)
    elite = 10  # preservar más élite mejora estabilidad


    ga = GeneticAlgorithm(
        tasks, events, slots, prefs,
       pop_size=pop_size, generations=generations, cx_rate=cx_rate, mut_rate=mut_rate,elite=elite
    )

    best = ga.run(verbose=True)

    print("\n=== Best schedule ===")
    pretty_print_schedule(best)

if __name__ == "__main__":
    main()
