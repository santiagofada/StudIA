import random
from math import ceil
from datetime import datetime, timedelta
from copy import deepcopy
from collections import defaultdict
from config_loader import CFG

# ------------------------------------------------------------------
# helpers globales
# ------------------------------------------------------------------
BASE_DATE = datetime.fromisoformat(CFG["base_date"])          # lunes de la semana
BLOCK_MIN = CFG["minutes_per_block"]                          # tamaño del bloque

def _to_dt(day: str, hour: str) -> datetime:
    dias = ["lunes","martes","miercoles","jueves","viernes",
            "sabado","domingo"]
    idx = dias.index(day.lower())
    h, m = map(int, hour.split(":"))
    return BASE_DATE + timedelta(days=idx, hours=h, minutes=m)

# ------------------------------------------------------------------
# Individuo = un horario semanal
# ------------------------------------------------------------------
class Individuo(dict):
    """
    Dict[(day,hour)] -> str | None
    Eventos fijos guardados en self.fixed_slots
    """

    def __init__(self, slots, events, tasks):
        # todos los bloques libres
        super().__init__({s: None for s in slots})
        self.fixed_slots = set()                         # bloques que no se mueven

        # coloca eventos multibloque
        seq  = sorted(slots, key=lambda s: (s[0], s[1]))
        idx  = {s: i for i, s in enumerate(seq)}

        for ev in events:
            if (ev.day.lower(), ev.start) not in idx:
                raise ValueError(f"Evento '{ev.name}' inicia fuera de disponibilidad")
            start_i = idx[(ev.day.lower(), ev.start)]
            blocks  = ceil(ev.duration_min / BLOCK_MIN)
            for k in range(blocks):
                slot = seq[start_i + k]
                self[slot] = ev.name
                self.fixed_slots.add(slot)

        # rellena tareas aleatoriamente en los huecos
        self._fill_random(tasks)

    # --------------- helpers internos ---------------- #
    def _fill_random(self, tasks):
        libres = [s for s,a in self.items() if a is None]
        random.shuffle(libres)
        for t in tasks:
            for _ in range(t.duration):
                if libres:
                    self[libres.pop()] = t.name

    # --------------- función de fitness -------------- #
    def fitness(self, tasks, events, prefs, slots):
        c = CFG
        score = 0

        dias_semana = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
        trabajo_por_dia = defaultdict(int)
        actividad_por_dia = defaultdict(lambda: defaultdict(int))

        # --------------------- TAREAS -----------------------
        for task in tasks:
            bloques = sum(1 for a in self.values() if a == task.name)

            # Recompensa por completar
            if bloques >= task.duration:
                score += c["task_complete_reward"]
            else:
                score += c["task_missing_penalty"] * (task.duration - bloques)

            # Penalización por bloques después del deadline
            if task.deadline:
                dl = datetime.strptime(task.deadline, "%Y-%m-%d %H:%M")
                tarde = sum(1 for (d, h), a in self.items()
                            if a == task.name and _to_dt(d, h) > dl)
                score += c["deadline_late_penalty"] * tarde

            # Penalización por mal día si es tarea "Trabajo Lunes", etc.
            nombre = task.name.lower()
            for (d, _), a in self.items():
                if a == task.name and "trabajo" in nombre:
                    for dia in dias_semana:
                        if dia in nombre and d != dia:
                            score += c["wrong_day_penalty"]
                            break

            # Penalización por mala distribución semanal
            bloques_dia = defaultdict(int)
            for (d, _), a in self.items():
                if a == task.name:
                    bloques_dia[d] += 1
            ideal = (bloques + len(dias_semana) - 1) // len(dias_semana)
            for b in bloques_dia.values():
                extra = max(0, b - ideal)
                score += c["task_balance_penalty_per_extra_block"] * extra

        # --------------------- PREFERENCIAS Y DISPONIBILIDAD -----------------------
        score += sum(prefs.get(s, c["base_pref"]) for s, a in self.items() if a)
        score += c["out_of_bounds_penalty"] * sum(
            1 for s, a in self.items() if a and s not in slots
        )

        # --------------------- CONFLICTOS CON EVENTOS -----------------------
        ev_map = {(ev.day, ev.start): ev.name for ev in events}
        score += c["conflict_penalty"] * sum(
            1 for s, a in self.items() if s in ev_map and a != ev_map[s]
        )

        # --------------------- BLOQUES DE TRABAJO DIARIOS -----------------------
        for (d, _), a in self.items():
            actividad_por_dia[d][a] += 1
            if a and a.lower().startswith("trabajo"):
                trabajo_por_dia[d] += 1

        for d in dias_semana:
            bloques = trabajo_por_dia[d]
            faltan = max(0, c["work_required_blocks_per_day"] - bloques)
            extra = max(0, bloques - c["max_work_blocks_per_day"])
            score += c["work_penalty_per_missing_block"] * faltan
            score += c["work_penalty_per_extra_block"] * extra

        # --------------------- PENALIZACIONES DE CALIDAD DE VIDA -----------------------
        for (d, h), a in self.items():
            if a and a.lower().startswith("trabajo"):
                if d in ["sabado", "domingo"]:
                    score += c["weekend_work_penalty_per_block"]
                hour = int(h.split(":")[0])
                if hour < 6:
                    score += c["night_work_penalty"]

        # --------------------- CONTIGÜIDAD Y FRAGMENTACIÓN -----------------------
        prev = None
        for _, a in sorted(self.items()):
            if prev and a != prev:
                score += c["fragmentation_penalty"]
            elif prev == a:
                score += c["contiguity_bonus"]
            prev = a

        return score

    # ---------------- operadores GA ------------------ #
    def mutate(self, prob: float = 0.1):
        if random.random() >= prob:
            return
        movibles = [s for s in self if s not in self.fixed_slots]
        if len(movibles) < 2:
            return
        a, b = random.sample(movibles, 2)
        self[a], self[b] = self[b], self[a]

    @staticmethod
    def crossover(p1, p2):
        """one–point crossover que respeta bloques fijos"""
        idx = random.randint(1, len(p1) - 2)
        keys = list(p1.keys())

        c1 = deepcopy(p1)
        c2 = deepcopy(p2)
        for i, k in enumerate(keys):
            if k in p1.fixed_slots:            # nunca intercambia eventos
                continue
            if i >= idx:
                c1[k], c2[k] = c2[k], c1[k]
        return c1, c2

# ------------------------------------------------------------------
# Motor del algoritmo genético
# ------------------------------------------------------------------
class GeneticAlgorithm:
    def __init__(self, tasks, events, slots, prefs,
                 pop_size=60, generations=300,
                 cx_rate=0.8, mut_rate=0.1, elite=10, seed=None):

        self.tasks, self.events = tasks, events
        self.slots, self.prefs = slots, prefs
        self.pop_size, self.generations = pop_size, generations
        self.cx_rate, self.mut_rate, self.elite = cx_rate, mut_rate, elite

        if seed is not None:
            random.seed(seed)

        self.population = [
            Individuo(slots, events, tasks) for _ in range(pop_size)
        ]

    def run(self, verbose=False):
        for gen in range(self.generations):
            scored = [(ind, ind.fitness(self.tasks, self.events,
                                        self.prefs, self.slots))
                      for ind in self.population]
            scored.sort(key=lambda x: x[1], reverse=True)

            if verbose and gen % 20 == 0:
                best_fit = scored[0][1]
                avg_fit = sum(s for _, s in scored) / len(scored)
                print(f"Gen {gen:3d} | Best: {best_fit:.1f} | Avg: {avg_fit:.1f}")

            new_pop = [deepcopy(ind) for ind, _ in scored[: self.elite]]

            while len(new_pop) < self.pop_size:
                p1, p2 = random.choices(scored[:30], k=2)
                if random.random() < self.cx_rate:
                    c1, c2 = Individuo.crossover(p1[0], p2[0])
                else:
                    c1, c2 = deepcopy(p1[0]), deepcopy(p2[0])
                c1.mutate(self.mut_rate)
                c2.mutate(self.mut_rate)
                new_pop.extend([c1, c2])

            self.population = new_pop[: self.pop_size]

        best = max(self.population,
                   key=lambda ind: ind.fitness(self.tasks, self.events,
                                               self.prefs, self.slots))
        return best
