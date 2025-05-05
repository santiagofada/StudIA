import csv
from pathlib import Path
from datetime import datetime, timedelta
from config_loader import CFG
from models import Task, Event

FMT = "%H:%M"
STEP = CFG["minutes_per_block"]

def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = csv.DictReader(f)
        out = []
        for row in rows:
            # si la primera celda es un comentario o la fila está vacía → saltar
            first_val = next(iter(row.values()))
            if first_val is None or first_val.strip().startswith("#"):
                continue
            out.append({k.strip().lower(): (v or "").strip() for k, v in row.items()})
        return out


def _range_hhmm(start, end, step=STEP):
    t = datetime.strptime(start, FMT)
    end_t = datetime.strptime(end, FMT)
    step = timedelta(minutes=step)
    while t < end_t:
        yield t.strftime(FMT)
        t += step

# ------------------------------------------------------------------
def load_disponibilidad(data_dir: str | Path):
    rows = _read(Path(data_dir) / "disponibilidad.csv")
    slots = set()
    for r in rows:
        day = r["dia"].lower()
        for hhmm in _range_hhmm(r["start"], r["end"]):
            slots.add((day, hhmm))
    return slots

def load_preferencias(data_dir: str | Path):
    rows = _read(Path(data_dir) / "preferencias.csv")
    prefs = {}
    for r in rows:
        if not (r["start"] and r["end"] and r["score"]):
            continue  # línea vacía o comentario
        day   = r["dia"].lower()
        score = int(r["score"])
        for hhmm in _range_hhmm(r["start"], r["end"]):
            prefs[(day, hhmm)] = score
    return prefs

def load_data(data_dir):
    p = Path(data_dir)
    tasks  = [Task(r)  for r in _read(p / "tareas.csv")]
    events = [Event(r) for r in _read(p / "eventos.csv")]
    slots  = load_disponibilidad(p)
    prefs  = load_preferencias(p)
    return tasks, events, slots, prefs
