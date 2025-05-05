from datetime import datetime
from math import ceil
from config_loader import CFG

DAYS = {"lunes","martes","miercoles","jueves","viernes","sabado","domingo"}
FMT = "%H:%M"
BLOCK_MIN = CFG["minutes_per_block"]

class Task:
    def __init__(self, d):
        self.name     = d["nombre"]
        self.duration = int(d["duracion"])
        self.deadline = d.get("deadline") or None
        self.deadline_dt = (
            datetime.strptime(self.deadline, "%Y-%m-%d %H:%M") if self.deadline else None
        )

class Event:
    def __init__(self, d):
        self.name  = d["nombre"]
        self.day   = d["dia"].lower()
        if self.day not in DAYS:
            raise ValueError(f"Dia invalido: {self.day}")
        self.start = d["hora"]
        # valida formato
        datetime.strptime(self.start, FMT)
        self.duration_min = int(d.get("duracion_min", BLOCK_MIN))
        self.blocks = ceil(self.duration_min / BLOCK_MIN)
