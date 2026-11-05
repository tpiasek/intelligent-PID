import csv
import itertools
import os

from optimize import evaluate, optimize_gains
from parameters import R_LIST, UMAX_LIST

TAU_LIST = [0, 2, 5, 10, 15, 20]
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "dataset_raw.csv")
COLUMNS = ["tau", "r", "umax", "Kp", "Ki", "Kd", "itae", "overshoot", "settling"]


def generate(taus=TAU_LIST, rs=R_LIST, umaxs=UMAX_LIST, path=OUT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cases = list(itertools.product(taus, rs, umaxs))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for i, (tau, r, umax) in enumerate(cases, start=1):
            Kp, Ki, Kd = optimize_gains(tau, r, umax)
            m = evaluate((Kp, Ki, Kd), tau, r, umax)
            writer.writerow([tau, r, umax, Kp, Ki, Kd, m["itae"], m["overshoot"], m["settling"]])
            f.flush()
            print(f"{i}/{len(cases)} tau={tau} r={r} umax={umax} Kp={Kp:.2f} Ki={Ki:.2f} Kd={Kd:.2f}")
    return path

if __name__ == "__main__":
    generate()