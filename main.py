from dataclasses import dataclass, asdict
import numpy as np
from sklearn.ensemble import RandomForestClassifier

@dataclass
class Trial:
    wrist_velocity: float
    path_deviation: float
    imu_jerk: float
    duration: float

TARGET = Trial(1.0, 0.05, 0.20, 1.0)

def vec(t):
    return np.array([t.wrist_velocity, t.path_deviation, t.imu_jerk, t.duration], float)

def make_training(seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    specs = {
        "too_slow": (0.75, 0.06, 0.22, 1.25),
        "on_target": (1.00, 0.05, 0.18, 1.00),
        "too_fast": (1.25, 0.08, 0.28, 0.80),
        "unstable": (1.00, 0.16, 0.50, 1.00),
    }
    for label, mu in specs.items():
        for _ in range(100):
            t = Trial(
                rng.normal(mu[0], 0.05),
                rng.normal(mu[1], 0.015),
                rng.normal(mu[2], 0.04),
                rng.normal(mu[3], 0.06),
            )
            X.append(vec(t)); y.append(label)
    return np.vstack(X), y

def deviations(t):
    return {
        "velocity_error_pct": 100*(t.wrist_velocity-TARGET.wrist_velocity)/TARGET.wrist_velocity,
        "path_error": t.path_deviation-TARGET.path_deviation,
        "smoothness_error": t.imu_jerk-TARGET.imu_jerk,
        "duration_error": t.duration-TARGET.duration,
    }

def language_feedback(label, d):
    parts = []
    v = d["velocity_error_pct"]
    if abs(v) <= 5:
        parts.append("Movement speed was close to the target.")
    elif v > 0:
        parts.append(f"Wrist speed was about {abs(v):.0f}% faster than target.")
    else:
        parts.append(f"Wrist speed was about {abs(v):.0f}% slower than target.")
    if d["path_error"] > 0.03:
        parts.append("The wrist path deviated more than desired.")
    if d["smoothness_error"] > 0.05:
        parts.append("The movement was less smooth than the target pattern.")
    action = {
        "too_fast": "Reduce speed slightly on the next trial while keeping the path stable.",
        "too_slow": "Increase speed slightly on the next trial without sacrificing smoothness.",
        "unstable": "Prioritize a smoother, more consistent trajectory on the next trial.",
        "on_target": "Maintain the current strategy on the next trial.",
    }[label]
    parts.append(action)
    return " ".join(parts)

def trial_score(d):
    return abs(d["velocity_error_pct"])/10 + abs(d["path_error"])*10 + abs(d["smoothness_error"])*10

X, y = make_training()
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X, y)

trials = [
    Trial(1.18, 0.10, 0.31, 0.84),
    Trial(1.08, 0.07, 0.24, 0.92),
    Trial(1.02, 0.05, 0.19, 0.99),
]

prev = None
for i, t in enumerate(trials, 1):
    probs = model.predict_proba(vec(t).reshape(1,-1))[0]
    idx = int(np.argmax(probs))
    label = model.classes_[idx]
    conf = float(probs[idx])
    d = deviations(t)
    print(f"\nTrial {i}")
    print("state:", label, "confidence:", round(conf, 3))
    print("deviation:", {k: round(v,3) for k,v in d.items()})
    print("feedback:", language_feedback(label, d))
    if prev is not None:
        print("next-trial improvement:", round(trial_score(prev)-trial_score(d), 3))
    prev = d
