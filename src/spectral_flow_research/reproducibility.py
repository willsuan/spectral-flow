
import os, sys, json, platform, datetime, hashlib

def init_seed(seed: int = 1337):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    return np.random.default_rng(seed)

def write_manifest(run_dir: str, seed: int, extra: dict = None):
    info = {"python_version": sys.version, "platform": platform.platform(), "time_utc": datetime.datetime.utcnow().isoformat()+"Z"}
    try:
        import numpy, scipy, matplotlib
        info.update(numpy=numpy.__version__, scipy=getattr(scipy,"__version__","(none)"), matplotlib=matplotlib.__version__)
    except Exception: pass
    manifest = {"seed": seed, "environment": info}
    if extra: manifest.update(extra)
    with open(os.path.join(run_dir, "repro_manifest.json"), "w", encoding="utf-8") as f: json.dump(manifest, f, indent=2)
    return manifest

def stamp_from_manifest(manifest: dict, run_dir: str) -> str:
    h = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return f"{os.path.basename(run_dir)} · {h}"
