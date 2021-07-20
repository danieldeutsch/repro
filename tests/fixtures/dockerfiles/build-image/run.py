import json
import os
import sys

sys.stdout.write("stdout contents")
sys.stderr.write("stderr contents")

results = {}

for dir_path in sys.argv[1:]:
    results[dir_path] = os.path.exists(dir_path)

for env in ["ENV1", "ENV2"]:
    results[env] = os.environ[env] if env in os.environ else None

if len(sys.argv) > 1:
    with open(f"{sys.argv[1]}/results.json", "w") as out:
        out.write(json.dumps(results, indent=2))
