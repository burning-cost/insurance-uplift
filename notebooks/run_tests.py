# Databricks notebook source
# MAGIC %pip install catboost polars pytest pytest-cov --quiet

# COMMAND ----------

import subprocess, sys, os, shutil, re

WORKSPACE_DIR = "/Workspace/Shared/insurance-uplift"
TMP_DIR = "/tmp/insurance-uplift"

if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)

shutil.copytree(
    WORKSPACE_DIR, TMP_DIR,
    ignore=shutil.ignore_patterns("*.pyc", "__pycache__", "run_tests*", ".git", "dist")
)

sys.path.insert(0, os.path.join(TMP_DIR, "src"))

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     os.path.join(TMP_DIR, "tests"),
     "-v", "--tb=short", "--no-header",
     "-p", "no:cacheprovider",
    ],
    capture_output=True,
    text=True,
    cwd=TMP_DIR,
    env={**os.environ, "PYTHONPATH": os.path.join(TMP_DIR, "src")},
)

output = result.stdout
if result.stderr:
    output += "\nSTDERR:\n" + result.stderr[:2000]

clean = re.sub(r'\x1b\[[0-9;]*m', '', output)
exit_marker = "RETURNCODE={}".format(result.returncode)
full_result = clean + "\n" + exit_marker

if len(full_result) > 900000:
    full_result = full_result[-900000:]

dbutils.notebook.exit(full_result)
