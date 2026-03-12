# Agent Notes

## Manifest Generation: No Checkpointing

`generate_manifests.py` has no built-in checkpointing. If a job times out
mid-run, all progress on the current run is lost (completed runs are safe
since output is written per-run).

**Recovery strategy:**
- Check which `*_entities.parquet` files exist in `data/broker/manifests/`
- Re-run with `--runs` listing only the missing runs

**Improvement options:**
1. Add skip-if-exists logic (~3 lines in `main()`) so re-runs are idempotent
2. Use Ray parallelism (`--num-workers 0` for auto) instead of sequential
   (`--num-workers 1`) to finish faster on multi-core nodes
