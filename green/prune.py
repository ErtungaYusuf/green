"""Pruning: what a node may forget, and when. Local policy, not consensus.

An archive node (the default) keeps every block and every blob, so others can
sync all of history from it. A pruned node (`--prune`) keeps:

  * every header, but block bodies only for the last `keep` blocks (at least the
    reorg window); it tells peers where its bodies start, so they ask others;
  * the ledger and its snapshots, so it still serves fast sync to new nodes;
  * every blob something still needs: the matrices of jobs that are not
    finished and of jobs used by recent blocks, the manifests and results of
    live batches and of delivered batches of unfinished jobs, the programs,
    inputs and outputs of tasks not yet settled, data named by mempool
    transactions, snapshot chunks, and this node's own unrevealed work.

Blobs are collected in two phases: a blob nothing references is first marked,
and deleted only BLOB_RETENTION_BLOCKS later if it is still unreferenced. The
delay gives buyers time to fetch results after their job finishes and covers
anything needed again after a reorg.
"""

from . import tx as txmod

PRUNE_KEEP_BLOCKS = 5760        # about a day of block bodies
BLOB_RETENTION_BLOCKS = 5760    # about a day between "unused" and "deleted"


def referenced_blobs(node, keep: int = PRUNE_KEEP_BLOCKS) -> set[str]:
    chain = node.chain
    refs: set[str] = set()
    with node.lock:
        state = chain.state
        batches = []
        for job_id, job in state.jobs.items():
            total = len(txmod.all_units(job["dims"]))
            finished = (job["closed"] or len(job["done"]) == total) and not job["batches"]
            if not finished:
                refs.add(job_id)
                batches += list(job["delivered"].values())
            batches += list(job["batches"].values())
        for t in state.tasks.values():
            if t["status"] not in ("done", "failed"):
                refs.update(x for x in (t["prog"], t["input"], t["output"]) if x)
        for t in node.mempool.values():
            kind = t["type"]
            if kind == "job_post":
                refs.add(t["job"])
            elif kind == "task_post":
                refs.update((t["prog"], t["input"]))
            elif kind == "job_reveal":
                refs.add(t["manifest"])
            elif kind == "job_respond":
                refs.add(t["result"])
        for bh in chain.main[-keep:]:
            work = chain.header(bh)["work"]
            if work["job"] is not None:
                refs.add(work["job"])
        for m in node.snapshots:
            refs.add(m["id"])
            refs.update(m["chunks"])
        for info in node.my_batches.values():
            batches.append({"manifest": info["manifest"], "count": info["count"], "root": info["root"]})
        for buf in node.batch_buf.values():
            refs.update(buf["results"])
    for b in batches:
        if b.get("manifest"):
            refs.add(b["manifest"])
            refs.update(node.leaves(b) or [])
    return refs


def collect(node, height: int, retention: int | None = None, keep: int = PRUNE_KEEP_BLOCKS) -> dict:
    """One collection round: mark newly unused blobs, delete those unused for `retention` blocks."""
    retention = BLOB_RETENTION_BLOCKS if retention is None else retention
    refs = referenced_blobs(node, keep)
    marks = node.store.blob_marks()
    present = set(node.blobs.keys())
    add, remove, deleted = {}, [], 0
    for key in present:
        if key in refs:
            if key in marks:
                remove.append(key)                    # in use again
        elif key not in marks:
            add[key] = height
        elif height - marks[key] >= retention:
            node.blobs.delete(key)
            remove.append(key)
            deleted += 1
    remove += [k for k in marks if k not in present]   # gone some other way
    node.store.set_blob_marks(add, remove)
    return {"marked": len(add), "deleted": deleted, "kept": len(present) - deleted}
