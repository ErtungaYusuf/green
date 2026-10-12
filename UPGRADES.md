# How Green changes its rules

Bitcoin's lesson is that the hard part of upgrading a cryptocurrency is not the code, it is changing the rules without splitting the network or letting a few people decide for everyone. This is the process Green follows once a public network exists. Before that, every release may still start a new network.

## Kinds of change

| Kind | What changes | How it ships |
|---|---|---|
| Policy | What a node relays or mines (fee rules, limits in SPEC.md §8, P2P behaviour) | Any release; nodes may differ without forking |
| Soft fork | A rule that only makes fewer blocks valid | A **deployment** activated by miner signalling (below) |
| Hard fork | A rule that makes previously invalid blocks valid | Only by a flag day announced long in advance, or as an emergency fix (below) |

New features are designed as soft forks whenever possible: nodes that do not upgrade keep following the chain, so nobody is forced off it.

## Soft forks: version bits

Mechanism (green/versionbits.py, SPEC.md §5.5): a deployment has a version bit, a start height, a timeout height, optionally a minimum activation height and whether it locks in at the timeout regardless of signalling. Periods are SIGNAL_PERIOD = 20,160 blocks (about 3.5 days); a deployment locks in when 90% of a period's blocks signal it and becomes active one period later.

Process:
1. **Proposal**: a written description of the rule, why it is needed, and why it is a soft fork; public discussion.
2. **Specification**: the rule is added to SPEC.md, with the deployment's parameters.
3. **Implementation and tests**: the rule is enforced only when `chain.deployments.active(name, parent)`; tests show it activates, fails and survives reorgs; the example deployment `testdummy` and `tests/test_versionbits.py` show the pattern.
4. **Release**: a signed release (below) with the deployment in `params.DEPLOYMENTS`, the start at least one month after the release.
5. **Signalling**: miners running the release signal automatically (`--no-signal <name>` opts out). `grn deployments` shows progress.
6. **Activation**: at 90% the deployment locks in, then activates. Nodes that never upgraded keep following the chain; nodes that see an unknown bit signalled by most blocks warn their operator (`grn status`).

A deployment scheduled with `lock_in_on_timeout` activates at its timeout even without miner signalling. It exists for changes that users have clearly decided on and miners refuse to signal; it is a last resort, because nodes that disagree would follow a different chain.

## Emergency fixes

A bug that lets someone create coins, steal funds, or crash nodes is fixed as follows:
1. Reported privately (SECURITY.md describes how once a public network exists).
2. A fix that only tightens the rules (a soft fork without signalling, enforced immediately by upgraded miners) is preferred, because it does not split nodes that have not upgraded yet.
3. The fix is released and signed without public description; the description follows once most of the network has upgraded.
4. If a bad block made it into the chain, the fix may include a rule that refuses it; this is the only kind of change that may happen without a signalling period, and the reasons are published afterwards.

## Releases

- Every release ships `SHA256SUMS` and detached signatures of it by maintainers (green/release.py). It is genuine only if at least `RELEASE_THRESHOLD` (2) different keys from `RELEASE_KEYS` signed it and every file matches its digest:
  ```bash
  python -m green release verify SHA256SUMS maintainer1.sig maintainer2.sig
  ```
- Maintainers sign with `python -m green release sign SHA256SUMS`; signatures are domain-separated, so a release signature can never be replayed as a transaction.
- Future work: reproducible builds, so anyone can rebuild a release from the source and get the same digests.

## Who decides

Nobody can force a rule on the network: a soft fork only takes effect when miners signal it and users run software that enforces it, and users can always run software without it. The maintainers' power is limited to what they release and sign; signatures by several independent keys make sure no single person can ship a release.
