"""Network-wide constants for the Green network.

Every node on the same network must use identical values here; changing any of
them is a hard fork. The reasoning behind each number is in SECURITY.md.
"""

COIN_NAME = "Green"
TICKER = "GRN"

# 1 GRN = 100_000_000 yaprak (the smallest unit, like satoshi in Bitcoin).
YAPRAK_PER_COIN = 100_000_000

# --- Proof of Useful Work ---------------------------------------------------
# A "work unit" is one UNIT x UNIT x UNIT integer matrix multiplication.
# Every work unit is exactly one lottery ticket, no matter whose matrices it
# multiplies, so mining power == matrix-multiplication throughput.
UNIT = 256
# The unit is multiplied block by block (BLOCK x BLOCK tiles). Every tile
# product goes into the transcript that the lottery ticket commits to.
BLOCK = 32
# Rank of the seed-derived noise added to both inputs. Must be >= BLOCK so that
# knowing A*B in advance gives no shortcut for producing the transcript.
NOISE_RANK = 32
# Freivalds checks work modulo this prime with random vectors over all of Z_p,
# so a wrong result survives one round with probability <= 1/p ~ 2^-31.
FREIVALDS_PRIME = 2**31 - 1
FREIVALDS_ROUNDS = 4            # error <= 2^-124

# --- Consensus ----------------------------------------------------------------
TARGET_BLOCK_SECONDS = 15
# Difficulty is set for every block by ASERT (aserti3-2d): each half-life the
# chain falls behind its ideal schedule halves the difficulty, each half-life
# ahead doubles it. Simulated against the windowed retarget of v0.5
# (bench/difficulty_sim.py): with miners that join only while mining is cheap,
# the windowed rule averaged 20.6 s blocks and fell 9.3 hours behind schedule
# in 6,000 blocks; ASERT kept 15.0 s and 0 drift. After a 100x hashrate drop it
# recovers in 4.0 h instead of 8.8 h. The cost: after a sudden 10x surge it
# takes ~1 h instead of minutes to reach 15 s again (and then keeps issuance on
# schedule, where the windowed rule drifts for good).
ASERT_HALFLIFE = 1800           # seconds (120 blocks)
MAX_TARGET = 2**256 - 1
# Start easy; the retarget algorithm finds the real difficulty within a few
# intervals.
INITIAL_TARGET = MAX_TARGET // 200
# A block's time must be above the median of the previous MTP_WINDOW blocks and
# at most MAX_FUTURE_SECONDS ahead of the validating node's clock.
MTP_WINDOW = 11
MAX_FUTURE_SECONDS = 60
# Nodes refuse to switch to a fork that branches off deeper than this. Stops
# cheap long-range forks built at the easy early difficulty.
MAX_REORG_DEPTH = 100
# Rule changes (green/versionbits.py): deployments are judged per period of
# SIGNAL_PERIOD blocks (~3.5 days); one locks in when SIGNAL_THRESHOLD blocks of
# a period (90%) signal it. Each deployment: {"bit": 0..28, "start": height,
# "timeout": height, "min_activation": height (optional), "lock_in_on_timeout":
# bool (optional)}. Heights should be multiples of SIGNAL_PERIOD. No deployment
# is scheduled yet.
SIGNAL_PERIOD = 20160
SIGNAL_THRESHOLD = 18144
DEPLOYMENTS: dict[str, dict] = {}

# Nodes write a snapshot of the ledger every this many blocks (about 12 hours),
# which new nodes can start from instead of replaying history (green/snapshot.py).
SNAPSHOT_INTERVAL = 2880

# Money (SPEC.md §3.5). The total supply is fixed: all of it sits in the protocol's
# reserve at genesis, and coins only ever move between the reserve and accounts, so
# the supply never grows (no inflation). Each block pays out reserve / EMISSION_DIVISOR:
# a smooth curve with no halvings, ~1.25 GRN at first. Part of every fee flows back
# into the reserve, so block rewards never dry up: once the reserve has been handed
# out, miners are paid from what users pay, smoothed over years.
TOTAL_SUPPLY = 21_000_000 * 100_000_000
EMISSION_DIVISOR = 2 ** 24        # a block pays out 1/16,777,216 of the reserve
FEE_RESERVE_PERCENT = 30          # of every block's fees, into the reserve (paid out again as rewards)
FEE_BURN_PERCENT = 20             # of every block's fees, destroyed: the supply shrinks a little with use
                                  # (the rest, 50%, to the block's miner)

# Shares: tickets that meet SHARE_FACTOR times the block target (on headers mined on
# one of the last SHARE_WINDOW blocks) are included in blocks and paid from the
# reserve, so every miner is paid in proportion to its work, not all or nothing.
# Nearly all of the emission goes to shares, so a miner earns in proportion to the work
# it does, a few times a minute, with no pool: finding the block itself is worth about
# one share (plus the inclusion cuts and the fees), not a jackpot.
SHARE_FACTOR = 128                # shares are 128 times easier than blocks: ~128 per block, one
                                  # every ~0.12 s; each costs every node a work unit to check
                                  # (~2 ms: ~2% of a core at 15 s blocks) and ~640 bytes
SHARES_PERCENT = 99               # of each block's emission, paid to shares (SHARE_FACTOR shares' worth)
SHARE_INCLUSION_PERCENT = 5       # of each share's reward, to the block that includes it
MAX_BLOCK_SHARES = 256
SHARE_WINDOW = 32                 # a share is mined on one of the last 32 blocks

# Transactions per block: 3000 every 15 s is 200 transfers a second. A node checks a
# full block of transfers in ~0.15 s when it saw them in its mempool (signatures cached)
# and ~0.4 s cold, on one core (bench/tps_bench.py); MAX_BLOCK_BYTES still bounds
# what nodes download and store.
MAX_BLOCK_TXS = 3000
# Serialized size of a block's transactions. Bounds what every node has to
# download, store and relay per block, like Bitcoin's block size limit.
MAX_BLOCK_BYTES = 1_000_000
# Upper bound on job size (in units per dimension) so a single job post cannot
# lock up validators.
MAX_JOB_UNITS_PER_DIM = 16
MAX_JOB_UNITS = MAX_JOB_UNITS_PER_DIM ** 3

# --- Fees and the compute market -----------------------------------------------
# The base fee is per signature a validator checks, as on Solana (5,000 lamports =
# 0.000005 SOL): 500 yaprak = 0.000005 GRN. Kept low on purpose; when blocks fill,
# transactions compete by fee per byte (the priority fee). Of every fee 30% goes back
# to the reserve and 20% is burned (Solana burns half of its base fees).
MIN_TX_FEE = 500               # per signature: 1 for a plain transaction, k for k-of-n
# With fees this low, opening ledger entries must not be: a transfer to an address that
# holds nothing must send at least this much (Solana's rent-exempt minimum plays the
# same part). Not a fee: the coins are the recipient's.
NEW_ACCOUNT_MIN = 10_000       # 0.0001 GRN
DATA_FEE_PER_BYTE = 1           # job data is stored by every node, so it pays per byte
# Results are sold in batches with commit-reveal. A miner commits to
# hash(address, units, root): the units (a bitmap over the job, so a batch can
# be the whole job) and the Merkle root of their result hashes. The commit
# reserves the units for RESERVE_BLOCKS; only then is the root revealed. Nobody
# can copy results they have not seen, and a revealed batch only pays its
# committer.
MAX_BATCH_UNITS = MAX_JOB_UNITS
RESERVE_BLOCKS = 20
# Delivery is optimistic: validators see neither the 256 KB results nor the
# list of their hashes, only the 32-byte root. The buyer fetches the results
# from the miner and checks every one; during the challenge window it (or
# anyone) can dispute any number of units at once. The miner must answer each
# disputed unit with its result, a Merkle proof and the data (every validator
# runs a Freivalds check) before the unit's deadline, or the whole batch is
# forfeit. Undisputed batches are paid when the window closes.
JOB_CHALLENGE_BLOCKS = 40
JOB_RESPONSE_BLOCKS = 20
# Each response makes every validator fetch one result (256 KB) and check it,
# so responses per block are capped, like the WASM adjudication budget. A new
# dispute's deadline is pushed back by the backlog of open disputes divided by
# this cap, so a flood of disputes can delay honest miners but never leave them
# without room to answer.
MAX_BLOCK_JOB_RESPONSES = 32
# The miner locks price / BOND_DIVISOR per unit as a bond with the commit; it
# comes back with the payment, goes to the buyer if the batch is never
# revealed, or to the challenger whose dispute goes unanswered.
BOND_DIVISOR = 10
# A cancel only takes effect CANCEL_DELAY blocks later; batches committed before
# it keep their share of the escrow and are settled as usual.
CANCEL_DELAY = 40

# --- General compute market (arbitrary WebAssembly) ---------------------------
# A task is a deterministic WASM program + input run under a fuel budget. The
# result is secured optimistically: a worker reveals it, and during the
# challenge window anyone can force on-chain re-execution to prove it wrong and
# take the worker's bond. Every validator re-runs a challenged task, so that
# work has to be bounded per task AND per block (like Ethereum's gas limit).
#
# Measured on wasmtime 49 (see SECURITY.md): worst case ~0.7 ns per fuel unit
# (cache-missing loads), compilation ~110 ns per program byte, parsing ~1 ns
# per byte. So a task's adjudication cost is fuel + program_bytes * 200.
WASMTIME_VERSION = "49.0.0"       # consensus: fuel accounting decides "did it finish?"
MAX_TASK_FUEL = 500_000_000       # ~0.35 s worst case
MAX_TASK_PROGRAM_BYTES = 1 << 20  # ~0.11 s to compile
MAX_TASK_INPUT_BYTES = 4 << 20
COMPILE_COST_PER_BYTE = 200       # adjudication cost units per program byte
MAX_BLOCK_ADJUDICATION = 1_500_000_000   # cost units per block, ~1 s worst case
# Data a block's posts make every validator fetch and check: a job's matrices (up to
# 33.5 MB) and a task's program and input. Without a cap, a block of 3,000 posts named
# gigabytes. A program counts PROGRAM_DATA_WEIGHT times its size: validating and
# instrumenting it costs ~0.4 s per MiB (green/wasm_stack.py), so a block holds at most
# 2 MiB of new programs.
MAX_BLOCK_DATA = 64 << 20
PROGRAM_DATA_WEIGHT = 32
ADJUDICATION_FEE_PER_MUNIT = 1_000       # yaprak per million cost units, paid by the challenger
TASK_RESERVE_BLOCKS = 20          # a commit reserves the task for this long
TASK_CHALLENGE_BLOCKS = 40        # window to dispute a revealed result
# Worker's bond, and the challenger's bond, as a fraction of the task price.
TASK_BOND_DIVISOR = 2

# --- Guarded accounts (accountability for autonomous agents) ------------------
# An account can name a guardian who sets a spending limit per period and can
# freeze it. The account itself can only tighten its guard; only the guardian
# can loosen, hand over or remove it. Counted as spending: transfers, job and
# task escrow, and fees (bonds are not: they come back unless fraud is proven).
MAX_GUARD_PERIOD = 10**8          # blocks

# Multisignature accounts: k of n keys (n <= MAX_MULTISIG_KEYS) must sign. A block
# may carry at most MAX_BLOCK_SIGS signatures in total, which bounds the time a
# validator spends verifying them (~0.45 s cold, 75 us each).
MAX_MULTISIG_KEYS = 16
MAX_BLOCK_SIGS = 6000

# A block's reward can be split among up to MAX_PAYEES addresses by weight, so a
# pool pays its miners in the block itself and never holds their coins.
MAX_PAYEES = 32
MAX_PAYEE_WEIGHT = 2**32

# Compute channels (green/compute.py): a buyer locks coins for one provider and
# pays it off chain, result by result, with vouchers the provider claims on
# chain; what is left returns to the buyer when the channel expires.
MIN_CHANNEL_BLOCKS = 240          # about an hour
MAX_CHANNEL_BLOCKS = 4 * 40_320   # about four weeks

# Wallets derive keys from a recovery phrase along m/44'/HD_COIN_TYPE'/account'
# (green/hd.py). Provisional: a SLIP-0044 number is registered before a public
# launch. Wallet files record their path, so changing it never moves a key.
HD_COIN_TYPE = 7777

DEFAULT_PORT = 8545               # HTTP API for wallets and tools
DEFAULT_P2P_PORT = 8546           # encrypted node-to-node protocol (green/p2p.py)
# Checkpoints (green/chain.py): the founder's node signs the block CHECKPOINT_DEPTH below
# its tip every CHECKPOINT_EVERY blocks, so a majority attacker can only rewrite the last
# minute or two; by default the period lasts a year of blocks unless the founder ends it.
CHECKPOINT_EVERY = 4
CHECKPOINT_DEPTH = 3
CHECKPOINT_DEFAULT_BLOCKS = 365 * 86400 // TARGET_BLOCK_SECONDS

# Nodes to ask for addresses when the address book is empty. Filled in when a
# public network launches; "host:port" or "<node id>@host:port". A host that is a
# DNS name is resolved and each address it answers with is tried (DNS seeds).
SEED_NODES: list[str] = []
# Maintainer keys (Ed25519 public keys, hex) that sign releases, and how many
# different ones a release needs (green/release.py). Filled in at launch.
RELEASE_KEYS: list[str] = []
RELEASE_THRESHOLD = 2

assert CANCEL_DELAY > RESERVE_BLOCKS


def task_bond(price: int) -> int:
    return max(1, -(-price // TASK_BOND_DIVISOR))


def adjudication_cost(fuel: int, program_bytes: int) -> int:
    """What re-running a task costs every validator, in cost units."""
    return fuel + program_bytes * COMPILE_COST_PER_BYTE


def challenge_fee(fuel: int, program_bytes: int) -> int:
    return MIN_TX_FEE + -(-adjudication_cost(fuel, program_bytes) * ADJUDICATION_FEE_PER_MUNIT // 1_000_000)


assert adjudication_cost(MAX_TASK_FUEL, MAX_TASK_PROGRAM_BYTES) <= MAX_BLOCK_ADJUDICATION


def emission(reserve: int) -> tuple[int, int, int]:
    """What a block whose parent holds `reserve` pays out: (the finder's part, the
    reward per included share, the finder's cut of each share it includes)."""
    e = reserve // EMISSION_DIVISOR
    share = e * SHARES_PERCENT // 100 // SHARE_FACTOR
    return e - SHARE_FACTOR * share, share, share * SHARE_INCLUSION_PERCENT // 100


def bond_for(price: int) -> int:
    return -(-price // BOND_DIVISOR)


def fmt_amount(yaprak: int) -> str:
    sign = "-" if yaprak < 0 else ""
    whole, frac = divmod(abs(yaprak), YAPRAK_PER_COIN)
    return f"{sign}{whole}.{frac:08d} {TICKER}"


def parse_amount(text: str) -> int:
    """Parse a decimal GRN amount like '1.5' into yaprak."""
    text = text.strip()
    if not text or text.startswith("-") or text.count(".") > 1:
        raise ValueError(f"invalid amount: {text!r}")
    whole, _, frac = text.partition(".")
    if not (whole or frac) or not (whole + frac).isdigit():
        raise ValueError(f"invalid amount: {text!r}")
    if len(frac) > 8:
        raise ValueError("at most 8 decimal places")
    return int(whole or "0") * YAPRAK_PER_COIN + int(frac.ljust(8, "0") or "0")
