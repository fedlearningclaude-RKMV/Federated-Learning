"""Modified Pedersen zero-knowledge proof for client authentication (pedersen2048 group).

Group. p is a 2048-bit prime and q a 256-bit prime with q | p - 1; G_q is the order-q subgroup
of Z_p*. Both are derived from the public seed SEED by hashing (FIPS 186-4 A.1.1.2 style:
the first prime q = H(seed|q|i) and then the first prime p = 1 mod 2q built from H(seed|p|j)),
so anyone can re-derive them (derive_group) and nothing is hidden in the parameters. The
lambda generators g_1 ... g_lambda of G_q are hashes of public labels raised to (p - 1) / q,
so nobody knows a discrete-log relation between them.

Registration. Client k draws a secret witness (x_1 ... x_lambda) in Z_q and registers the
Pedersen commitment Y_k = prod g_i^x_i mod p (lambda = 2 is the classic g^x h^r: secret key x,
blinding factor r).

Proof for one update (non-interactive via Fiat-Shamir):
    commit      k_i <- Z_q,  T = prod g_i^k_i
    challenge   c = H(group, lambda, Y_k, T, SHA-256(update), context) mod q
                context = schema id | round | client id
    respond     s_i = k_i + c * x_i mod q
The server accepts iff prod g_i^s_i = T * Y_k^c (mod p).

The modification of a plain Pedersen opening proof is the challenge: it binds the proof to the
exact update bytes and to the schema, round and client id, so one proof authenticates one
update. Changing a byte of the trees, replaying the update in another round or presenting it
as another client fails verification. The protocol is complete, special-sound and
honest-verifier zero-knowledge (an Okamoto-style proof of knowledge of the opening of Y_k).
"""

import hashlib
import secrets
from dataclasses import dataclass
from functools import lru_cache

GROUP = "pedersen2048"
SEED = "fedrf-diff/pedersen2048"
Q_COUNTER, P_COUNTER = 75, 138  # first counters that give primes; derive_group(SEED) reproduces them
Q = 0xEB64612784C39D165222B65FCC1D7B00878C6F06C9569CB4F714A3692EB5C41D
P = int(
    "F95C1FC805CA949B69A2706198BA7EE2D128DD2315A34AF86F2D2162122E5F63233758145C0CF270C7EDC4E335557E91"
    "E6FD23D3FBEC9A4575E24AF711E173ED2CB1825449FE1AC2009B1551704DA2ADEF27BEB0DA580E703182F76B89BB8186"
    "8B953825FA3C68E620947990A72A5EB1703458F75E7E704C179B66E2859798C181008F5B5D546B678AC05931B0C366C9"
    "E126C3E1DF9FA54CC2DCF17A4CF0A010C59F376EB082791E331AB6EED187860923641C5F0D8C27CA803FA711F2260A1D"
    "74FDC84FFFECBB1117B76C09994D4FF5079757C94270D532E627569728496B43CC3EE96D862E40AFAAA7DEF23154F0F6"
    "C8E2E0B0EE448FBAFA76D75CA7294425", 16)
ELEMENT_BYTES, SCALAR_BYTES = 256, 32
_SMALL_PRIMES = [n for n in range(3, 2000) if all(n % d for d in range(2, int(n ** 0.5) + 1))]


def _hash_int(label, bits):
    stream, counter = b"", 0
    while len(stream) * 8 < bits:
        stream += hashlib.sha256(f"{label}|{counter}".encode()).digest()
        counter += 1
    return int.from_bytes(stream, "big") >> (len(stream) * 8 - bits)


def is_probable_prime(n, label, rounds=64):
    """Miller-Rabin with hash-derived bases (error below 4^-rounds)."""
    if n < 2:
        return False
    for small in _SMALL_PRIMES:
        if n % small == 0:
            return n == small
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for i in range(rounds):
        x = pow(2 + _hash_int(f"{label}|mr|{i}", n.bit_length() + 64) % (n - 3), d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def derive_group(seed=SEED, p_bits=2048, q_bits=256, q_start=0, p_start=0):
    """Re-derive (p, q, q counter, p counter) from the public seed."""
    i = q_start
    while True:
        q = _hash_int(f"{seed}|q|{i}", q_bits) | (1 << (q_bits - 1)) | 1
        if is_probable_prime(q, seed):
            break
        i += 1
    j = p_start
    while True:
        x = _hash_int(f"{seed}|p|{j}", p_bits) | (1 << (p_bits - 1))
        p = x - x % (2 * q) + 1
        if p.bit_length() == p_bits and is_probable_prime(p, seed):
            return p, q, i, j
        j += 1


def _hash_to_group(label):
    h = pow(_hash_int(f"{GROUP}|{label}", ELEMENT_BYTES * 8 + 256) % P, (P - 1) // Q, P)
    if h == 1:
        raise ValueError(f"degenerate generator for {label!r}")
    return h


@lru_cache(maxsize=None)
def generators(lam):
    return tuple(_hash_to_group(f"g{i}") for i in range(1, lam + 1))


def _multi_exp(exponents, lam):
    out = 1
    for g, e in zip(generators(lam), exponents):
        out = out * pow(g, e, P) % P
    return out


@dataclass(frozen=True)
class KeyPair:
    witness: tuple  # secret (x_1 ... x_lambda)
    commitment: int  # public Y = prod g_i^x_i


@dataclass(frozen=True)
class Proof:
    announcement: int  # T
    responses: tuple  # (s_1 ... s_lambda)

    @property
    def size_bytes(self):
        return ELEMENT_BYTES + SCALAR_BYTES * len(self.responses)


def keygen(lam):
    witness = tuple(secrets.randbelow(Q - 1) + 1 for _ in range(lam))
    return KeyPair(witness, _multi_exp(witness, lam))


def in_group(element):
    """Membership in G_q, checked when a client registers its commitment."""
    return 1 < element < P and pow(element, Q, P) == 1


def update_context(schema_id, rnd, client_id):
    return f"schema={schema_id}|round={rnd}|client={client_id}"


def _challenge(lam, commitment, announcement, digest, context):
    data = b"|".join([GROUP.encode(), str(lam).encode(), commitment.to_bytes(ELEMENT_BYTES, "big"),
                      announcement.to_bytes(ELEMENT_BYTES, "big"), digest, context.encode()])
    return int.from_bytes(hashlib.sha256(data).digest(), "big") % Q


def prove(keypair, digest, context):
    lam = len(keypair.witness)
    nonces = [secrets.randbelow(Q) for _ in range(lam)]
    announcement = _multi_exp(nonces, lam)
    c = _challenge(lam, keypair.commitment, announcement, digest, context)
    return Proof(announcement, tuple((k + c * x) % Q for k, x in zip(nonces, keypair.witness)))


def verify(commitment, digest, context, proof, lam):
    if len(proof.responses) != lam or not 1 < proof.announcement < P:
        return False
    if not all(0 <= s < Q for s in proof.responses):
        return False
    c = _challenge(lam, commitment, proof.announcement, digest, context)
    return _multi_exp(proof.responses, lam) == proof.announcement * pow(commitment, c, P) % P
