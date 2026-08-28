# Recurring-event placebo-separation protocol

Status: preregistered before fresh evaluation on 2026-08-28.

## Change under test

For the recurring-event executable, episode estimators require the
episode-level 95% lower bound of the absolute effect to exceed the strongest
of twelve fixed displaced-schedule placebos. Residual and detrended estimators
replay their own identical-fold scorer under the same placebos and must beat
the strongest by the ordinary admission margin. The immutable history-only
primary, known-at filtering, fold construction, shrinkage, and publication
policy are unchanged.

The previously inspected standard corpus with seed
`5045431918660058516` is diagnostic only. It cannot graduate the change.

## Frozen validation

The following first corpus set diagnosed an estimator-family bypass and was
therefore spent during development. It is retained for audit but cannot
graduate the revised implementation:

| Corpus | Seed | Cases SHA-256 | Oracle SHA-256 |
| --- | ---: | --- | --- |
| standard-1 | 886503434522399952 | `957eba9b3f9291a6b721460c7198fc6e1d9953a10a61b4ed39f598f04c61287b` | `a64d221064c8ee701d3482a2929f7a230adf04ce26106df4e86b9517ed952f0a` |
| standard-2 | 7097755664058623658 | `dbfbdcab40f7fe7deadb8520d0d3943ad36f9c080304558a8c101df256f6c71a` | `6315a42028b3ef1f870823f46856c0957a46aa51d2883559f412d689821d53de` |
| standard-3 | 8950528693820448630 | `9dbbc6f795d42445dafe22ddaabca0e9fec6e8eaa602d1f0acbb3a04880a227e` | `75f568ac90bce0bb661c9d33c3c131424c8bcf5f66e3b3f4b95ca3945b9eddcf` |
| stress-1 | 1005055899785588237 | `1926293e9cbd9672d34807b6a2c0a70ba9936e3379ef46c299d17e571904131b` | `121a2ad434d23cac6d37b2bdae643486d8be8ba5d011d2c2d8fc227d0f03ca66` |
| stress-2 | 1660064188514182809 | `79dc85df07b917d265cd6036966cd060b2ab3dd1438eaea4e35c9ac0bc46e0a0` | `05614ff3f13164956f6094a07dd3e6ac77471808c73b1aa729033428c8c30ac4` |
| stress-3 | 8049010174415117375 | `c4b00266e1c26489a918fc32e547687201830f8fcb4521882308cc8156ded152` | `d0beb44959656d53c5fde5505834c0bc827837382ae60db607abc45594d09c1d` |

Standard corpora contain 20 cases per family; stress corpora contain eight
cases per stratum.

The revised estimator-specific gate at `968c1d3` is evaluated only on this
second, untouched graduation set, frozen before execution:

| Corpus | Seed | Cases SHA-256 | Oracle SHA-256 |
| --- | ---: | --- | --- |
| standard-1 | 4865898794418414059 | `cf05ce47ad0953367a32c03049a7682ffe072bc105ee0d2488ad916cd9dc4fb2` | `ed92d108a56185546182cb0022a4d4ea8d88d1deae188eb53b781221b3255ca5` |
| standard-2 | 3581236821206984004 | `cb924e1b21a947a040fc4d6438526b8d47493ebd5201c27fab737bc84d23c66f` | `9dae78ca7a2cb220e7c876b05bf79501c8d6b96a7ef0a2d732fb7266942e9a59` |
| standard-3 | 1982017067946730369 | `44ac3b9bbd49057f8286f0dd1af5daf4391656f1e23abd7575f7f79f836f8955` | `5fcc50a1d1cf8ba7f8daa80e20a92dc67e74596232dd551f3c7497c3c9272dc6` |
| stress-1 | 5575011721418376358 | `b5e74050b1473b4187e23360f131677dad0c5130e631bc08fd11c8f6c8824066` | `1f6e0d7f88f361d63385819f0dbdc646f50382acd92841467ccd9cc25e698b92` |
| stress-2 | 8027168944844480793 | `e96762c9ead57dcd8494cdd9863cb3f75bd84fc2cc12c1aaca1bece1638e1835` | `96efe4d08117ac9083e10bff03afe51a82ee8bbdff08c0ff20e721346b55f6c7` |
| stress-3 | 5510475529957827327 | `ae64f37b21be1e52fd7674fe3e35cd19ea72ce91070f47fc37bd4c390fbf7691` | `935497b9a2867f720a479e8537299c5db99b3172dbc10f184ad67d5534d947ca` |

The first stress execution exposed a pre-existing final-fit domain crash in
Croston-SBA. The three standard results above remain evaluations of the event
gate at clean commit `cab4e65`; the stress set is diagnostic and cannot
graduate the repaired engine. After the general final-domain fallback landed
at `818e2c9`, this replacement stress-only set was frozen before execution:

| Corpus | Seed | Cases SHA-256 | Oracle SHA-256 |
| --- | ---: | --- | --- |
| stress-1 | 4980103135766468476 | `36b53a7449423dc9643837bd858328cc297689be61ed532f7a1cee06363e5b9c` | `73804f338dbe02f3a08306fe350f1693aed9413101c8bc757f8c51611b6fb193` |
| stress-2 | 6120661279311270264 | `f14ebf9b6c9182b506a9baa8e3275b0f100e36bb9d1adfa28447839ff79562c0` | `fb2b2847a9e752714ed2dab873f6d0c681cb00085e66cdc71bb2249360663362` |
| stress-3 | 1480469107934715842 | `406bac3b378bbb8869562a2a1001d28d186fa334f103d53fbf4c357e20ad6049` | `ad3d12c313cd5c102e763a1f6296ca76657134806bcfd02cba3354f48d4f7ab2` |

Run the repository's unchanged ContextBench scorer and retain every case and
oracle file. The change graduates only if:

- all six runs complete with zero temporal leakage and publication parity;
- the pooled standard false-influence rate is below 1%;
- standard admission precision is at least 90% and recall at least 80%;
- both true standard influence families improve mean sMAPE;
- every stress run retains all required strata and default-policy immutability;
- pooled high-SNR stress recall is at least 80%; and
- no false numeric or structural assertion changes the immutable primary
  under the default policy.

Per-run confidence intervals and failures remain visible. No family, seed,
case identifier, oracle field, or benchmark label may enter runtime logic.
