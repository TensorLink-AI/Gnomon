# Source-header probe 034

This is metadata preparation under the existing evaluation-development goal,
not a broader agent evaluation. Freeze the following archive identities before
download; reject mismatches rather than using a different source silently.

| Source | Immutable record | File | Publisher MD5 |
| --- | --- | --- | --- |
| Electricity hourly | [4656140, version 3](https://zenodo.org/records/4656140) | electricity_hourly_dataset.zip | 18096614662b02640d265ad2a6a416bd |
| Melbourne pedestrian counts | [4656626, version 3](https://zenodo.org/records/4656626) | pedestrian_counts_dataset.zip | 420043b57ed8577564d299742e8acf97 |

The sources represent client electricity measurements and city pedestrian
sensors. Metadata research recovered the pedestrian record and the newer
electricity record after the failed lookups retained in draft 033. Their
published descriptions establish potential task relevance, not ledger headroom.

Download to a new ignored local directory, retain exact URLs/statuses/checksums
and calculate SHA-256. Open the TSF member only through its `@data` marker:
save comments, attribute declarations, frequency and metadata, never output or
parse an observation row during this probe. No extraction to arbitrary paths,
candidate forecasting, statistical scoring or Engy calls are performed.

This does not claim that downloading bytes is the same as keeping an archive
physically inaccessible. The protection here is against inspecting, parsing or
scoring its observation values before the source-specific eligibility/split
protocol is frozen. No previously reserved M5 or Favorita targets are accessed.

Before using data, determine whether timestamps denote actual source instants,
period labels or inferred regular-grid positions; whether preprocessing has
changed missing values/DST observations; and whether reported units are native
measurements or transformed values. A usable file format alone is insufficient.
If these cannot be established, preserve that limitation and reject unsupported
claims about real-time availability. Both arms must use the same declared
synthetic availability convention if true publication/recording times are absent.

## Encoding recovery after acquisition

Both downloads matched the publisher checksums. The original UTF-8 header probe
failed on a Windows-1252 punctuation byte in the pedestrian comments. The
[publisher's TSF loader](https://github.com/rakshitha123/TSForecasting/blob/master/utils/data_loader.py)
explicitly uses `cp1252`. Use that encoding with strict decoding and re-read
only the headers from the unchanged local archives; retain the original failure
receipt. This is a format correction, not permission to inspect observation rows.
