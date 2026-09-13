# Original pedestrian timestamp source: metadata amendment 036

The TSF source failed panel 035's fixed coverage requirement. Preserve that
failure. Investigate the original publisher's timestamped archive to distinguish
source coverage from the implicit regular-grid representation. This amendment
does not reduce the 24-series requirement, change the source dates or select
series using forecast errors. No forecasting or Engy calls are authorized here.

The [City of Melbourne data portal](https://data.melbourne.vic.gov.au/explore/dataset/pedestrian-counting-system-monthly-counts-per-hour/information/)
publishes hourly sensor counts and warns that sensor-location changes matter.
It documents zero as a recorded count when no pedestrians pass, not permission
to replace absent records with zeros. Its current API metadata lists an archive
named `Pedestrian_Counting_System_Monthly_counts_per_hour_may_2009_to_14_dec_2022.csv.zip`.
The current API's live schema need not equal the archive schema.

Freeze the following attachment identity before acquisition:

- Catalog: `pedestrian-counting-system-monthly-counts-per-hour`.
- Attachment: `pedestrian_counting_system_monthly_counts_per_hour_may_2009_to_14_dec_2022_csv_zip`.
- [Exact attachment URL](https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/pedestrian-counting-system-monthly-counts-per-hour/attachments/pedestrian_counting_system_monthly_counts_per_hour_may_2009_to_14_dec_2022_csv_zip).
- Observed HEAD Content-Length: 42,654,028 bytes.
- Observed ETag: `"22cc821dafae42a5180a74080014e2ad-6"` (opaque HTTP validator,
  **not** asserted to be a file MD5).
- Observed Last-Modified: 2024-08-15 01:02:22 GMT.
- Catalog metadata SHA-256:
  `cc4ecb812834f9985444db370734035b8760952dcd6d83709bd70676c7821f00`.

Use a conditional GET with the frozen ETag, reject size/validator changes and
calculate SHA-256 before opening the ZIP member. Preserve all responses/failures.
Read only the CSV header at this stage; do not parse counts or print data rows.
The archive has no asserted measured historical recording vintages. Its 2024
modification time does not tell when each earlier count first became knowable.

If the schema permits a suitable preparation, define timestamp and sensor-ID
mapping prospectively, including missing/duplicate hours and DST ambiguity.
Do not claim that the original counts are identical to TSF or mix their values
in one series without verifying provenance. The previous first-history and
identity-selection conditions must remain explicit. Reserved later counts stay
unparsed until a separately frozen final evaluation is justified.

## Timestamp-only coverage stage

Acquisition passed the frozen HTTP validator and size checks. The downloaded
archive SHA-256 is
`5fa1d8fd8a50b0b2eededb85149a541336c2cfe1ab53706a0dbb1e81a526bc8a`.
Its CSV schema has Date_Time, redundant calendar fields, Sensor_ID, Sensor_Name
and Hourly_Counts. A timestamp-only format example was inspected; the count
field was not parsed. Inspect the same 4,954 nominal-hour span from protocol
035 using timestamp and sensor fields only. Record missing/duplicate labels,
sensor-name changes and coverage bitmaps. Do not read or summarize count values,
change dates, select a smaller favorable subwindow or infer DST fold semantics.

The first coverage attempt stopped before reading rows: the ZIP also contains
a macOS resource-fork entry ending in `.csv`. Pin the exact previously observed
publisher CSV member, excluding the resource fork. Preserve that failed attempt
in `timestamp-attempt-001.json`; this changes no timestamp or eligibility rule.
