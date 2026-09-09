# MCP stdio evidence workflow

Run the installed standalone example:

```sh
python -m gnomon.examples.mcp_workflow
```

It starts the installed server with synthetic CSV data in a temporary directory.
The client uses only Python's standard library. Its JSON output includes the
complete sent/received transcript and the verified forecast length/hash.

The sequence is:

1. `initialize` with `protocolVersion: 2025-06-18`; verify the returned version is
   supported by the client before continuing.
2. `notifications/initialized` (a notification, without an ID).
3. `tools/list` to discover available schemas.
4. `gnomon_capabilities` to read provider, storage and protocol scope.
5. `gnomon_inspect` with the source path and explicit source timezone.
6. `gnomon_forecast` with the returned `data_ref`, provider and horizon.
7. `gnomon_read` repeatedly with the returned `result_ref` and `next_offset` until
   the latter is null. Concatenate exact text, check Unicode character count and
   SHA-256 of UTF-8 bytes, then parse the complete JSON.

The example requests 2,000 points and limits response size to force pagination.
It verifies every point and the timestamp count. Payload partiality is not
partial scoring; `partial_scope: response_payload` tells you to retrieve the
rest. Large evaluation responses separately expose `full_study`, which retrieves
complete saved fold evidence. `full_result` only reconstructs the payload that
was retained; that payload may be a compact study summary.

Gnomon advertises its implemented protocol version, the negotiated version after
initialization, and stdio transport. This example establishes this exchange only;
neither it nor capabilities claims compatibility with every MCP host. A client
must stop if the returned version is unsupported. Routing/ledger MCP tools are
available when operator configuration enables a ledger; temporal tools likewise
require `enable_temporal = true`. Inspect capabilities instead of assuming a tool
is absent from every deployment.

No credentials or remote providers are needed. The example terminates the server
by closing stdin and checks process exit status. See the installed source printed
by `python -c "import inspect,gnomon.examples.mcp_workflow as m; print(inspect.getsource(m))"`
for its runnable client implementation.
