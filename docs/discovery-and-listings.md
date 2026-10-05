# Discover Gnomon and prepare listings

Canonical description (GitHub About, PyPI and MCP metadata):

> Inspect time series, forecast with your models, compare forecasts, and review observed outcomes.

The package description is in `pyproject.toml`; runtime discovery uses the same
text. Run `python scripts/prepare_discovery.py` after version/description changes,
and `python scripts/prepare_discovery.py --check` before release. The script
prepares files only; it never publishes or edits a remote account.

## Prepared surfaces

- [GitHub About metadata](../integrations/mcp/github-about.json): description and search topics.
- [MCP Registry manifest](../server.json): pinned PyPI package, stdio transport and startup arguments.
- [Client recipes](../integrations/mcp/README.md): existing installation or isolated package launch.
- [llms.txt](../llms.txt): a small reading index for repository crawlers. It supplements normal docs;
  it does not promise that any host or search engine consumes it. There is no hosted website
  deployment in this change; publish the file at `/llms.txt` if a documentation site is added.

## Installed task discovery

```bash
gnomon capabilities --task forecast
```

Python: `session.capabilities(task="forecast")`. MCP:
`gnomon_capabilities` with `{"task":"forecast"}`. Ordinary capabilities includes
a compact `tasks` index with per-task configuration states. The detailed reply
names the tool, exact schema-discovery call, a template and its missing `requires`
fields, configured providers/routers where relevant, and setup requirements.
A complete synthetic example is labelled as a smoke test, not a user forecast.

Tasks: `inspect`, `describe`, `forecast`, `compare`, `route`, `review`, `recall`,
`submit_actuals`, `temporal`. `route` means advisory selection from a saved study;
configured adaptive routers are used under `forecast`.

Availability describes this session's configuration, not whether data or evidence
is sufficient. CLI capabilities runs configuration discovery without opening a
ledger; ledger tasks therefore report `open_configured_ledger` until used in an
opened session. Permissions and evaluation budgets remain unchanged. Discovery
itself executes no forecast or outcome write. Optional configured provider
initialization/model discovery at session startup retains its existing behavior.

## Publication prerequisites

The manifest follows the [official server.json format](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/generic-server-json.md).
The [PyPI verification rules](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/package-types.mdx)
require the released package README to contain the matching `mcp-name` marker;
this repository's README includes it. Registry publication also requires
namespace ownership authentication under the [official requirements](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md).

The registry launcher uses `uvx gnomon-forecast@VERSION mcp serve`. The package
exposes both `gnomon` and `gnomon-forecast` entry points to the same CLI because
its distribution and original executable names differ. Do not submit a manifest
for an older wheel that lacks the alias or ownership marker.

Before publishing:

1. Release the reviewed package through the [existing release process](ci-cd.md).
   Unpublished checkout changes are not present in an already-uploaded version.
2. Regenerate metadata for that exact version. Validate `server.json` against its
   pinned `$schema` and smoke-test the released package's actual launcher.
3. Verify the PyPI description marker and authenticate ownership of
   `io.github.TensorLink-AI/gnomon`. Follow the official publisher's current
   validation and publication flow. Local schema validity is not registry admission.
4. After publication, verify the returned registry record and exact package version.
   Other directories can reuse the same description and connection recipes; no
   additional directory membership is implied by these files.

A GitHub About update is independent of publishing package code. PyPI descriptions
come from released artifact metadata and cannot be updated by editing this checkout.
