# Migration Guide: v2.4.x → v2.5.0

## Overview

v2.5.0 advances the pinned `github-copilot-sdk` from `1.0.2` to `1.0.6` (stable)
and raises the import-time support floor from the pre-GA beta `1.0.0b10` to GA
`1.0.0`. The provider's public surface (`mount`, `GitHubCopilotProvider`, entry
point, documented configuration env vars) is unchanged — no config key, env var,
public API symbol, or CLI flag is removed or renamed. The action on upgrade is
ensuring SDK `1.0.6` is installed. Pre-GA beta SDKs (`1.0.0bN`, `1.0.0rcN`) are
no longer supported and are rejected at import time.

---

## What Changed: SDK requirement is now `github-copilot-sdk==1.0.6`

- **What:** The provider now pins `github-copilot-sdk==1.0.6` (was `==1.0.2`) and
  enforces a GA support floor (`>=1.0.0`; was the pre-GA `>=1.0.0b10`). On import
  it checks the installed SDK version and raises a clear `ImportError` if it is
  below `1.0.0`. The `1.0.2 → 1.0.6` move is **non-breaking** — no SDK symbol the
  provider imports was removed or renamed, protocol version is unchanged, and the
  imported shapes (reasoning-effort literal, context-tier literal, capability
  overrides) are identical.
- **New GPT-5.6 models:** SDK `1.0.6` bundles Copilot CLI `1.0.69`, which surfaces
  the server-driven GPT-5.6 model family (`gpt-5.6-luna`, `gpt-5.6-sol`,
  `gpt-5.6-terra`) via `list_models`. The provider has no model allowlist, so
  these appear automatically once the backend advertises them — no provider code
  change enables them.
- **New runtime dependency:** SDK `1.0.6` adds `httpx>=0.24.0` (with its transitive
  deps: `httpcore`, `h11`, `anyio`, `certifi`, `idna`). A clean install
  pulls these via the regenerated `uv.lock`.
- **Get the new SDK:** Reinstall through Amplifier
  (`amplifier provider install --force github-copilot`), or pin manually:
  `pip install 'github-copilot-sdk==1.0.6'`. A clean `pip install` of `2.5.0`
  pulls `1.0.6` automatically via the `pyproject.toml` pin.
- **Dropped support:** Pre-GA beta SDKs (`1.0.0b10` and earlier betas, `rc`) that
  previously cleared the `1.0.0b10` floor now raise `ImportError` on import.
- **Rollback:** Pin provider `==2.4.0` with `github-copilot-sdk==1.0.2`.

---

## When

Provider version `2.5.0`.

---

## Rollback

If the new SDK breaks your workflow, pin provider `==2.4.0` with
`github-copilot-sdk==1.0.2`.

---

# Migration Guide: v2.3.x → v2.4.0

## Overview

v2.4.0 advances the pinned `github-copilot-sdk` from `1.0.0` to `1.0.2`. The
provider's public surface (`mount`, `GitHubCopilotProvider`, entry point,
documented configuration env vars) is unchanged — no config key, env var,
public API symbol, or CLI flag is removed or renamed. The only action on
upgrade is ensuring SDK `1.0.2` is installed.

---

## What Changed: SDK requirement is now `github-copilot-sdk==1.0.2`

- **What:** The provider now requires `github-copilot-sdk==1.0.2` (was `==1.0.0`).
  On import it checks the installed SDK version and raises a clear `ImportError`
  if it is below the supported floor (`>=1.0.0b10`). The `1.0.0 → 1.0.2` move is
  **non-breaking** — no SDK symbol the provider imports was removed or renamed.
- **Get the new SDK:** Reinstall through Amplifier
  (`amplifier provider install --force github-copilot`), or pin manually:
  `pip install 'github-copilot-sdk==1.0.2'`. A clean `pip install` of `2.4.0`
  pulls `1.0.2` automatically via the `pyproject.toml` pin.
- **Upgrading source in place?** If you update the provider from a 2.3.x checkout
  without reinstalling dependencies, the existing `1.0.0` is **not** replaced. It
  still imports (it clears the floor), but the test/version gate pins `1.0.2`, so
  reinstall the SDK to go green.
- **Rollback:** Pin provider `==2.3.0` with `github-copilot-sdk==1.0.0`.

---

## When

Provider version `2.4.0`.

---

## Rollback

If the new SDK breaks your workflow, pin provider `==2.3.0` with
`github-copilot-sdk==1.0.0`.

---

# Migration Guide: v2.2.x → v2.3.0

## Overview

v2.3.0 raises the `github-copilot-sdk` requirement and tightens the
environment variables the provider forwards to the SDK runtime. The
provider's public surface (`mount`, `GitHubCopilotProvider`, entry point,
version, documented configuration env vars
`AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME` / `GITHUB_TOKEN`) is unchanged.

User-visible changes:

1. SDK requirement is now `github-copilot-sdk==1.0.0b10`.
2. `COPILOT_CLI_PATH` is no longer honored.
3. `COPILOT_HOME` is no longer honored.
4. Ambient `COPILOT_SDK_AUTH_TOKEN` is no longer forwarded to the SDK runtime.

---

## What Changed

### 1. SDK requirement: `github-copilot-sdk==1.0.0b10`

- **What:** The provider now requires `github-copilot-sdk==1.0.0b10`. On
  import, the provider checks the installed SDK version and raises a
  clear `ImportError` if the installed version is older. The bump from
  b9 to b10 adds 9 new `MinimalMode` session-config pins (MUST:7-15:
  `enable_session_store`, `enable_skills`, `enable_file_hooks`,
  `enable_host_git_operations`, `enable_on_demand_instruction_discovery`,
  `skip_embedding_retrieval`, `embedding_cache_storage`,
  `enable_session_telemetry`, `mcp_oauth_token_storage`) so the SDK's
  defense-in-depth defaults are pinned explicitly. Wire-shape change only — no
  provider API change.
- **Behavior on upgrade:** Older SDK installs (`1.0.0b4`–`b9`) raise an
  actionable error at provider import time that names the required
  version and the install command.
- **Replacement:** Reinstall through Amplifier
  (`amplifier provider install --force github-copilot`), or pin manually:
  `pip install 'github-copilot-sdk==1.0.0b10'`.
- **Rollback:** Pin provider `==2.2.0` with `github-copilot-sdk==1.0.0b4`.
  The b10 pin and b4 pin cannot be mixed. (Provider `2.3.0` requires
  `==1.0.0b10`; pinning `2.3.x` against `b4` would fail at provider
  import time.)

### 2. `COPILOT_CLI_PATH` is no longer honored

- **What:** The provider no longer forwards `COPILOT_CLI_PATH` to the SDK
  runtime. The SDK uses its managed Copilot CLI binary for deterministic
  behavior.
- **Behavior on upgrade:** Any ambient `COPILOT_CLI_PATH` set by the user
  or shell is ignored when the provider spawns the SDK runtime.
- **Replacement:** None. If you depended on a custom CLI binary, pin
  provider `<=2.2.x` and open an issue describing the use case.
- **Rollback:** Pin provider `==2.2.0` with `github-copilot-sdk==1.0.0b4`.

### 3. `COPILOT_HOME` is no longer honored

- **What:** The provider no longer forwards `COPILOT_HOME` to the SDK
  runtime. Session state location is controlled by the provider, not by
  shell env.
- **Behavior on upgrade:** Any ambient `COPILOT_HOME` set by the user or
  shell is ignored when the provider spawns the SDK runtime.
- **Replacement:** Use `AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME` to control the
  provider's state root. It is resolved at provider-init time and applied
  through the provider's path configuration.
- **Rollback:** Pin provider `==2.2.0` with `github-copilot-sdk==1.0.0b4`.

### 4. `copilot.SubprocessConfig` is no longer imported from the SDK

- **What:** The provider no longer references the
  `copilot.SubprocessConfig` symbol. Process options that used to be passed
  via that wrapper (`copilot_home`, `cli_path`, `env`, `log_level`) are now
  passed directly as keyword arguments on the `CopilotClient(...)`
  constructor (`base_directory`, `env`, `log_level`, `github_token`). The
  SDK removed `SubprocessConfig` from its public API at b7 and it has
  remained absent through b10 — keeping it in the import surface would
  fail at import time on any supported SDK version.
- **Behavior on upgrade:** None for end users — `SubprocessConfig` was
  never part of the provider's public API. Fork maintainers who
  re-exported it from provider internals (or patched it in tests) need to
  switch to patching `CopilotClient` directly (e.g.,
  `monkeypatch.setattr(client_mod, "CopilotClient", FakeCopilotClient)`).
- **Replacement:** Construct `CopilotClient(base_directory=..., env=...,
  log_level=..., mode="copilot-cli", github_token=...)` directly. The
  `mode="copilot-cli"` argument is required — leaving it unset falls
  through to the SDK's default mode, which does not match the provider's
  wiring invariants. See
  `sdk_adapter/client.py::_ensure_client_initialized` for the canonical
  pattern.
- **Rollback:** Not applicable — the symbol was removed upstream.

### 5. Ambient `COPILOT_SDK_AUTH_TOKEN` is no longer forwarded to the SDK

- **What:** The provider scrubs `COPILOT_SDK_AUTH_TOKEN` from the env
  handed to the spawned SDK subprocess. The SDK uses this variable as the
  transport for the GitHub token it injects from the `github_token`
  constructor argument (via `--auth-token-env`); the SDK only writes it
  into the subprocess env when `github_token` is truthy. Without scrubbing,
  an ambient parent-shell value would survive into the spawned process on
  the no-token branch and authenticate against a credential the provider
  never resolved.
- **Behavior on upgrade:** When the provider runs without a resolved
  token (`GITHUB_TOKEN`, `COPILOT_AGENT_TOKEN`, `COPILOT_GITHUB_TOKEN`,
  and `GH_TOKEN` all unset), the SDK subprocess no longer inherits an
  ambient `COPILOT_SDK_AUTH_TOKEN`. The provider still fails closed on
  the no-token path with the usual `ProviderUnavailableError`. When the
  provider resolves a token through its documented variables, the SDK
  re-injects `COPILOT_SDK_AUTH_TOKEN` from that resolved value, so
  authenticated runs are unchanged.
- **Replacement:** Set one of the documented token variables
  (`GITHUB_TOKEN`, `COPILOT_AGENT_TOKEN`, `COPILOT_GITHUB_TOKEN`, or
  `GH_TOKEN`) so the provider resolves the token explicitly. The SDK
  will then inject `COPILOT_SDK_AUTH_TOKEN` into the subprocess on the
  provider's behalf.
- **Rollback:** Pin provider `==2.2.0` with `github-copilot-sdk==1.0.0b4`.

---

## When

Provider version `2.3.0`.

---

## Rollback

If the new SDK or env-var behavior breaks your workflow, pin provider
`==2.2.0` with `github-copilot-sdk==1.0.0b4`. The two cannot be mixed.

---

## Older versions (≤ v2.1.0)

Migration guides for upgrades onto **v2.0.0** and **v2.1.0** were retired when the
SDK reached GA (provider v2.4.0 / `github-copilot-sdk==1.0.2`). They remain in git
history — e.g. `git show v2.1.0:MIGRATION.md` or `git show v2.0.0:MIGRATION.md`.

---

# Migration Guide: v2.1.x → v2.2.0 (filesystem-layout V1.0)

## Overview

v2.2.0 introduces an explicit, contract-anchored filesystem layout for the
provider (see `contracts/filesystem-layout.md`). Two user-visible changes:

1. **Provider home is now provider-owned** — SDK subprocess state
   (`session-store.db`, `session-state/`, `config.json`, `logs/`) is written
   under a provider-owned directory instead of the default `~/.copilot/`.
2. **Cache directory was renamed** to a single flat distribution-name segment.

Existing files under `~/.copilot/` are not deleted, but the new provider
does not read them — prior session, auth, and CLI logs effectively start
fresh under the new `provider_home`. Users authenticated via the
documented `GITHUB_TOKEN` flow are unaffected; users who relied on
undocumented `~/.copilot/` state should re-authenticate on first call
(`export GITHUB_TOKEN=$(gh auth token)`). Auto-migration is forbidden
by the contract (`Lifecycle:MUST:4`); the legacy directories may be
deleted at the user's leisure.

## Cache directory rename

| OS      | v2.1.x (old)                                                | v2.2.0 (new)                                          |
|---------|-------------------------------------------------------------|-------------------------------------------------------|
| Linux   | `~/.cache/amplifier/provider-github-copilot/`               | `~/.cache/amplifier-provider-github-copilot/`         |
| macOS   | `~/Library/Caches/amplifier/provider-github-copilot/`       | `~/Library/Caches/amplifier-provider-github-copilot/` |
| Windows | `%LOCALAPPDATA%\amplifier\provider-github-copilot\`         | `%LOCALAPPDATA%\amplifier-provider-github-copilot\Cache\` |

**Behavior on upgrade:** the only file written here is `models_cache.json`
(regenerable from `provider.list_models()` on first call). First call after
upgrade will repopulate the new path; the old path is orphaned but harmless.

**Optional cleanup** (Linux example):

```bash
rm -rf ~/.cache/amplifier/provider-github-copilot
```

## Provider home introduction

SDK state previously written to `~/.copilot/` is now written under a
provider-owned `provider_home`. The V1.0 contract (`filesystem-layout.md`
§Paths:MUST:1) defines a **platform-uniform** resolution chain — there is no
per-OS branching for the fallback:

1. `${AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME}` if set, non-empty, and
   absolute after `Path.expanduser()`.
2. `${XDG_DATA_HOME}/amplifier-provider-github-copilot/` if `XDG_DATA_HOME`
   is set, non-empty, and absolute.
3. `~/.amplifier-provider-github-copilot/` (dot-prefixed directory under
   the user's home, on **all** platforms — Linux, macOS, Windows).

Examples of resolved paths:

| Scenario                           | Resolved `provider_home`                                |
|------------------------------------|---------------------------------------------------------|
| Override env set                   | the absolute override path                              |
| Linux with `XDG_DATA_HOME` set     | `$XDG_DATA_HOME/amplifier-provider-github-copilot/`     |
| Linux without `XDG_DATA_HOME`      | `~/.amplifier-provider-github-copilot/`                 |
| macOS (default)                    | `~/.amplifier-provider-github-copilot/`                 |
| Windows (default)                  | `~\.amplifier-provider-github-copilot\`                 |

Note that `cache_home` follows a different (platform-aware) chain — see
`contracts/filesystem-layout.md` §Paths:MUST:2 for the full table.

**Behavior on upgrade:** any session state previously stored at `~/.copilot/`
remains in place but is not read by this provider. Users who relied on that
state (rare — `session-store.db` is per-bundle and regenerates) may delete it
or re-authenticate via `GITHUB_TOKEN` (the documented auth flow since v2.0.0).

## What did NOT change

- Authentication (`GITHUB_TOKEN`) flow.
- Public API symbols.
- Cache schema or `models_cache.json` format.
- Cache TTL or invalidation policy.
