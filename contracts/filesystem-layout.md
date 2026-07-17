# Contract: Filesystem Layout

**Version:** 1.2
**Status:** Normative
**Implementation:** `amplifier_module_provider_github_copilot/config/_paths.py`,
`amplifier_module_provider_github_copilot/sdk_adapter/client.py::CopilotClientWrapper._ensure_client_initialized`
**Update:** 2026-05-30 — SDK pin bumped b9 → b10. All v1.1 Wiring/Isolation/Identity/Paths/Lifecycle/Acceptance/Portability MUST clauses verified byte-compatible against b10: `CopilotClient.__init__` signature unchanged (b10 `client.py:1073-1089`), `base_directory` semantics unchanged (b10 `client.py:1081,1108-1111`), `env=` REPLACE semantics unchanged (b10 `client.py:3182-3185`), `COPILOT_HOME` set from `base_directory` unchanged (b10 `client.py:3199`). No production source change required by filesystem layout; the b10 bump adds 9 new MinimalMode kwargs that ride on the unchanged base_directory/env wiring.
**History:**
  - **1.2** — SDK v1.0.0b10: no filesystem-layout clauses changed. All v1.1 wiring (base_directory + env REPLACE + COPILOT_HOME) survived byte-compatible against b10; v1.1 line citations shifted in b10 (env REPLACE 2937-2940 → 3182-3185; COPILOT_HOME 2952-2953 → 3199; base_directory 1059 → 1081 and 1083-1086 → 1108-1111; CopilotClient.__init__ 1051-1067 → 1073-1089) but every semantic invariant survived byte-compatible against b10. This row affirms the b10 evaluation; the 9 b10 MinimalMode pins live in `contracts/sdk-boundary.md` v1.10, not here.
  - **1.1** — SDK v1.0.0b9: Wiring:MUST:1/2/3/5 rewritten — `SubprocessConfig(copilot_home=...)` replaced by `CopilotClient(base_directory=..., github_token=..., log_level=..., env=...)` direct kwargs. Line citations bumped to b9: client.py:1059 (base_directory kwarg), 1083-1086 (base_directory docstring), 2937-2940 (env REPLACE), 2942-2943 (COPILOT_SDK_AUTH_TOKEN), 2952-2953 (COPILOT_HOME from base_directory). Isolation:MUST:1 Test Anchors restructured to use `monkeypatch.setattr(client_mod, "CopilotClient", FakeCopilotClient)` (scoped module patch, not class-level).
  - **1.0** — Initial contract (Wiring/Isolation/Identity/Paths/Lifecycle/Acceptance/Portability MUST clauses).

## Overview

The provider redirects every file the underlying `github-copilot-sdk`
would write to its default `~/.copilot/` into a provider-owned home
directory. The provider does not depend on, read from, or write to any
host application's state directory.

## Identity

- **MUST:1.** `PROVIDER_ID = "github-copilot"` is the registry-facing
  identity. The literal `"github-copilot"` MUST appear in package
  source exactly once, at `config/_models.py:PROVIDER["id"]`, which is
  the shared origin of both `PROVIDER_ID` (the import-time constant
  exported by `_identity.py`) and `ProviderConfig.provider_id` (the
  runtime view built by `config_loader.load_provider_config`).
  `PROVIDER_ID`, `ProviderConfig.provider_id`, `Provider().name`, and
  `Provider().get_info().id` MUST be equal. References to the
  provider id elsewhere in the package MUST import `PROVIDER_ID` from
  `_identity` or read it via `ProviderConfig.provider_id`; the
  literal `"github-copilot"` MUST NOT be reproduced.
- **MUST:2.** `PROVIDER_DISTRIBUTION_NAME = "amplifier-provider-github-copilot"`
  is the filesystem-facing identity, defined exactly once in
  `config/_paths.py`. The literal MUST NOT be reproduced; all
  derivations MUST reference the constant.
- **MUST:3.** Mechanical derivations:
  - Override env var: `PROVIDER_DISTRIBUTION_NAME.upper().replace("-", "_") + "_HOME"`
    → `AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME`.
  - Dotted home dirname: `"." + PROVIDER_DISTRIBUTION_NAME`.
  - XDG / cache subdirectory: `PROVIDER_DISTRIBUTION_NAME` verbatim.

In normative path rules below, `${PROVIDER_DISTRIBUTION_NAME}` denotes
the MUST:2 constant, `amplifier-provider-github-copilot`. Packaging
entry-point keys (`pyproject.toml`) and registry manifest
`provider.id` (YAML) are governed by `provider-protocol.md`; this
contract does not bind them.

## Paths

- **MUST:1.** `provider_home` resolves in this order:
  1. `${AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME}` if non-empty after
     `str.strip()`. The stripped value, after `Path.expanduser()`,
     MUST be absolute; if it is non-empty but non-absolute the
     provider MUST raise `ValueError`.
  2. `${XDG_DATA_HOME}/${PROVIDER_DISTRIBUTION_NAME}` if
     `XDG_DATA_HOME` is non-empty after `str.strip()` and absolute.
     If `XDG_DATA_HOME` is non-empty but non-absolute the provider
     MUST raise `ValueError`.
  3. `~/.${PROVIDER_DISTRIBUTION_NAME}`.

  At every level, an *empty or whitespace-only* value MUST fall
  through to the next level (POSIX-XDG convention). The final
  returned path MUST be absolute.
- **MUST:2.** `cache_home` resolves in this order, with the same
  fail-closed rule on non-absolute non-empty values:
  1. `${XDG_CACHE_HOME}/${PROVIDER_DISTRIBUTION_NAME}`.
  2. `~/Library/Caches/${PROVIDER_DISTRIBUTION_NAME}` when
     `sys.platform == "darwin"`.
  3. `~/.cache/${PROVIDER_DISTRIBUTION_NAME}` when
     `sys.platform not in {"darwin", "win32"}`.
  4. `${LOCALAPPDATA}/${PROVIDER_DISTRIBUTION_NAME}/Cache` when
     `sys.platform == "win32"`; if `LOCALAPPDATA` is unset, fall back
     to `Path.home() / "AppData" / "Local" / PROVIDER_DISTRIBUTION_NAME / "Cache"`.
- **MUST:3.** `provider_home` and `cache_home` MUST NOT be equal, and
  neither MUST be contained in the other.
- **MUST:4.** `Path.expanduser()` MUST be applied to any environment
  value beginning with `~`.
- **MUST:5.** `ProviderPaths` field reassignment after construction
  MUST raise (`@dataclass(frozen=True)`).

## Isolation

- **MUST:1.** The provider MUST NOT read, default to, fall back to, or
  probe any of: `AMPLIFIER_HOME`, `AMPLIFIER_APP_CLI_HOME`,
  `AMPLIFIER_DISTRO_HOME`, `COPILOT_HOME`, or `COPILOT_CLI_PATH`.
  Any `AMPLIFIER_*` variable not prefixed with
  `AMPLIFIER_PROVIDER_GITHUB_COPILOT_` is host-owned. `COPILOT_HOME`
  is owned by the standalone GitHub Copilot CLI; the provider only
  writes it into the spawned subprocess (`Wiring:MUST:1`) and never
  inherits it. `COPILOT_CLI_PATH` steers the SDK's CLI-binary
  selection at spawn time and MUST be scrubbed from the subprocess
  env (`Wiring:MUST:5`).
- **MUST:2.** The provider MUST NOT read, default to, fall back to, or
  probe `~/.amplifier/` or `~/.copilot/`, or any subdirectory thereof
  (including the host-shared cache subtree `${XDG_CACHE_HOME|LOCALAPPDATA|~/.cache|~/Library/Caches}/amplifier/...`),
  except for the negative-space assertion in
  `filesystem-layout:Acceptance:MUST:1`.
- **MUST:3.** A host MAY inject an absolute `provider_home` /
  `cache_home` through `ProviderPaths` constructor arguments;
  injection is the only permitted form of host-provider path coupling.
  The host integration contract defines the injection point.

## Wiring

- **MUST:1.** The token branch of `_ensure_client_initialized` MUST
  call `CopilotClient(base_directory=str(load_provider_paths().provider_home),
  github_token=..., log_level=..., env=...)`. b10 wires `base_directory` into
  the spawned runtime as `COPILOT_HOME` (`copilot/client.py:3199`); the
  `base_directory` kwarg is declared at `copilot/client.py:1081` and documented
  at `copilot/client.py:1108-1111`.
- **MUST:2.** The no-token branch MUST pass the same `base_directory`
  value as the token branch and MUST NOT pass `github_token`.
- **MUST:3.** `load_provider_paths()` MUST be uncached. A change to
  any of `AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME`, `XDG_DATA_HOME`,
  `XDG_CACHE_HOME`, `LOCALAPPDATA`, `HOME`, or `USERPROFILE` between
  two `_ensure_client_initialized` calls MUST be reflected in the
  next `CopilotClient(base_directory=...)` construction.
- **MUST:4.** Every provider-owned regenerable artifact MUST resolve
  its directory through `load_provider_paths().cache_home`. No module
  in the package MAY synthesize a cache-base path; precedence and
  platform branching live exclusively in `config/_paths.py`.
- **MUST:5.** `CopilotClient(env=...)` MUST be set to an explicit dict
  derived from `os.environ` with `COPILOT_HOME` and `COPILOT_CLI_PATH`
  removed. Relying on the SDK's `env=None` default is forbidden: b10 treats
  `env=` as full REPLACE -- `copilot/client.py:3182-3185` reads
  `env = dict(opts.env)` when the caller supplies a dict, otherwise
  `dict(os.environ)`. The SDK then sets `env["COPILOT_HOME"] =
  opts.base_directory` (`copilot/client.py:3199`) and
  `env["COPILOT_SDK_AUTH_TOKEN"] = opts.github_token`
  (`copilot/client.py:3189`) AFTER consuming the caller's env, so the
  scrub MUST remove `COPILOT_HOME` before the call to guarantee no ambient
  leak. If a future SDK release switches to env-merge semantics, the provider
  MUST be updated to re-scrub after merge or fail closed.

## Lifecycle

- **MUST:1 (POSIX, macOS).** `provider_home` and `cache_home` MUST be
  created lazily and idempotently before first use. The implementation
  MUST NOT mutate the process-global `os.umask`. For each leaf path,
  the implementation MUST:
  1. call `Path.mkdir(parents=True, exist_ok=True)` to materialize
     the subtree;
  2. then call `os.lstat()` on the leaf and refuse with `OSError` if
     it is a symbolic link;
  3. then call `os.chmod(leaf, 0o700)` on the leaf only. Parents
     inherit the user's normal modes; only the leaf carries the
     privacy guarantee.
- **MUST:2 (Windows).** `Path.mkdir(parents=True, exist_ok=True)`
  only. NTFS ACL inheritance is the supported confidentiality model;
  no `mode` argument is passed and no `chmod` call is made.
- **MUST:3.** When `provider_home` or `cache_home` collides with a
  non-directory, the exception raised at SDK initialization MUST
  carry `FileExistsError` or `NotADirectoryError` on its
  `__cause__` / `__context__` chain.
- **MUST:4.** The provider MUST NOT delete, rename, or auto-migrate
  `provider_home`, `cache_home`, or their contents on any code path.

## Portability

- **MUST:1.** `config/_paths.py` MUST import only from the Python
  standard library. No host package, no Amplifier package, no
  `github_copilot_sdk` symbols. Path resolution is inlined so the
  provider loads identically under every host and in standalone use.
- **MUST:2.** `[project].dependencies` in `pyproject.toml` MUST NOT
  include any package that wraps, embeds, or distributes this provider.

## Acceptance (live)

- **MUST:1.** Two sequential real `complete()` calls with
  `AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME` set to a tmp path MUST:
  (a) leave the user's `~/.copilot/` *path-set* (the set of
      `Path.relative_to(~/.copilot)` for every entry under it)
      unchanged across both calls, including remaining absent if it
      was absent.
  (b) leave `~/.amplifier/` path-set unchanged across both calls,
      including absent.
  (c) populate the configured `provider_home` with one or more
      SDK-written files after the first call.
  (d) preserve the post-first-call `provider_home` path-set as a
      subset of the post-second-call path-set (no deletion or rename
      between calls; new files MAY appear).
  (e) populate the configured `cache_home` with at least one
      provider-written file after the first call; the post-first AND
      post-second `provider_home` / `cache_home` path-sets MUST each
      be pairwise disjoint (no relative path appears under both); and
      the post-first `cache_home` path-set MUST be a subset of the
      post-second `cache_home` path-set (no deletion or rename
      between calls).

## Notes (non-normative)

- **Why two homes.** The SDK subprocess takes a single `COPILOT_HOME`
  and writes opaque state beneath it — session store, auth, config,
  CLI logs. That state lives under `provider_home` (XDG-DATA
  semantics: persistent user content). The provider's own regenerable
  artifacts — currently `models_cache.json`, a TTL-bounded model-list
  cache derivable from a live SDK call — live under `cache_home` (XDG-CACHE
  semantics). They are disjoint so deleting cache cannot lose session,
  auth, or log history. `XDG_CACHE_HOME`, if set, wins on every
  platform including macOS; platform defaults apply only when it is
  unset (Paths:MUST:2).
- The provider validates `provider_home` absoluteness, not ownership.
  Ownership / world-writable validation (via `os.fstat` on a directory
  fd plus mode-bit check) is a follow-up; pointing the override at a
  shared directory co-locates prompt text and is the user's
  responsibility.
- The POSIX `lstat`→`chmod` sequence has a check-then-act window if
  the parent is attacker-writable. Mitigation (`os.open` with
  `O_DIRECTORY | O_NOFOLLOW`, then `os.fchmod` on the fd, with inode
  revalidation across the sequence) is a follow-up; out of scope for
  single-user workstations.
- **Compatibility (non-binding).** The cascade
  `${OVERRIDE} → ${XDG_DATA_HOME}/<dist> → ~/.<dist>` for persistent
  state and `${XDG_CACHE_HOME}/<dist> → platform default →
  ~/.cache/<dist>` for regenerable cache follows the
  [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/latest/).
  Examples of XDG-aware CLI tools that adopt the same per-user
  cascade — listed for context only, not as a behavioral parity
  claim, since each tool's exact layout varies by platform and
  version — include `gh`, `git`, `aws`, `pip`, and `kubectl`. Each
  host process running as the same OS user therefore shares
  `provider_home` (auth, session, logs) and `cache_home` (model-list
  cache) with every other host of the same user — one identity, one
  regenerable cache. Hosts that require per-instance isolation use
  the `Isolation:MUST:3` constructor-injection escape hatch.

## Test Anchors

Each MUST has at least one test bearing a `# Contract:` marker citing
its anchor. Parametrized rows use explicit `pytest.param(..., id=…)`;
assertion messages name the violated rule fragment.

| Anchor | Test |
|---|---|
| `filesystem-layout:Identity:MUST:1` | `test_provider_id_equals_name_equals_get_info_id`; `test_provider_id_literal_appears_only_in_provider_config` (AST scan) |
| `filesystem-layout:Identity:MUST:2` | `test_distribution_name_literal_appears_only_in_paths_module` (AST scan over package source) |
| `filesystem-layout:Identity:MUST:3` | `test_env_var_dotdir_xdg_subdir_all_derived_from_distribution_name` |
| `filesystem-layout:Paths:MUST:1` | `test_provider_home_resolves_per_precedence_override` (parametrized: `override-abs-wins`, `override-empty-falls`, `override-ws-falls`, `override-relative-raises`, `override-tilde-expands`), `test_provider_home_resolves_per_precedence_xdg` (parametrized: `xdg-abs-wins`, `xdg-empty-falls`, `xdg-ws-falls`, `xdg-relative-raises`, `xdg-tilde-expands`), `test_provider_home_all_unset_default` |
| `filesystem-layout:Paths:MUST:2` | `test_cache_home_xdg_absolute_wins`, `test_cache_home_xdg_empty_falls_through`, `test_cache_home_xdg_whitespace_falls_through`, `test_cache_home_xdg_tilde_expands`, `test_cache_home_xdg_relative_raises`, `test_cache_home_localappdata_relative_raises`, `test_cache_home_localappdata_absolute_wins`, `test_cache_home_localappdata_unset_appdata_local_fallback`, `test_cache_home_darwin_default_library_caches`, `test_cache_home_linux_default_dot_cache` (in `test_filesystem_cache_wiring.py`) |
| `filesystem-layout:Paths:MUST:3` | `test_provider_home_and_cache_home_are_disjoint` |
| `filesystem-layout:Paths:MUST:4` | covered by `override-tilde-expands` and `xdg-tilde-expands` rows of the Paths:MUST:1 matrix and the `xdg-tilde-expands` row of the Paths:MUST:2 matrix |
| `filesystem-layout:Paths:MUST:5` | `test_provider_paths_is_immutable` |
| `filesystem-layout:Isolation:MUST:1` | `test_host_env_sentinel_never_appears_in_resolved_paths` (parametrized over `AMPLIFIER_HOME`, `AMPLIFIER_APP_CLI_HOME`, `AMPLIFIER_DISTRO_HOME`, `AMPLIFIER_RUNTIME_HOME`, `AMPLIFIER_FOUNDATION_HOME`, `COPILOT_HOME`, `COPILOT_CLI_PATH`); each row sets the sentinel in the parent env and asserts (i) `provider_home != sentinel`, (ii) the captured `CopilotClient(**kwargs)` call has `kwargs["base_directory"] == str(provider_home)`, `"COPILOT_HOME" not in kwargs["env"]`, and `"COPILOT_CLI_PATH" not in kwargs["env"]`, AND (iii) an AST scan over every `.py` in the package finds no `ast.Subscript` indexing `os.environ` and no `os.getenv` / `os.environ.get` call with key `"COPILOT_HOME"` or `"COPILOT_CLI_PATH"`. Sub-clause (ii) is additionally anchored by `test_filesystem_subprocess_env.py::test_token_branch_passes_provider_home_to_sdk`; sub-clause (iii) is additionally anchored by `test_filesystem_isolation.py::test_no_forbidden_env_reads_in_package`. The fixture interception point is `CopilotClient.__init__` kwargs capture via `monkeypatch.setattr(client_mod, "CopilotClient", FakeCopilotClient)`, scoped to the provider module under test. |
| `filesystem-layout:Isolation:MUST:2` | `test_dot_amplifier_and_dot_copilot_never_probed_or_resolved` |
| `filesystem-layout:Isolation:MUST:3` | `test_constructor_injection_overrides_env_resolution` (constructs `ProviderPaths(provider_home=<abs tmp>, cache_home=<abs tmp>)` and asserts wiring uses those values verbatim) |
| `filesystem-layout:Wiring:MUST:1` | `test_token_branch_passes_provider_home_to_sdk` (asserts the captured `CopilotClient(**kwargs)` call has `kwargs["base_directory"] == str(provider_home)`) |
| `filesystem-layout:Wiring:MUST:2` | `test_token_and_no_token_branches_agree_on_copilot_home` (asserts equality with the value captured in the `Wiring:MUST:1` fixture, and that the no-token branch does not pass `github_token`) |
| `filesystem-layout:Wiring:MUST:3` | `test_load_provider_paths_is_uncached_across_env_change` (asserts the next captured `CopilotClient(**kwargs)` call reflects the changed `kwargs["base_directory"]`) |
| `filesystem-layout:Wiring:MUST:4` | `test_models_cache_file_resolves_under_cache_home` (monkeypatches `config._paths.load_provider_paths` to return a `ProviderPaths` whose `cache_home` is an absolute tmp sentinel, then asserts the path the model-cache module would write to has that sentinel as a prefix — i.e. the assertion fails if the module recomputes precedence). Plus `test_no_cache_base_synthesis_outside_paths_module`: an AST scan over every `.py` in the package other than `config/_paths.py` MUST find no `ast.Subscript` indexing `os.environ` (or `os.getenv` call) with any of `{"LOCALAPPDATA", "XDG_CACHE_HOME", "XDG_DATA_HOME"}`, and no `ast.Compare` whose left is `Attribute(value=Name("sys"), attr="platform")` |
| `filesystem-layout:Wiring:MUST:5` | `test_subprocess_env_omits_copilot_home_and_cli_path` (sets `COPILOT_HOME=/sentinel/h` and `COPILOT_CLI_PATH=/sentinel/cli` in the parent env, asserts the captured `CopilotClient(**kwargs)` call passes an `env` dict, and that `"COPILOT_HOME" not in kwargs["env"]` and `"COPILOT_CLI_PATH" not in kwargs["env"]`) |
| `filesystem-layout:Lifecycle:MUST:1` | `test_provider_home_created_with_0700_on_posix`, `test_provider_home_creation_refuses_symlink_on_posix`, `test_creation_does_not_mutate_global_umask` |
| `filesystem-layout:Lifecycle:MUST:2` | `test_provider_home_creation_omits_mode_arg_on_windows` |
| `filesystem-layout:Lifecycle:MUST:3` | `test_collision_with_file_raises_FileExistsError` |
| `filesystem-layout:Lifecycle:MUST:4` | covered by `Acceptance:MUST:1` clauses (d) and (e): clause (d) asserts `provider_home` post-first ⊆ post-second; clause (e) asserts `cache_home` post-first ⊆ post-second and that `provider_home`/`cache_home` path-sets are pairwise disjoint after each call |
| `filesystem-layout:Portability:MUST:1` | `test_paths_module_imports_only_stdlib` (AST scan over `config/_paths.py`; allow-set = `sys.stdlib_module_names`) |
| `filesystem-layout:Portability:MUST:2` | `test_pyproject_dependencies_have_no_host_packages_at_runtime` |
| `filesystem-layout:Acceptance:MUST:1` | `test_live_two_call_session_redirects_sdk_files_and_preserves_legacy_path_sets` |

## Related

`provider-protocol.md` (identity), `sdk-boundary.md`
(`CopilotClient` wiring), `observability.md` (SDK log location).
