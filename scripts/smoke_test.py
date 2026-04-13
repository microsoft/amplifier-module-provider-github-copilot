#!/usr/bin/env python3
"""
Smoke Test for GitHub Copilot Provider
=======================================

A quick end-to-end verification script to validate the provider works with
the Copilot SDK. Use this during development to catch integration issues
early, before running the full test suite.

Why This Exists
---------------
The full test suite (1100+ tests) takes ~2 minutes. This smoke test runs in
~5 seconds and validates the critical path:

  1. Provider initializes correctly
  2. SDK binary is found (cross-platform: Windows .exe vs Unix)
  3. SDK authenticates successfully (via copilot credentials)
  4. list_models() returns available models
  5. Provider closes cleanly

When to Use
-----------
  - After pulling latest changes
  - After modifying _platform.py, sdk_adapter/client.py, or provider.py
  - After upgrading github-copilot-sdk
  - Before submitting a PR
  - When debugging "it works on my machine" issues

Usage
-----
From repo root:

    # Using make (recommended - WSL/Linux/macOS)
    make smoke

    # Direct execution (any platform)
    python scripts/smoke_test.py

    # With verbose output (shows model list)
    python scripts/smoke_test.py --verbose

Prerequisites
-------------
  - Copilot SDK authenticated: `copilot auth login`
  - Virtual environment activated with package installed

Exit Codes
----------
  0 = All checks passed
  1 = Provider initialization failed
  2 = No models returned (SDK connection issue)

For Full Testing
----------------
  pytest tests/ -q                    # Full suite
  make test                           # Full suite (via make)

"""
import argparse
import asyncio
import sys
import time

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_status(label: str, status: str, color: str = GREEN) -> None:
    """Print a formatted status line."""
    print(f"  {label:.<40} [{color}{status}{RESET}]")


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{BOLD}{CYAN}▶ {text}{RESET}")


async def run_smoke_test(verbose: bool = False) -> int:
    """
    Run the smoke test suite.

    Args:
        verbose: Print detailed output (model list, full SDK binary path)

    Returns:
        Exit code (0 = success)
    """
    start_time = time.time()

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  GitHub Copilot Provider - Smoke Test{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    # -------------------------------------------------------------------------
    # Test 1: Provider Initialization
    # -------------------------------------------------------------------------
    print_header("Provider Initialization")

    try:
        from amplifier_module_provider_github_copilot import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        print_status("Import and instantiate", "OK")
    except ImportError as e:
        print_status("Import provider", "FAILED", RED)
        print(f"\n{RED}Error: {e}{RESET}")
        print("Make sure the package is installed: pip install -e '.[dev]'")
        return 1
    except Exception as e:
        print_status("Instantiate provider", "FAILED", RED)
        print(f"\n{RED}Error: {e}{RESET}")
        return 1

    # -------------------------------------------------------------------------
    # Test 2: Platform Detection
    # -------------------------------------------------------------------------
    print_header("Platform Detection")

    try:
        from amplifier_module_provider_github_copilot._platform import (
            get_platform_info,
            locate_cli_binary,
        )

        platform_info = get_platform_info()
        print_status(f"Platform: {platform_info.name}", "OK")
        print_status(f"Binary name: {platform_info.cli_binary_name}", "OK")

        cli_path = locate_cli_binary()
        if cli_path:
            print_status(f"SDK binary found: {cli_path.name}", "OK")
            if verbose:
                print(f"    Full path: {cli_path}")
        else:
            print_status("SDK binary", "NOT FOUND", YELLOW)
            print(f"\n{YELLOW}Warning: Copilot SDK binary not in PATH. SDK may fail.{RESET}")
    except Exception as e:
        print_status("Platform detection", "FAILED", RED)
        if verbose:
            print(f"    Error: {e}")

    # -------------------------------------------------------------------------
    # Test 3: List Models (SDK Call)
    # -------------------------------------------------------------------------
    print_header("SDK Connection (list_models)")

    try:
        models = await provider.list_models()
        model_count = len(models)

        if model_count > 0:
            print_status(f"Models returned: {model_count}", "OK")
            if verbose:
                print("    Available models:")
                for m in models[:10]:  # Show first 10
                    print(f"      - {m.id}")
                if model_count > 10:
                    print(f"      ... and {model_count - 10} more")
        else:
            print_status("Models returned", "NONE", RED)
            await provider.close()
            return 2

    except Exception as e:
        print_status("list_models()", "FAILED", RED)
        print(f"\n{RED}Error: {e}{RESET}")
        print("\nPossible causes:")
        print("  - Copilot SDK not authenticated: run 'copilot auth login'")
        print("  - Network connectivity issues")
        print("  - Copilot service outage")
        await provider.close()
        return 1

    # NOTE: Completion test is skipped because it requires amplifier-foundation
    # and ChatRequest construction. The list_models() test validates:
    #   ✓ SDK binary discovery (cross-platform)
    #   ✓ SDK authentication (via copilot credentials)
    #   ✓ Network connectivity to Copilot service
    #   ✓ Provider lifecycle (init → operation → close)
    # For full completion testing, run: pytest tests/test_provider.py -v

    # -------------------------------------------------------------------------
    # Test 4: Clean Shutdown
    # -------------------------------------------------------------------------
    print_header("Cleanup")

    try:
        await provider.close()
        print_status("Provider closed", "OK")
    except Exception as e:
        print_status("Provider close", "FAILED", RED)
        if verbose:
            print(f"    Error: {e}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{GREEN}✓ Smoke test passed{RESET} in {elapsed:.2f}s")
    print(f"{BOLD}{'='*60}{RESET}\n")

    return 0


def main() -> int:
    """Entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Smoke test for GitHub Copilot Provider",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/smoke_test.py              # Quick test
  python scripts/smoke_test.py --verbose    # Detailed output (shows model list)
  
For full completion testing, use:
  pytest tests/ -q --tb=short
        """,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed output"
    )

    args = parser.parse_args()

    return asyncio.run(run_smoke_test(verbose=args.verbose))


if __name__ == "__main__":
    sys.exit(main())
