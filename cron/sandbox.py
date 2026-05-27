"""
Kernel-level sandboxing for cron job scripts using nono.

Provides OS-enforced isolation (Landlock on Linux, Seatbelt on macOS)
for pre-run scripts and no_agent scripts. The gateway process stays
unsandboxed; only the child process executing the script is restricted.

Configuration lives in two places:
  - Global defaults: config.yaml → cron.sandbox
  - Per-job overrides: jobs.json  → job.sandbox

Per-job config inherits from global defaults. Explicit per-job fields
override the corresponding global default.

Requires: pip install nono-py
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import nono_py as nono
    HAS_NONO = True
except ImportError:
    nono = None  # type: ignore[assignment]
    HAS_NONO = False


def is_available() -> bool:
    """Check if nono is installed and the platform supports sandboxing."""
    if not HAS_NONO:
        return False
    try:
        return nono.is_supported()
    except Exception:
        return False


def _resolve_sandbox_config(
    job: Dict[str, Any],
    global_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Merge per-job sandbox config with global defaults from config.yaml.

    Returns None if sandboxing is disabled for this job.

    Precedence (highest to lowest):
      1. job["sandbox"] fields
      2. config.yaml cron.sandbox.defaults
      3. Built-in defaults (deny-all)

    A job can opt out with {"sandbox": {"enabled": false}} even when
    the global default enables sandboxing.
    """
    cron_cfg = global_config.get("cron", {}) or {}
    global_sandbox = cron_cfg.get("sandbox", {}) or {}
    global_defaults = global_sandbox.get("defaults", {}) or {}
    global_enabled = global_sandbox.get("enabled", False)

    job_sandbox = job.get("sandbox") or {}
    job_enabled = job_sandbox.get("enabled")

    # Determine if sandbox is active for this job
    if job_enabled is False:
        return None
    if job_enabled is None and not global_enabled:
        return None

    # Merge: job overrides global defaults
    merged: Dict[str, Any] = {}

    # Filesystem
    global_fs = global_defaults.get("filesystem", {}) or {}
    job_fs = job_sandbox.get("filesystem", {}) or {}
    merged["filesystem"] = {
        "allow_read": job_fs.get("allow_read") or global_fs.get("allow_read", []),
        "allow_write": job_fs.get("allow_write") or global_fs.get("allow_write", []),
    }

    # Network
    global_net = global_defaults.get("network", {}) or {}
    job_net = job_sandbox.get("network", {}) or {}
    merged["network"] = {
        "allow_hosts": job_net.get("allow_hosts") or global_net.get("allow_hosts", []),
    }

    # Credentials (proxy injection)
    global_creds = global_defaults.get("credentials", {}) or {}
    job_creds = job_sandbox.get("credentials", {}) or {}
    # Job credentials override global per-key
    all_creds = {**global_creds, **job_creds}
    merged["credentials"] = all_creds

    # Audit
    global_audit = global_defaults.get("audit", {}) or {}
    job_audit = job_sandbox.get("audit", {}) or {}
    merged["audit"] = {
        "enabled": job_audit.get("enabled", global_audit.get("enabled", False)),
        "dir": job_audit.get("dir") or global_audit.get("dir"),
    }

    return merged


def _expand_path(p: str) -> str:
    """Expand ~ and environment variables in a path string."""
    return str(Path(os.path.expandvars(os.path.expanduser(p))).resolve())


def _resolve_profile_paths(hermes_home: str) -> Dict[str, str]:
    """Determine the active profile's directory and the base hermes dir.

    Returns a dict with:
      - 'profile_dir': the active profile's data directory (read-write)
      - 'base_dir': the base ~/.hermes directory
      - 'shared_dirs': directories shared across profiles (code, scripts)
      - 'other_profiles': path to other profiles dir (to deny)
    """
    hermes_home_path = Path(hermes_home).resolve()

    # Detect if we're running inside a named profile
    # Named profiles live at ~/.hermes/profiles/<name>/
    profiles_parent = hermes_home_path.parent
    if profiles_parent.name == "profiles":
        base_dir = str(profiles_parent.parent)
        profile_name = hermes_home_path.name
    else:
        base_dir = str(hermes_home_path)
        profile_name = None

    profiles_dir = str(Path(base_dir) / "profiles")

    # Shared read-only paths that all profiles need
    shared_dirs = [
        str(Path(base_dir) / "hermes-agent"),  # code
        str(Path(base_dir) / "scripts"),        # pre-run scripts
    ]

    return {
        "profile_dir": hermes_home,
        "profile_name": profile_name,
        "base_dir": base_dir,
        "profiles_dir": profiles_dir,
        "shared_dirs": shared_dirs,
    }


def _build_capabilities(
    sandbox_cfg: Dict[str, Any],
    script_dir: str,
    hermes_home: str,
) -> "nono.CapabilitySet":
    """Build a nono CapabilitySet from merged sandbox config.

    Always grants:
      - Read access to the script's directory
      - Read access to system Python/bash paths needed for the interpreter
      - Read-write to the ACTIVE profile's directory only
      - Read to shared dirs (code, scripts)
      - NO access to other profiles' directories
    """
    caps = nono.CapabilitySet()

    # Always allow reading the script directory
    caps.allow_path(script_dir, nono.AccessMode.READ)

    # --- Profile isolation ---
    # Only allow access to the active profile's data directory.
    # Other profiles' memories, sessions, and configs are invisible.
    profile_paths = _resolve_profile_paths(hermes_home)

    profile_dir = profile_paths["profile_dir"]
    base_dir = profile_paths["base_dir"]
    profile_name = profile_paths.get("profile_name")

    if profile_name:
        # Named profile: grant only its own directory
        if os.path.isdir(profile_dir):
            caps.allow_path(profile_dir, nono.AccessMode.READ_WRITE)
    else:
        # Default profile: grant individual subdirectories, NOT the
        # entire ~/.hermes/ tree (which would include profiles/).
        _default_profile_dirs = [
            "cron", "sessions", "memories", "skills", "hooks",
            "logs", "audio_cache", "image_cache", "workspace",
            "plans", "pairing", "skins", "home",
        ]
        for subdir in _default_profile_dirs:
            subpath = os.path.join(base_dir, subdir)
            if os.path.isdir(subpath):
                caps.allow_path(subpath, nono.AccessMode.READ_WRITE)

    # Shared code and scripts get read-only
    for shared in profile_paths["shared_dirs"]:
        if os.path.isdir(shared):
            caps.allow_path(shared, nono.AccessMode.READ)

    # Base config files (config.yaml, .env) need read access
    for cfg_file in ["config.yaml", ".env", "SOUL.md"]:
        cfg_path = os.path.join(base_dir, cfg_file)
        if os.path.isfile(cfg_path):
            caps.allow_file(cfg_path, nono.AccessMode.READ)

    # NOTE: ~/.hermes/profiles/ is never granted to the default profile,
    # and named profiles only get their own dir — Landlock denies the rest.

    # System paths required for interpreter execution (bash, python, libs)
    _system_read_paths = [
        "/bin", "/usr/bin", "/usr/lib", "/usr/lib64",
        "/lib", "/lib64", "/etc/alternatives",
        "/usr/share/python3", "/usr/share/bash-completion",
        "/etc/ssl", "/etc/ca-certificates", "/usr/share/ca-certificates",
        "/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf",
        "/proc/self", "/proc/version",
        "/tmp",
    ]
    # Add the Python venv if we're running from one
    _venv = os.environ.get("VIRTUAL_ENV")
    if _venv and os.path.isdir(_venv):
        _system_read_paths.append(_venv)

    # /dev needs read-write for /dev/null, /dev/urandom etc.
    if os.path.isdir("/dev"):
        caps.allow_path("/dev", nono.AccessMode.READ_WRITE)

    for sp in _system_read_paths:
        if os.path.isdir(sp):
            caps.allow_path(sp, nono.AccessMode.READ)
        elif os.path.isfile(sp):
            caps.allow_file(sp, nono.AccessMode.READ)

    fs_cfg = sandbox_cfg.get("filesystem", {})

    for p in fs_cfg.get("allow_read", []):
        expanded = _expand_path(p)
        if os.path.isdir(expanded):
            caps.allow_path(expanded, nono.AccessMode.READ)
        elif os.path.isfile(expanded):
            caps.allow_file(expanded, nono.AccessMode.READ)

    for p in fs_cfg.get("allow_write", []):
        expanded = _expand_path(p)
        if os.path.isdir(expanded):
            caps.allow_path(expanded, nono.AccessMode.READ_WRITE)
        elif os.path.isfile(expanded):
            caps.allow_file(expanded, nono.AccessMode.READ_WRITE)
        else:
            parent = str(Path(expanded).parent)
            if os.path.isdir(parent):
                caps.allow_path(parent, nono.AccessMode.READ_WRITE)

    net_cfg = sandbox_cfg.get("network", {})
    if not net_cfg.get("allow_hosts"):
        caps.block_network()

    return caps


def _build_env(
    sandbox_cfg: Dict[str, Any],
    base_env: Dict[str, str],
) -> List[Tuple[str, str]]:
    """Build a minimal env var list for the sandboxed child.

    sandboxed_exec does NOT inherit parent env by default.
    We pass through only what the script needs.
    """
    safe_keys = {
        "PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM",
        "HERMES_HOME", "TMPDIR", "TZ",
        "PYTHONPATH", "PYTHONIOENCODING",
    }

    env_list = []
    for key in safe_keys:
        val = base_env.get(key)
        if val is not None:
            env_list.append((key, val))

    return env_list


def _start_credential_proxy(
    sandbox_cfg: Dict[str, Any],
) -> Optional[Any]:
    """Start a nono network proxy with credential injection if configured.

    Each credential entry in the config maps to a RouteConfig that
    intercepts requests and injects the real API key from an env var.
    The sandboxed process only sees phantom tokens.
    """
    creds = sandbox_cfg.get("credentials", {})
    net_cfg = sandbox_cfg.get("network", {})
    allowed_hosts = net_cfg.get("allow_hosts", [])

    if not creds and not allowed_hosts:
        return None

    routes = []
    for name, cred_cfg in creds.items():
        upstream = cred_cfg.get("upstream")
        if not upstream:
            continue

        real_key = os.environ.get(cred_cfg.get("key_env", ""), "")
        if not real_key:
            logger.warning("Sandbox credential '%s': env var '%s' not set, skipping proxy route",
                           name, cred_cfg.get("key_env", ""))
            continue

        route = nono.RouteConfig(
            prefix=cred_cfg.get("prefix", "/"),
            upstream=upstream,
            credential_key=f"hermes-cron-{name}",
            inject_mode=nono.InjectMode.HEADER,
            inject_header=cred_cfg.get("inject_header", "Authorization"),
            credential_format=cred_cfg.get("format", "Bearer {credential}"),
        )
        routes.append(route)

    if not routes and not allowed_hosts:
        return None

    try:
        config = nono.ProxyConfig(
            allowed_hosts=allowed_hosts,
            routes=routes,
        )
        return nono.start_proxy(config)
    except Exception as exc:
        logger.warning("Failed to start nono proxy: %s", exc)
        return None


def run_sandboxed_script(
    argv: List[str],
    script_dir: str,
    hermes_home: str,
    base_env: Dict[str, str],
    timeout_secs: float,
    sandbox_cfg: Dict[str, Any],
) -> Tuple[bool, str]:
    """Execute a script inside a nono sandbox.

    Returns (success, output) matching _run_job_script's contract.
    """
    if not is_available():
        logger.warning("nono sandbox requested but not available, falling back to unsandboxed execution")
        return _fallback_unsandboxed(argv, script_dir, base_env, timeout_secs)

    caps = _build_capabilities(sandbox_cfg, script_dir, hermes_home)
    env_list = _build_env(sandbox_cfg, base_env)

    proxy = _start_credential_proxy(sandbox_cfg)
    try:
        if proxy:
            proxy_env = proxy.sandbox_env()
            env_list.extend(proxy_env.items() if hasattr(proxy_env, 'items') else proxy_env)

        logger.info("Running script in nono sandbox: %s", " ".join(argv))
        result = nono.sandboxed_exec(
            caps,
            argv,
            cwd=script_dir,
            timeout_secs=timeout_secs,
            env=env_list,
        )

        stdout = (result.stdout.decode() if isinstance(result.stdout, bytes) else result.stdout or "").strip()
        stderr = (result.stderr.decode() if isinstance(result.stderr, bytes) else result.stderr or "").strip()

        try:
            from agent.redact import redact_sensitive_text
            stdout = redact_sensitive_text(stdout)
            stderr = redact_sensitive_text(stderr)
        except Exception:
            pass

        if proxy:
            try:
                events = proxy.drain_audit_events()
                if events:
                    logger.info("Sandbox audit: %d network events for script", len(events))
            except Exception:
                pass

        if result.exit_code != 0:
            parts = [f"Script exited with code {result.exit_code} (sandboxed)"]
            if stderr:
                parts.append(f"stderr:\n{stderr}")
            if stdout:
                parts.append(f"stdout:\n{stdout}")
            return False, "\n".join(parts)

        return True, stdout

    except Exception as exc:
        return False, f"Sandboxed script execution failed: {exc}"

    finally:
        if proxy:
            try:
                proxy.shutdown()
            except Exception:
                pass


def _fallback_unsandboxed(
    argv: List[str],
    cwd: str,
    env: Dict[str, str],
    timeout: float,
) -> Tuple[bool, str]:
    """Fallback to subprocess.run when nono is not available."""
    import subprocess
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        try:
            from agent.redact import redact_sensitive_text
            stdout = redact_sensitive_text(stdout)
            stderr = redact_sensitive_text(stderr)
        except Exception:
            pass

        if result.returncode != 0:
            parts = [f"Script exited with code {result.returncode}"]
            if stderr:
                parts.append(f"stderr:\n{stderr}")
            if stdout:
                parts.append(f"stdout:\n{stdout}")
            return False, "\n".join(parts)

        return True, stdout

    except subprocess.TimeoutExpired:
        return False, f"Script timed out after {timeout}s"
    except Exception as exc:
        return False, f"Script execution failed: {exc}"
