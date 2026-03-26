import re
import logging

logger = logging.getLogger("Guardrails")

DANGEROUS_PATTERNS = [
    (r"\bos\.system\s*\(", "os.system() call"),
    (r"\bos\.popen\s*\(", "os.popen() call"),
    (r"\bos\.remove\s*\(", "os.remove() call"),
    (r"\bos\.unlink\s*\(", "os.unlink() call"),
    (r"\bos\.rmdir\s*\(", "os.rmdir() call"),
    (r"\bshutil\.rmtree\s*\(", "shutil.rmtree() call"),
    (r"(?<!\.)eval\s*\(", "eval() call"),
    (r"(?<!\.)exec\s*\(", "nested exec() call"),
]


def check_code_safety(code):
    violations = []
    for pattern, description in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            violations.append(description)
    is_safe = len(violations) == 0
    if not is_safe:
        logger.warning(f"Code safety violations: {violations}")
    return is_safe, violations


def get_restricted_builtins():
    import builtins
    safe = dict(vars(builtins))
    for name in ("eval", "exec", "compile"):
        safe.pop(name, None)
    return safe
