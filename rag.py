import math
import re
import inspect
import importlib
import pkgutil
import logging

logger = logging.getLogger("RAG")


class KnowledgeBase:
    def __init__(self):
        self.documents = []
        self.versions = {}
        self._index_dirty = True
        self._tokenized = []
        self._avg_dl = 0
        self._idf = {}

    def index_library(self, package_name, max_docs=2000):
        try:
            mod = importlib.import_module(package_name)
        except ImportError:
            logger.warning(f"Cannot import {package_name}, skipping")
            return 0

        try:
            from importlib.metadata import version
            ver = version(package_name)
        except Exception:
            ver = getattr(mod, "__version__", "unknown")

        self.versions[package_name] = ver
        count = 0

        import io, contextlib
        devnull = io.StringIO()

        for name, obj in self._walk_module(mod, package_name):
            if count >= max_docs:
                break
            try:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    doc = inspect.getdoc(obj)
            except Exception:
                continue
            if not doc or len(doc) < 20:
                continue
            try:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    sig = str(inspect.signature(obj))
            except (ValueError, TypeError):
                sig = ""

            header = f"{package_name} {ver} — {name}"
            text = f"{header}\n{name}{sig}\n{doc[:500]}"
            self.documents.append(text)
            count += 1

        self._index_dirty = True
        logger.info(f"Indexed {package_name} {ver}: {count} docs")
        return count

    _SKIP = {"f2py", "testing", "tests", "test", "_", "conftest", "setup", "distutils", "compat"}

    def _walk_module(self, mod, prefix, depth=0):
        if depth > 2:
            return
        seen = set()
        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            try:
                obj = getattr(mod, attr_name)
            except Exception:
                continue
            full_name = f"{prefix}.{attr_name}"
            obj_id = id(obj)
            if obj_id in seen:
                continue
            seen.add(obj_id)

            if inspect.isclass(obj) or inspect.isfunction(obj):
                yield full_name, obj
                if inspect.isclass(obj):
                    for method_name in dir(obj):
                        if method_name.startswith("_"):
                            continue
                        try:
                            method = getattr(obj, method_name)
                        except Exception:
                            continue
                        if callable(method):
                            yield f"{full_name}.{method_name}", method

        if hasattr(mod, "__path__"):
            try:
                for importer, modname, ispkg in pkgutil.iter_modules(mod.__path__):
                    if any(s in modname.lower() for s in self._SKIP):
                        continue
                    try:
                        submod = importlib.import_module(f"{prefix}.{modname}")
                        yield from self._walk_module(submod, f"{prefix}.{modname}", depth + 1)
                    except Exception:
                        continue
            except Exception:
                pass

    def _rebuild_index(self):
        self._tokenized = [self._tokenize(doc) for doc in self.documents]
        if self._tokenized:
            self._avg_dl = sum(len(d) for d in self._tokenized) / len(self._tokenized)
        else:
            self._avg_dl = 1
        n = len(self._tokenized)
        df = {}
        for doc in self._tokenized:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self._idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
        self._index_dirty = False

    def _tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def _score(self, query_tokens, doc_tokens, k1=1.5, b=0.75):
        dl = len(doc_tokens)
        tf = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for term in query_tokens:
            if term in tf:
                freq = tf[term]
                score += self._idf.get(term, 0) * freq * (k1 + 1) / (
                    freq + k1 * (1 - b + b * dl / self._avg_dl)
                )
        return score

    def search(self, query, top_k=5):
        if self._index_dirty:
            self._rebuild_index()
        if not self.documents:
            return []
        tokens = self._tokenize(query)
        scores = [(i, self._score(tokens, doc)) for i, doc in enumerate(self._tokenized)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [self.documents[i] for i, s in scores[:top_k] if s > 0]

    def get_versions_summary(self):
        return "\n".join(f"  {pkg}: {ver}" for pkg, ver in self.versions.items())
