"""Deterministic documentation matching and module-local AST evidence."""
import ast
from pathlib import Path
import re

from .phrases import CON, NOCON, GENERIC_NAMES

PARSERS = {'ChartParser', 'RecursiveDescentParser', 'ShiftReduceParser',
           'EarleyChartParser', 'ViterbiParser', 'ProbabilisticChartParser',
           'StanfordParser', 'CoreNLPParser'}
TREE_ACCESS = {'label', 'leaves', 'subtrees', 'height', 'pretty_print', 'pformat'}
EXCLUDED = {'.git', '__pycache__', '.venv', 'venv', 'env', '.env', 'virtualenv'}


def phrase_matches(text):
    """Remove fenced blocks, then consume denial and positive spans longest first."""
    lines = []
    fenced = False
    for line in text.splitlines(keepends=True):
        if re.match(r'^\s*```', line):
            fenced = not fenced
            lines.append('\n')
        elif not fenced:
            lines.append(line)
    remaining = ''.join(lines).replace('`', '').lower()
    matches = []
    for kind, phrases in [('NOCON', NOCON), ('CON', CON)]:
        for phrase in sorted(phrases, key=len, reverse=True):
            start = remaining.find(phrase)
            while start >= 0:
                matches.append((kind, phrase))
                remaining = remaining[:start] + ' ' * len(phrase) + remaining[start + len(phrase):]
                start = remaining.find(phrase)
    return matches


class Evidence:
    """Small abstract interpreter: symbols and produced values never cross files."""

    def __init__(self, tree):
        self.shadowed = {n.name for n in ast.walk(tree)
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        self.found = False

    def bind(self, target, value, env):
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self.bind(item, None, env)

    def expr(self, node, env):
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return None if node.id in self.shadowed else env.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.expr(node.value, env)
            if base in ('document', 'sentence') and node.attr == 'constituency':
                self.found = True
            if base == 'tree' and node.attr in TREE_ACCESS:
                self.found = True
            if base == 'document' and node.attr == 'sentences':
                return 'sentences'
            if base and base.startswith(('symbol:', 'module:')):
                return 'symbol:' + base[7:] + '.' + node.attr
            return None
        if isinstance(node, ast.Call):
            func = self.expr(node.func, env)
            for arg in node.args:
                self.expr(arg, env)
            for kw in node.keywords:
                self.expr(kw.value, env)
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
                if name in GENERIC_NAMES and name not in self.shadowed:
                    self.found = True
                base = self.expr(node.func.value, env)
                if base == 'parser' and name in ('parse', 'raw_parse'):
                    self.found = True
                if base == 'tree-class' and name == 'fromstring':
                    return 'tree'
            if func and func.startswith('symbol:'):
                qualified = func[7:]
                parts = qualified.split('.')
                name = parts[-1]
                if name in GENERIC_NAMES and name not in self.shadowed:
                    self.found = True
                if qualified == 'stanza.Pipeline':
                    for kw in node.keywords:
                        if (kw.arg == 'processors' and isinstance(kw.value, ast.Constant)
                                and isinstance(kw.value.value, str)
                                and 'constituency' in kw.value.value.lower()):
                            return 'pipeline'
                if name in PARSERS and parts[0] in ('nltk', 'stanford', 'corenlp') and len(parts) > 1:
                    return 'parser'
                if parts[0] == 'nltk' and parts[-2:] == ['Tree', 'fromstring']:
                    return 'tree'
                if parts[0] == 'nltk' and name == 'Tree' and len(parts) > 1:
                    return 'tree-class'
            if func == 'pipeline':
                return 'document'
            return None
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            scope = env.copy()
            for gen in node.generators:
                val = self.expr(gen.iter, scope)
                self.bind(gen.target, 'sentence' if val in ('document', 'sentences') else None, scope)
                for cond in gen.ifs:
                    self.expr(cond, scope)
            for field in ('elt', 'key', 'value'):
                self.expr(getattr(node, field, None), scope)
            return None
        if isinstance(node, ast.NamedExpr):
            value = self.expr(node.value, env)
            self.bind(node.target, value, env)
            return value
        for child in ast.iter_child_nodes(node):
            self.expr(child, env)
        return None

    def statements(self, statements, env):
        for node in statements:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    env[alias.asname or alias.name.split('.')[0]] = 'module:' + (alias.name if alias.asname else alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module and not node.level:
                    for alias in node.names:
                        env[alias.asname or alias.name] = 'symbol:' + node.module + '.' + alias.name
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = self.expr(node.value, env)
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    self.bind(target, value, env)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scope = env.copy()
                if not isinstance(node, ast.ClassDef):
                    for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                        scope[arg.arg] = None
                    for arg in (node.args.vararg, node.args.kwarg):
                        if arg:
                            scope[arg.arg] = None
                self.statements(node.body, scope)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                value = self.expr(node.iter, env)
                self.bind(node.target, 'sentence' if value in ('document', 'sentences') else None, env)
                self.statements(node.body, env)
                self.statements(node.orelse, env)
            elif isinstance(node, (ast.If, ast.While, ast.With, ast.AsyncWith, ast.Try)):
                self.expr(getattr(node, 'test', None), env)
                for item in getattr(node, 'items', []):
                    self.expr(item.context_expr, env)
                    if item.optional_vars:
                        self.bind(item.optional_vars, None, env)
                for field in ('body', 'orelse', 'finalbody'):
                    self.statements(getattr(node, field, []), env.copy())
                for handler in getattr(node, 'handlers', []):
                    self.statements(handler.body, env.copy())
            else:
                self.expr(node, env)


def has_evidence(source):
    tree = ast.parse(source)
    visitor = Evidence(tree)
    visitor.statements(tree.body, {})
    return visitor.found


def audit(repo, ignore=(), verbose=None):
    """Return findings; IO and syntax errors propagate to the CLI."""
    root = Path(repo)
    if not root.is_dir():
        raise OSError('Repository is not a directory: ' + str(root))
    docs = []
    tests = []
    evidence = False
    if verbose:
        verbose('Scope: README.md, docs/**/*.md, tests/ fixtures, recursive *.py: ' + str(root))
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        name = relative.as_posix()
        # Detect environments even when they have a custom directory name.
        environment = any((parent / 'pyvenv.cfg').is_file() for parent in path.parents if parent != root and root in parent.parents)
        if any(part in EXCLUDED for part in relative.parts) or environment or any(name.startswith(p) for p in ignore):
            if verbose:
                verbose('Skipped: ' + name)
            continue
        if not path.is_file():
            continue
        is_doc = name == 'README.md' or (relative.parts[0] == 'docs' and path.suffix == '.md')
        is_test = 'tests' in relative.parts[:-1]
        if path.suffix == '.py' or is_doc or is_test:
            try:
                text = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                if is_test and path.suffix != '.py' and not is_doc:
                    # Binary fixtures have no documentation prose.
                    continue
                raise
            if verbose:
                verbose('Read: ' + name)
            if path.suffix == '.py':
                evidence = has_evidence(text) or evidence
            if is_doc or is_test:
                matches = phrase_matches(text)
                (docs if is_doc else tests).extend((name, kind, phrase) for kind, phrase in matches)
    findings = []
    for name, kind, phrase in docs + tests:
        test_only = (name, kind, phrase) not in docs
        if test_only and (kind != 'CON' or any(k == 'CON' for _, k, _ in docs)):
            continue
        if (kind == 'CON' and not evidence) or (kind == 'NOCON' and evidence):
            rule = ('documented-constituency-parsing-claim-false' if kind == 'CON'
                    else 'documented-no-constituency-parsing-claim-false')
            severity = 'MEDIUM' if test_only else 'HIGH'
            findings.append({'rule': rule, 'verdict': severity,
                             'message': 'Documentation assertion conflicts with static source evidence.',
                             'path': name, 'phrase': phrase})
    return findings
