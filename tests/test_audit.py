"""Independent literal oracles; tests never read the specification."""
POSITIVE = ['constituency parsing', 'constituency parser', 'constituency parse', 'parses constituency', 'parse constituency', 'constituency parses', 'parsing constituency', 'constituency-parsed', 'constituency tree', 'constituency trees', 'phrase structure tree', 'phrase structure trees', 'phrase-structure tree', 'phrase-structure trees', 'phrase structure parsing', 'phrase-structure parsing', 'phrase structure parser', 'phrase-structure parser', 'parses phrase structure', 'parse phrase structure', 'constituency grammar', 'constituency structure', 'constituency structures', 'syntactic constituency', 'syntactic phrase structure', 'assigns phrase labels', 'assign phrase labels', 'labels phrases', 'label phrases', 'identifies constituents', 'identify constituents', 'detects constituents', 'detect constituents', 'extracts constituents', 'extract constituents', 'supports constituency parsing', 'provides constituency parsing', 'performs constituency parsing', 'constituency parsing support', 'constituency parsing feature', 'constituency parsing capability', 'constituency parsing system', 'constituency parsing model', 'supports a constituency parser', 'provides a constituency parser', 'constituency annotation', 'constituency annotations', 'syntactic constituency annotation', 'constituency brackets', 'constituency bracketing']
NEGATIVE = ['does not parse constituency', 'do not parse constituency', "doesn't parse constituency", "don't parse constituency", 'no constituency parsing', 'without constituency parsing', 'never parses constituency', 'does not perform constituency parsing', 'never performs constituency parsing', 'constituency parsing not supported', 'constituency parsing unsupported', 'no constituency parsing support', 'lacks constituency parsing', 'missing constituency parsing', 'constituency parsing disabled', 'no constituency parser', 'without a constituency parser', 'does not provide constituency parsing', 'does not support constituency parsing', 'no constituency tree', 'without constituency parsing support', 'does not identify constituents', 'constituency parsing not available', 'no syntactic constituency parsing']
NAMES = ['parse_constituency', 'constituency_parse', 'get_constituency_tree', 'extract_constituency_tree', 'build_constituency_tree', 'constituency_parse_text', 'parse_constituency_tree', 'constituency_parse_sentence', 'analyze_constituency', 'analyse_constituency', 'detect_constituents', 'identify_constituents', 'label_constituents', 'assign_phrase_labels', 'constituency_annotate', 'annotate_constituency', 'constituency_parsing', 'phrase_structure_parse', 'parse_phrase_structure', 'extract_phrase_structure', 'get_phrase_structure', 'parse_with_constituency', 'build_phrase_structure_tree', 'extract_constituents']

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from constituency_parsing_claim_audit.audit import audit, has_evidence, phrase_matches
from constituency_parsing_claim_audit.phrases import CON, NOCON, GENERIC_NAMES


class AuditTests(unittest.TestCase):
    def fixture(self, files):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
        return root

    def cli(self, root, *args):
        return subprocess.run([sys.executable, '-m', 'constituency_parsing_claim_audit',
                               'audit', str(root), *args], capture_output=True, text=True)

    def assert_sources(self, sources, expected):
        for source in sources:
            with self.subTest(source=source):
                self.assertEqual(has_evidence(source), expected)

    def test_phrase_oracle(self):
        self.assertEqual(CON, POSITIVE)
        self.assertEqual(NOCON, NEGATIVE)
        self.assertEqual((len(CON), len(NOCON)), (50, 24))

    def test_generic_names_oracle(self):
        self.assertEqual(GENERIC_NAMES, NAMES)
        self.assertEqual(len(GENERIC_NAMES), 24)

    def test_all_positive_phrases(self):
        for phrase in POSITIVE:
            with self.subTest(phrase=phrase):
                self.assertEqual(phrase_matches(phrase.upper()), [('CON', phrase)])

    def test_all_denials(self):
        for phrase in NEGATIVE:
            with self.subTest(phrase=phrase):
                self.assertEqual(phrase_matches(phrase.upper()), [('NOCON', phrase)])

    def test_stanza_assigned(self):
        for prefix, make in [('import stanza', 'stanza.Pipeline'), ('import stanza as s', 's.Pipeline'),
                             ('from stanza import Pipeline', 'Pipeline'), ('from stanza import Pipeline as P', 'P')]:
            self.assert_sources([prefix + '\np=' + make + '(processors="tokenize,CONSTITUENCY")\nd=p(text)\nd.constituency',
                                 prefix + '\np=' + make + '(processors="constituency")\np(text).constituency'], True)

    def test_stanza_chained(self):
        self.assert_sources(['import stanza\nstanza.Pipeline(processors="constituency")(text).constituency',
                             'from stanza import Pipeline as P\nP(processors="constituency")(text).constituency'], True)

    def test_stanza_sentence_iteration(self):
        prefix = 'import stanza\np=stanza.Pipeline(processors="constituency")\nd=p(text)\n'
        self.assert_sources([prefix + 'for s in d.sentences:\n s.constituency',
                             prefix + 'for s in d:\n s.constituency',
                             prefix + '[s.constituency for s in d.sentences]',
                             'import stanza\n[s.constituency for s in stanza.Pipeline(processors="constituency")(text).sentences]',
                             'import stanza\nfor s in stanza.Pipeline(processors="constituency")(text):\n s.constituency'], True)

    def test_parser_assigned(self):
        names = ['ChartParser', 'RecursiveDescentParser', 'ShiftReduceParser', 'EarleyChartParser',
                 'ViterbiParser', 'ProbabilisticChartParser', 'StanfordParser', 'CoreNLPParser']
        for name in names:
            for module in ['nltk', 'nltk.parse', 'stanford', 'corenlp', 'nltk.parse.stanford', 'nltk.parse.corenlp']:
                for method in ['parse', 'raw_parse']:
                    self.assert_sources(['from ' + module + ' import ' + name + ' as P\np=P(grammar)\np.' + method + '(text)',
                                         'import ' + module + ' as m\np=m.' + name + '(grammar)\np.' + method + '(text)'], True)

    def test_parser_chained(self):
        self.assert_sources(['import nltk as nk\nnk.ChartParser(grammar).parse(text)',
                             'from nltk.parse.stanford import StanfordParser as P\nP().raw_parse(text)',
                             'import nltk.parse\nnltk.parse.ChartParser(grammar).parse(text)'], True)

    def test_tree_assigned(self):
        for prefix, tree in [('import nltk', 'nltk.Tree'), ('import nltk as nk', 'nk.Tree'),
                             ('from nltk import Tree', 'Tree'), ('from nltk.tree import Tree as T', 'T'), ('from nltk.tree.tree import Tree as T', 'T')]:
            for attr in ['label', 'leaves', 'subtrees', 'height', 'pretty_print', 'pformat']:
                self.assert_sources([prefix + '\nt=' + tree + '.fromstring("(S x)")\nt.' + attr,
                                     prefix + '\nt=' + tree + '.fromstring("(S x)")\nt.' + attr + '()'], True)

    def test_tree_chained(self):
        self.assert_sources(['import nltk\nnltk.Tree.fromstring("(S x)").leaves()',
                             'from nltk import Tree as T\nT.fromstring("(S x)").label()',
                             'from nltk import Tree as T\nT().fromstring("(S x)").height()'], True)

    def test_generic_attribute_any_base(self):
        for name in NAMES:
            self.assert_sources(['x.' + name + '(text)', 'make().nested.' + name + '(text)'], True)

    def test_generic_attribute_on_locally_created_base(self):
        self.assert_sources(['def make(): pass\nmake().parse_constituency(text)',
                             'class Client: pass\nClient().parse_constituency(text)'], True)

    def test_generic_imported_bare(self):
        for name in NAMES:
            self.assert_sources(['from arbitrary import ' + name + '\n' + name + '(text)',
                                 'from arbitrary import ' + name + ' as f\nf(text)'], True)

    def test_module_import_is_not_from_imported_generic(self):
        self.assert_sources(['import parse_constituency\nparse_constituency(text)',
                             'import tool.parse_constituency as f\nf(text)'], False)

    def test_bare_imports_rejected(self):
        self.assert_sources(['import stanza', 'import nltk', 'from nltk import Tree',
                             'from nltk.parse import ChartParser', 'from tool import parse_constituency'], False)

    def test_dependency_rejected(self):
        self.assert_sources(['x.dep_\nx.head', 'from nltk.parse.stanford import StanfordDependencyParser as P\nP().parse(text)',
                             'import stanza\np=stanza.Pipeline(processors="depparse")\np(text).dependencies'], False)

    def test_pos_and_ner_rejected(self):
        self.assert_sources(['nltk.pos_tag(text)', 'x.pos_\nx.tag_', 'nltk.ne_chunk(text)', 'x.ents',
                             'pipeline("ner")(text)', 'NERTagger().predict(text)'], False)

    def test_other_tasks_rejected(self):
        self.assert_sources(['x.extract_keywords(text)', 'KeyphraseModel()(text)', 'pipeline("sentiment-analysis")(text)',
                             'x.sentiment(text)', 'pipeline("text-classification")(text)', 'Classifier()(text)',
                             'tokenizer(text)', 'lemmatizer(text)', 'x.lemma_'], False)

    def test_tree_incomplete_rejected(self):
        self.assert_sources(['from nltk import Tree\nTree("S", []).leaves()', '"(S (NP x))"',
                             'from nltk import Tree\nTree.fromstring("(S x)")',
                             'from other import Tree\nTree.fromstring("(S x)").leaves()',
                             'from nltk import Tree\nTree.fromstring("(S x)")\nx.leaves()'], False)

    def test_parser_incomplete_and_wrong_origin_rejected(self):
        self.assert_sources(['from nltk import ChartParser\nChartParser(grammar)',
                             'from other import ChartParser\nChartParser(grammar).parse(text)',
                             'ChartParser(grammar).parse(text)', 'x.parse(text)',
                             'from nltk import ChartParser\np=ChartParser(grammar)\nx.raw_parse(text)'], False)

    def test_stanza_incomplete_rejected(self):
        self.assert_sources(['import stanza\nstanza.Pipeline(processors="constituency")',
                             'import stanza\np=stanza.Pipeline(processors="tokenize,pos")\np(text).constituency',
                             'import stanza\np=stanza.Pipeline(processors="constituency")\nx.constituency',
                             'import stanza\np=stanza.Pipeline(processors="constituency")\np.constituency',
                             'import stanza\np=stanza.Pipeline(processors=processors)\np(text).constituency',
                             'Pipeline(processors="constituency")(text).constituency',
                             'from other import Pipeline\nPipeline(processors="constituency")(text).constituency'], False)

    def test_strings_comments_references_subprocess_rejected(self):
        self.assert_sources(['"constituency parsing"\n# x.parse_constituency(text)',
                             '"""x.parse_constituency(text)"""', 'parse_constituency',
                             'parse_constituency(text)', 'from x import parse_constituency\nparse_constituency',
                             'subprocess.run(["constituency-parser"])'], False)

    def test_shadowed_generic(self):
        self.assert_sources(['from x import parse_constituency\ndef parse_constituency(x): pass\nparse_constituency(text)',
                             'from x import parse_constituency as f\nclass f: pass\nf(text)',
                             'from x import parse_constituency\nparse_constituency(text)\ndef parse_constituency(x): pass'], False)

    def test_shadowed_pairs(self):
        self.assert_sources(['from nltk import ChartParser\nclass ChartParser: pass\nChartParser().parse(text)',
                             'from nltk import ChartParser as P\ndef P(): pass\nP().raw_parse(text)',
                             'import nltk as nk\nclass nk: pass\nnk.Tree.fromstring("(S x)").leaves()',
                             'from nltk import Tree\nclass Tree: pass\nTree.fromstring("(S x)").label()',
                             'from stanza import Pipeline\nclass Pipeline: pass\nPipeline(processors="constituency")(text).constituency'], False)

    def test_reassignment_and_function_parameter(self):
        self.assert_sources(['from nltk import ChartParser\np=ChartParser()\np=other\np.parse(text)',
                             'import stanza\ndef f(stanza):\n stanza.Pipeline(processors="constituency")(text).constituency'], False)

    def test_single_module_scope(self):
        for first, second in [('import stanza\np=stanza.Pipeline(processors="constituency")', 'p(text).constituency'),
                              ('from nltk import ChartParser\np=ChartParser()', 'p.parse(text)'),
                              ('from nltk import Tree\nt=Tree.fromstring("(S x)")', 't.leaves()')]:
            root = self.fixture({'README.md': POSITIVE[0], 'a.py': first, 'b.py': second})
            self.assertEqual(audit(root)[0]['verdict'], 'HIGH')

    def test_span_removal(self):
        self.assertEqual(phrase_matches('does not parse constituency'), [('NOCON', 'does not parse constituency')])
        root = self.fixture({'README.md': 'does not parse constituency'})
        self.assertEqual(audit(root), [])

    def test_longest_first(self):
        self.assertEqual(phrase_matches('constituency parsing system'), [('CON', 'constituency parsing system')])
        self.assertEqual(phrase_matches('constituency trees; constituency tree'),
                         [('CON', 'constituency trees'), ('CON', 'constituency tree')])

    def test_fenced_code_and_inline(self):
        self.assertEqual(phrase_matches('```text\nconstituency parsing\n```\n'), [])
        self.assertEqual(phrase_matches('`CONSTITUENCY PARSING`'), [('CON', 'constituency parsing')])
        self.assertEqual(phrase_matches('constituency-parsing'), [])
        self.assertEqual(phrase_matches('constituency  parsing'), [])

    def test_documentation_scope(self):
        root = self.fixture({'README.md': 'Neutral', 'docs/deep/page.md': POSITIVE[0],
                             'other.md': POSITIVE[0], 'docs/not.txt': POSITIVE[0]})
        self.assertEqual([f['path'] for f in audit(root)], ['docs/deep/page.md'])

    def test_cli_good_pass(self):
        root = self.fixture({'README.md': POSITIVE[0], 'main.py': 'x.parse_constituency(text)'})
        result = self.cli(root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'PASS')

    def test_cli_bad_exact_json(self):
        root = self.fixture({'README.md': POSITIVE[0]})
        result = self.cli(root, '--json')
        self.assertEqual(result.returncode, 2, result.stderr)
        findings = json.loads(result.stdout)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]['rule'], 'documented-constituency-parsing-claim-false')
        self.assertEqual(findings[0]['verdict'], 'HIGH')
        self.assertTrue(findings[0]['message'])

    def test_cli_test_only(self):
        root = self.fixture({'tests/fixtures/sample.md': POSITIVE[0]})
        result = self.cli(root, '--json')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)[0]['verdict'], 'MEDIUM')
        self.assertEqual(self.cli(root, '--fail-on', 'medium').returncode, 2)

    def test_test_fixture_extension_is_unrestricted(self):
        root = self.fixture({'tests/sample.fixture': POSITIVE[0]})
        self.assertEqual(audit(root)[0]['verdict'], 'MEDIUM')
        self.assertEqual(self.cli(root, '--fail-on', 'medium').returncode, 2)

    def test_cli_denial(self):
        root = self.fixture({'README.md': NEGATIVE[0], 'main.py': 'x.parse_constituency(text)'})
        result = self.cli(root, '--json')
        self.assertEqual(result.returncode, 2, result.stderr)
        finding = json.loads(result.stdout)[0]
        self.assertEqual(finding['rule'], 'documented-no-constituency-parsing-claim-false')
        self.assertEqual(finding['verdict'], 'HIGH')

    def test_cli_no_claim(self):
        root = self.fixture({'README.md': 'Nested phrase grouping', 'main.py': 'x.parse_constituency(text)'})
        self.assertEqual(self.cli(root).stdout.strip(), 'PASS')
        self.assertEqual(self.cli(root, '--json').stdout.strip(), '[]')

    def test_cli_version_and_help(self):
        for arg, expected in [('--version', '0.1.0'), ('--help', 'usage:')]:
            result = subprocess.run([sys.executable, '-m', 'constituency_parsing_claim_audit', arg], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            self.assertIn(expected, result.stdout)
        result = subprocess.run([sys.executable, '-m', 'constituency_parsing_claim_audit', '--version'], capture_output=True, text=True)
        self.assertEqual(result.stdout, '0.1.0\n')

    def test_cli_ignore_and_verbose(self):
        root = self.fixture({'README.md': POSITIVE[0], 'docs/a.md': POSITIVE[0]})
        result = self.cli(root, '--ignore', 'README', '--ignore', 'docs/', '--verbose')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'PASS')
        self.assertIn('Scope:', result.stderr)
        self.assertIn('Skipped: README.md', result.stderr)

    def test_ignored_sources(self):
        root = self.fixture({'README.md': POSITIVE[0], 'vendor/a.py': 'x.parse_constituency(text)'})
        self.assertEqual(audit(root), [])
        self.assertEqual(audit(root, ['vendor/'])[0]['verdict'], 'HIGH')

    def test_excluded_directories(self):
        root = self.fixture({'README.md': POSITIVE[0], '.git/a.py': 'x.parse_constituency(text)',
                             '__pycache__/a.py': 'x.parse_constituency(text)', '.venv/a.py': 'x.parse_constituency(text)',
                             'custom/pyvenv.cfg': '', 'custom/a.py': 'x.parse_constituency(text)'})
        self.assertEqual(audit(root)[0]['verdict'], 'HIGH')

    def test_cli_errors(self):
        root = self.fixture({'main.py': 'not valid syntax !'})
        self.assertEqual(self.cli(root).returncode, 1)
        self.assertEqual(self.cli(root / 'missing').returncode, 1)
        self.assertEqual(self.cli(root, '--fail-on', 'low').returncode, 1)
        self.assertEqual(self.cli(root, '--unknown').returncode, 1)
        (root / 'main.py').write_bytes(b'\xff')
        self.assertEqual(self.cli(root).returncode, 1)

    def test_readme_self_audit(self):
        readme = Path(__file__).resolve().parents[1] / 'README.md'
        self.assertTrue(readme.is_file())
        self.assertEqual(phrase_matches(readme.read_text(encoding='utf-8')), [])
        root = self.fixture({'README.md': readme.read_text(encoding='utf-8')})
        result = self.cli(root, '--json')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])

    def test_default_repo(self):
        root = self.fixture({'README.md': 'Neutral'})
        package_root = str(Path(__file__).resolve().parents[1])
        import os
        env = os.environ.copy()
        env['PYTHONPATH'] = package_root
        result = subprocess.run([sys.executable, '-m', 'constituency_parsing_claim_audit', 'audit'],
                                cwd=root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'PASS')


if __name__ == '__main__':
    unittest.main()
