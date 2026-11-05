# Documentation claim auditor

Compare documentation assertions about nested phrase grouping against Python source patterns. The checker runs offline using only the standard library. It reads source without executing it, makes no network requests, sends no telemetry, and writes no files.

## Usage

Requires Python 3.9 or newer.

```sh
python3 -m pip install constituency-parsing-claim-audit
constituency-parsing-claim-audit audit . --json
python3 -m constituency_parsing_claim_audit audit . --verbose --ignore vendor/ --ignore build/ --fail-on medium
constituency-parsing-claim-audit --version
```

The repository defaults to the current directory. Repeat the ignore option to skip relative path prefixes. Scope includes the root readme, Markdown recursively under the documentation directory, fixtures under test directories, and recursive Python sources. Version control, bytecode caches and virtual environments are excluded.

Fenced blocks are excluded from documentation prose. Inline code counts as prose. Matching uses exact case-insensitive substrings, longest first, and removes denial spans before checking positive assertions. Hyphens retain their spelling.

HIGH findings fail by default. Positive assertions found only in test fixtures receive MEDIUM severity and fail at the medium threshold. Exit codes are 0 for passing the threshold, 2 for findings meeting it, and 1 for usage or input errors. JSON output is an array of findings, each with rule, verdict, message, path and phrase fields. Verbose output describes scope and file decisions on standard error.

```text
documented-constituency-parsing-claim-false
documented-no-constituency-parsing-claim-false
```

## Source checks

The checked feature is bracketed sentence grouping. Imported aliases resolve within one module; local function and class definitions shadow imported names. Paired patterns must connect construction to the appropriate use of its produced value in that module.

```text
stanza.Pipeline(processors="tokenize,constituency") + call + .constituency
NLTK / Stanford listed parser construction + .parse(...) / .raw_parse(...)
NLTK Tree.fromstring(...) + .label / .leaves / .subtrees / .height / .pretty_print / .pformat
24 exact generic call names
```

Imports alone, incomplete pairs, strings and unrelated tasks do not establish evidence. Assigned values, chained calls and sentence iteration are recognized for the relevant patterns.

## Differentiation

These neighboring tools cover distinct outputs:

```text
dependency-parsing-claim-audit: head-dependent relations
part-of-speech-tagging-claim-audit: per-token word-class labels
named-entity-recognition-claim-audit: typed entity spans
```

Head-dependent relations link individual words. Per-token word-class labels describe individual tokens. Typed entity spans locate named referents. This checker instead examines assertions about nested phrase grouping.

## Limitations

Static pattern matching is not semantic judgement. Exact phrases miss paraphrases. Generic call names can produce false positives. Dynamic dispatch, wrappers and relationships across files can be missed. Recognized patterns do not prove runtime correctness or output quality. Unreadable text and malformed Python sources cause input errors.

## Development

```sh
python3 -m unittest discover
python3 -m build
```

The source distribution includes tests. Licensed under MIT.
