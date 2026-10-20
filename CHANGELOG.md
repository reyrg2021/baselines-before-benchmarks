# Changelog

## v1.0.0

First public release, accompanying the article.

Differences with respect to the scripts as they were executed for the article (none of
them changes a computation, a parameter, or a reported number):

- **Language and comments.** Comments, docstrings, log and print messages, and the text
  of the Markdown reports were translated from Spanish into English, and comments were
  condensed. Spanish identifiers were renamed consistently in all files (for example
  `semilla` to `seed_value`, `cargar_modelos` to `load_models`).
- **Output keys and labels.** Spanish keys and display labels in output files were
  translated together with every script that reads them: the JSON keys and CSV column of
  script 40, the JSON keys and variant names of script 42, the column `familia` of
  script 24, the key `barrido_results` shared by scripts 15 and 24, the event-table
  column of `events/18_event_extraction.py`, and the quality-control list written by
  script 02 and parsed by script 03 (now `problematic_files.txt`, with the lines
  `File:`, `Severe clipping:`, and `Low SNR:`). The result files in `results/` and the
  logs in `logs/` were translated accordingly, with every number unchanged.
- **Paths.** Absolute paths of the original workstation were replaced by paths relative
  to the repository root, the ICBHI folder now points to `data/raw/`, scripts 02 and 03
  locate the repository root from their own location, the default prediction folder of
  `rodri2_roc.py` and `rodri3_confusion.py` is `results/metrics/full_reeval/arrays/`,
  and script 38 also reads the training logs from `logs/`.

Verification: every script was compared with its executed version through its abstract
syntax tree (only comments, docstrings, message text, and the renamings above differ);
scripts 29, 35, 36, and 38 regenerate their outputs with identical values, and the
figure scripts regenerate Figures 1 to 5 with identical pixels.
