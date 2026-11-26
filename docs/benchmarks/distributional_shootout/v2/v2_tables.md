## lss_trees decision

- synth: ratio 1.004 (0.995–1.018), 1 better, 14 tied, 1 worse over 16 tasks
- real: ratio 0.998 (0.958–1.036), 1 better, 31 tied, 3 worse over 35 tasks
- median time ratio 1.22; adopt `lss_trees=True`: False

## Held-out mean accuracy of mb.fit

- runs 285; failed before 87, after 2
- datasets fixed: [29, 1063, 1590, 42165, 44962, 44974, 45012, 100009, 100015, 100032, 100047, 100060, 100074, 100105, 100117, 100161, 100172, 100183, 100222, 100225, 100277, 100332, 100350, 100357, 100365, 100368, 100492, 100529, 100597, 100603, 100848, 100967]; datasets newly failing: [100572]
- loss ratio after/before 0.8063 (0.6474–0.9766), 34 better, 22 tied, 13 worse over 69 datasets
- median seconds before 9.34, after 16.29
- mean accuracy holds: True
