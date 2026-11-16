# Observed-return capture

The first capture's numerical results remain in `initial_observer/`. Its hook
looked only at a direct caller; tune_mstop's warning wrapper intervened, so it
recorded no tuning stages. Those rows do not prove the declared/actual grid
contract and were not used to claim an audit pass. The observer now walks the
wrapper stack to _tune_mean and refuses a successful high-mode record without
actual tuner returns. Statistical calls, data, seeds and frozen analytical
protocol are unchanged. All96API/replay jobs are repeated with the corrected
observer; their predictions/MSEs must match the initial capture exactly.
