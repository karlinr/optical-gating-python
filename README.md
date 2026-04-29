# optical-gating-python

Implementing prospective optical gating in Python. Primarily for testing the new MLE approach to prospective optical gating phase estimation and integration of the Kalman filter for improved prediction.

# Structure
```text
src/
├── app/
│   ├── config.py          # Central configuration (cameras, timing box, gating params)
│   ├── main.py            # Entry point — orchestrates the experiment loop
|   └── state.py           # Thread safe state manager for eventual future UI integration
├── interfaces/
│   ├── camera.py          # Ximea camera driver
│   ├── timing_box.py      # Serial interface to the timing box
│   └── system.py          # High-level system controller
├── logic/
│   ├── phase_estimator.py # Phase estimation: SAD and MLE estimators
│   ├── phase_predictor.py # (Placeholder) Future prediction logic
│   ├── drift_corrector.py # (Placeholder) Drift correction
│   └── utils.py           # SAD, chi-squared, V-fitting
└── hardware_emulators/
    ├── camera.py          # Software camera emulator (generates synthetic heart frames)
    └── timing_box.py      # Software timing box emulator (runs in a separate process)
```

# WIP
## Implemented
- Ximea camera setup and synchronisation to timing box
- Timing box hardware integration
- Camera and timing box emulators for testing
- Phase estimation using SAD method
- Thread safe state manager for eventual future UI integration
## Partially implemented
- Phase estimation using MLE method
## Not implemented
- Phase prediction
- Fluorescence camera triggering
- Drift correction
