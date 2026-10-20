# Historical FaceChannel service

This directory retains the original perception/continual-learning implementation. Keep its TensorFlow/Keras environment separate from the root verification environment and the robot SDK. Historical TF 2.4 tested configurations use Python 3.6-3.8; installing all repository requirements into one interpreter is unsupported.

The original model assets and matching lab adapters are required. The root CSV verifier works without them. API GET/POST routes and prediction keys remain available to loopback clients; browser-origin/Host protections now run before experiment side effects. Optional token and remote-host configuration, pinned model acquisition, state trust and compatibility limits are described in [maintenance guidance](../../../docs/maintenance.md).

The retained service exposes `/join`, `/start`, `/stop`, `/next` and `/exit`, with its original URL and JSON forms. It is a historical service interface; its two-session helper is not an independent implementation of the manuscript's complete demo/counterbalanced study protocol. Do not substitute it for the original lab orchestration without validation.

Missing FaceChannel model files require an explicitly trusted asset installation. CAAE checkpoint entries are present in the Git tree but may be omitted from source-focused sparse checkouts. Model mathematical code and the historical dependency declarations remain unchanged by maintenance.
