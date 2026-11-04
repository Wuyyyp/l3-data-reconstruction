# DVP depth mask policy v2

OpenMVS masks use 255=ignore, 0=keep. DVP blocks use the opposite convention: 255=keep, 0=ignore. New blocks are lossless PNG; old JPEG scenes remain readable.

`build_effective_masks.py` unions the supplied sky/dynamic masks, optional recomputed sky masks, an explicit camera-10 ego polygon, and black padding components attached to the source image boundary. It expands ignored boundaries by 2 pixels at width 960. Interior dark objects are not discarded by color alone. Downsampling invalidates a target pixel if any contributing source pixel is ignored.

For the examined pilotGtParser rig, `camera10_pilotGtParser_rig.json` masks the lower 36 percent of the unpadded camera-10 image, with a small boundary margin. It covers the dashboard/wipers/hood; it is NOT a calibration for arbitrary cameras. Review the overlays before reuse on another rig.

The batch entry point requires `DVP_CAMERA10_RIG_PROFILE` or `<sample-root>/06_masks/camera10_rig_profile.json`. It accepts additional sky masks through `DVP_SKY_MASKS` or `<sample-root>/06_masks/safe_sky`. Effective masks are saved under `<result-root>/06_masks/effective_masks`. Their content fingerprint is included in the pipeline request; changed masks require a new result directory. Per-view preparation also fingerprints its mask.

MoGe priors are masked during preparation. Every DVP outer pass writes zero depth/normal and clears selected views inside the resized ignore mask. This applies before subsequent passes read the saved depths. OpenMVS DMAP export and COLMAP depth export independently enforce the same PNG blocks. OpenMVS fusion receives the effective source-resolution ignore masks.

`filter_saved_dvp_depths.py --run OLD/09_dvp --masks EFFECTIVE_MASKS --output NEW/09_dvp` creates an isolated filtered copy of saved depths, normals, priors, image copies and blocks. It does not rerun MVS or regenerate a point cloud. Re-estimation is needed to remove any effect that formerly invalid regions had on adjacent valid pixels.

Sky segmentation keeps confident sky components reaching the upper third, rather than requiring contact with the outer image border: camera-10 windshield occlusions can disconnect real sky from that border. Existing sky/building confidence checks and conservative erosion are retained.
