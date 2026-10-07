# GeoTwin MVP: local WGS84 labels and camera-coverage annotations

The current geometry is produced in a local world frame aligned to the DJI telemetry by the VGGT pose/depth adapter. `real_dataset_builder` defines its horizontal frame as East-North-Up metres around `dataset.origin_lat` / `dataset.origin_lon`; `build_pose_depth_source` aligns VGGT camera centers to those telemetry positions. The Poisson mesh therefore can be annotated in WGS84 by converting mesh X/Y (east/north metres) back from that same origin. This is approximate georeferencing, not survey control: the current run reported 1.17 m median and 4.42 m maximum camera-to-telemetry residual.

The interactive viewer is `web/geotwin_mvp.html`. It loads a local GLB or OBJ, and optionally reads a matching DJI SRT/CSV selected alongside the model. It lets the presenter click visible surfaces to place building/road/tree landmarks and displays a coordinate callout. It also draws amber, separately stored patch overlays in camera-unseen gaps. Patch overlays are explicitly marked inferred and are not merged into or counted as observed mesh geometry. This keeps the MVP visually legible without making an unsupported surface appear measured.

## Run it (no coordinate entry required)

1. Use the locally downloaded model made from **density trim 0.01 + minimum component size 500**. GLB is fastest; OBJ is also accepted.

2. In the project folder, double-click `Open_GeoTwin_MVP.bat`. It starts a small server on this computer and opens the viewer page in the browser. Leave its command window open while using the viewer. The browser needs an internet connection to load the Three.js display libraries.

3. Choose the model. If you have its matching SRT or CSV, choose that file in the same file dialog too (Ctrl-click both files), or drag both onto the viewer. The viewer reads the first valid latitude/longitude pair and applies it; there is no coordinate typing. If no log is supplied, it uses the bundled DJI0004 SRT origin. It checks the loaded model bounds against the clean500 reference and warns if they differ. The exact expected GLB is `exp2_depth12_trim001_clean500/vggt_poisson_depth12.glb` from Colab.

4. Add presentation landmarks: choose a feature name/type, click **Click model to place landmark**, then click a visible roof/footprint or road point. Repeat for selected building centroids/corners and key road locations. Each pin shows approximate latitude/longitude.

5. Add gap callouts/fills. Click **Set fallback Z from a road click** and click a visible road surface once. Then click **Click a gap to add inferred patch** and click the gap. A hit on an existing surface uses that height; a click over empty space uses the selected road height. Amber patches are separate, planar display placeholders marked `inferred_not_camera_observed`. They are not recovered building backs or terrain. For gaps at a different elevation, change the fallback Z in the Advanced area first.

6. Export both files:

   - `geotwin_annotations.geojson`: WGS84 landmark points and inferred patch polygons, suitable for GIS inspection. Vertical values are stored as relative `model_z_m` properties, not absolute elevation.
   - `geotwin_session.json`: editable viewer annotations in model ENU coordinates. Keep this with the exact GLB and origin metadata; load it next time to restore the annotations.

For a submission/demo, keep the reviewed GLB, these two annotation files, and screenshots from the viewer together. Show the model from an oblique view and a top view, with both legend entries visible. If opening a different flight/run, update `web/geotwin_origin.json` and its expected mesh bounds from that run before presenting it.

## Interpretation and limits

- The origin-to-WGS84 conversion is a local tangent-plane approximation over this small corridor. Coordinates are rounded to five decimal places for display, but that rounding is not the positional accuracy; the alignment residual is metre-scale.
- This viewer is preconfigured for the submitted DJI0004 run. It automatically loads the origin from `web/geotwin_origin.json`, derived from the matching SRT's first GPS sample (the same origin rule used by `real_dataset_builder`). An optional SRT/CSV selected with a new model overrides that default using its first valid GPS pair. No presenter needs to know or type latitude/longitude values.
- The viewer does not automatically identify all buildings. For the MVP, manually highlight a curated set of important building centroids/corners. An automatic building detector/segmenter is a later feature and would need manual checks against this video.
- A green/colored surface comes from the reconstructed mesh. An amber patch is a presentation overlay for a hole or unseen region. It is explicitly uncertain and should remain a separate layer in the report and GeoJSON.
- The amber overlay is intentionally not a watertight mesh repair. The current single oblique flight cannot establish geometry behind occluders or outside camera coverage. A second overlapping pass would be required to replace those inferred patches with observed surfaces.
- The GLB does not by itself retain the local ENU-to-WGS84 origin. Distribute it with the session/GeoJSON and show the origin, coordinate quality, and `inferred` legend in the MVP viewer.

## Important: choose the matching GLB

The earlier local `Downloads/vggt_poisson_depth12 (1).glb` does **not** match clean500: it reaches X=516.93 while clean500 ends at X=447.46. The newer uploaded `Downloads/vggt_poisson_depth12 (3).glb` does match the clean500 OBJ bounds. Prefer the newer uploaded file. For the viewer, download the GLB directly from the matching Colab output folder:

```python
from google.colab import files
files.download('/content/exp2_depth12_trim001_clean500/vggt_poisson_depth12.glb')
```

If that exact file is no longer in the Colab runtime, regenerate only that already-selected candidate from the saved cloud using the recorded `density_trim_quantile=0.01` and `min_component_vertices=500` settings. Do not use a similarly named GLB from a different output folder.
