# Data layout and provenance

## Segmentation experiment

`segmentation_split.json` contains the fixed training, validation, and event-disjoint test filenames used by `code/tpsm_train.py`. The data root is expected to contain aligned 256×256 GeoTIFF windows arranged as pre-fire reflectance, post-fire reflectance, NBR-family composite, and four-class mask directories. Each Sentinel-2 reflectance stream has nine bands; the index stream has three channels.

The repository does not duplicate the source Sentinel-2 rasters. They can be obtained from the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/), subject to its terms and access procedures.

## CBI field-reference experiment

`cbi_split.json` contains the portable metadata for the fixed 75/46/28 train/validation/test split. The 28 test locations come from Fuller and Legion Lake. Complete 256×256 input windows were kept non-overlapping across the three splits, and predictions were aggregated over the nominal 30 m field-plot support.

The source field observations are available from the USGS data release:

- Composite Burn Index data for the conterminous United States: <https://doi.org/10.5066/P91BH1BZ>

The public manifest removes machine-specific local paths while preserving location identifiers, fire names, CBI values, coordinates, field dates, and split assignments.

