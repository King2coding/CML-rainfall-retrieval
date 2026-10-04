# Commercial microwave-link rainfall retrieval

Research code for detecting, estimating, mapping, and evaluating rainfall from
commercial microwave-link (CML) attenuation, with supporting Meteosat Second
Generation (MSG), gauge, ERA5, and related observations. The repository
preserves successive research and operational workflow versions used for work
in data-sparse regions of Africa, including Ghana.

## Repository status

This is a code-only research archive. Source files are retained as analysis
records rather than refactored into a software package. Credentials, raw signal
data, research datasets, generated outputs, caches, and logs are excluded.

Several scripts contain rain-server paths or expect environment variables and
external services. Review configuration sections carefully before execution.

## Related publications

- Kumah, K. K., B. H. P. Maathuis, J. C. B. Hoedjes, and Z. Su, 2022:
  Near real-time estimation of high spatiotemporal resolution rainfall from
  cloud top properties of the MSG satellite and commercial microwave link
  rainfall intensities. *Atmospheric Research*, **279**, 106357.
  [https://doi.org/10.1016/j.atmosres.2022.106357](https://doi.org/10.1016/j.atmosres.2022.106357)
- Kumah, K. K., J. C. B. Hoedjes, N. David, B. H. P. Maathuis, H. O. Gao, and
  Z. Su, 2021: The MSG Technique: Improving Commercial Microwave Link Rainfall
  Intensity by Using Rain Area Detection from Meteosat Second Generation.
  *Remote Sensing*, **13**(16), 3274.
  [https://doi.org/10.3390/rs13163274](https://doi.org/10.3390/rs13163274)
- Kingsley, K. K., B. H. P. Maathuis, J. C. B. Hoedjes, D. T. Rwasoka,
  B. V. Retsios, and Z. Su, 2021: Rain Area Detection in South-Western Kenya
  by Using Multispectral Satellite Data from Meteosat Second Generation.
  *Sensors*, **21**(10), 3547.
  [https://doi.org/10.3390/s21103547](https://doi.org/10.3390/s21103547)
- Kumah, K. K., J. C. B. Hoedjes, N. David, B. H. P. Maathuis, H. O. Gao, and
  Z. Su, 2020: Combining MWL and MSG SEVIRI Satellite Signals for Rainfall
  Detection and Estimation. *Atmosphere*, **11**(9), 884.
  [https://doi.org/10.3390/atmos11090884](https://doi.org/10.3390/atmos11090884)

## Code organization

The repository contains several generations of the research workflow:

- CML signal-level quality control and metadata coupling;
- wet/dry classification and baseline estimation;
- attenuation and wet-antenna correction;
- rain-rate retrieval;
- 15-minute spatial interpolation and gridding;
- CML, gauge, and satellite evaluation;
- MSG acquisition and cloud-mask processing; and
- rainfall-map production and service workflows.

Files carrying dates or labels such as `prime`, `TAHMO`, `one_stage`, or
`twostage` are retained to preserve the analysis sequence. They should not be
assumed to be interchangeable.

## Credentials and configuration

Never commit operational credentials. Create a local `.env` file from
`.env.example` and populate it outside version control. Some archived scripts
read configuration from environment variables, while service-oriented scripts
may require additional deployment configuration.

## Data requirements

No CML observations, gauge data, satellite data, generated rainfall products,
or evaluation outputs are included. See [`DATA.md`](DATA.md) for the data
categories expected by the workflows.

## Software requirements

The scripts use a broad scientific and geospatial Python environment. Common
imports include `numpy`, `pandas`, `xarray`, `scipy`, `scikit-learn`,
`matplotlib`, `seaborn`, `geopandas`, `rasterio`, `pyproj`, `cartopy`, `dask`,
and CML-specific processing libraries. Some acquisition and service scripts
also require network-client and cloud-storage packages.

Exact dependency versions vary across workflow generations and were not
consistently recorded. Reproduction should begin from the relevant script and
its import statements rather than assuming one environment runs every version.

## Citation and reuse

Please cite the publication most relevant to the workflow used. Citation
metadata is provided in [`CITATION.cff`](CITATION.cff). No software reuse
license has yet been assigned.

## Contact

Kwabena Kingsley Kumah — [GitHub profile](https://github.com/King2coding)
