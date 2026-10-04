import importlib
import logging
import mimetypes
from datetime import timezone as dt_timezone
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from django.conf import settings
from apps.api.models import CMLFile, RainfallMapArtifact

logger = logging.getLogger(__name__)


@dataclass
class RainfallRunResult:
    status: str
    message: str
    generated_files: List[str]
    uploaded_files: List[str]
    latest_timestamp: Optional[str] = None


class RainfallMapService:
    def __init__(self) -> None:
        self.config = settings.RAINFALL_MAP_CONFIG
        self.raw_archive_dir = Path(self.config["raw_archive_dir"])
        self.output_dir = Path(self.config["output_dir"])
        self.metadata_path = Path(settings.BASE_DIR) / "tahmocml" / "data" / "matched_metadata_kkk_20250527.csv"

    def run(self) -> Dict[str, Any]:
        if not self.config.get("enabled"):
            return self._result("skipped", "Rainfall generation is disabled.")

        if not self.metadata_path.exists():
            return self._result(
                "skipped",
                f"Rainfall metadata file not found: {self.metadata_path}.",
            )

        try:
            pipeline = self._load_pipeline_module()
            latest_dt, recent_files = self._select_recent_files(pipeline)
            if not recent_files:
                return self._result("skipped", "No archived raw CML files were found for rainfall generation.")

            matched_metadata = pd.read_csv(self.metadata_path)
            coupled_frames = []
            for file_path in recent_files:
                frame = pipeline.cml2metadata_coupling_framework_fast(str(file_path), matched_metadata)
                if frame is not None and not frame.empty:
                    coupled_frames.append(frame)

            if not coupled_frames:
                return self._result("skipped", "No archived files could be coupled to rainfall metadata.")

            df_raw = pd.concat(coupled_frames, ignore_index=True)
            cfg = pipeline.R0AutoConfig(semantics="auto", regularize_grid=True)
            df_clean, _ = pipeline.clean_minmax_auto(df_raw, cfg)
            ts_15 = pipeline.build_15min_timeseries(df_clean)
            df_a = pipeline.rainlink_strict_Aobs(
                ts_15,
                two_pass=True,
                use_drycount_guard=False,
            )
            df_rate = pipeline.rainlink_strict_R(df_a, R_min=0.0)
            df_s5, meta_xy_grid = pipeline.prepare_inputs_for_gridding(df_rate, ts_15)

            # rain_da, _ = pipeline.grid_rain_15min_rainlink_ok(
            #     df_s5,
            #     meta_xy_grid,
            #     grid_res_deg=self.config["grid_resolution_deg"],
            #     domain_pad_deg=self.config["domain_pad_deg"],
            #     wet_thr=1.0,
            #     dry_thr=0.0,
            #     ok_model="exponential",
            #     ok_range_km=15.0,
            #     ok_nugget_frac=0.5,
            #     min_pts_ok=self.config["min_pts_ok"],
            #     support_k=self.config["support_k"],
            #     support_radius_km=self.config["support_radius_km"],
            #     drizzle_to_zero=self.config["drizzle_to_zero"],
            #     n_jobs=self.config["n_jobs"],
            #     parallel_backend_name="processes",
            #     outside_support_fill=float("nan"),
            #     insufficient_training_fill=float("nan"),
            #     smooth_kernel_px=self.config["smooth_kernel_px"],
            #     smooth_fill_holes=True,
            # )

            grid_ds, diag = pipeline.grid_rain_15min_rainlink_ok_full_ghana(
            df_s5,
            meta_xy_grid,
            grid_res_deg=self.config["grid_resolution_deg"],
            domain_pad_deg=self.config["domain_pad_deg"],   # kept but ignored when fixed_extent is used

            # Ghana AOI used in recent Rainboo testing
            fixed_extent=(-3.5, 1.5, 4.5, 11.5), # lon_min, lon_max, lat_min, lat_max
            
            # wet/dry classification for gridding support/training
            wet_thr=0.8,
            dry_thr=0.05,

            # OK / variogram
            ok_model="exponential",
            ok_range_km=22.0,
            ok_nugget_frac=0.4,
            min_pts_ok=self.config["min_pts_ok"],

            # link-path support/confidence geometry
            support_geometry="link_path",
            n_support_points_per_link=5,
            support_point_spacing_km=2.0,
            min_support_points_per_link=3,
            max_support_points_per_link=7,
            use_length_conditioned_support_points=True,

            # support mask controls
            support_k=self.config["support_k"],
            support_radius_km=self.config["support_radius_km"],
            dry_radius_km=3.0,
            use_dry_constraint=True,

            # confidence layer controls
            use_soft_confidence=True,
            confidence_floor=0.03,
            confidence_power=1.1,
            confidence_dry_penalty_weight=0.50,

            # rainfall handling
            drizzle_to_zero=self.config["drizzle_to_zero"],
            outside_support_fill=float("nan"),
            insufficient_training_fill=float("nan"),

            # scientific field: keep conservative
            smooth_kernel_px=1,
            smooth_fill_holes=False,

            # support mask cleanup
            clean_support=True,
            support_closing_iters=1,
            support_fill_holes=True,
            support_max_hole_px=20,

            # display field retained internally, not exported operationally by default
            make_display_field=True,
            display_smooth=True,
            display_smooth_kernel_px=3,
            display_smooth_fill_holes=False,
            apply_display_edge_taper=True,
            display_edge_taper_pixels=6,
            display_edge_taper_min_weight=0.05,

            # categorical Rainboo decision layer
            make_coverage_quality=True,
            coverage_quality_med_thr=0.50,
            coverage_quality_high_thr=0.75,

            # parallel processing
            n_jobs=self.config["n_jobs"],
            parallel_backend_name="processes",

            # return full Dataset, not only DataArray
            return_dataset=True,
            )

            logger.info("Rainfall gridding diagnostics: %s", diag.get("counts"))
            logger.info("Generated gridded CML timestamps: %s", grid_ds.sizes.get("time"))

            self.output_dir.mkdir(parents=True, exist_ok=True)
            generated_files = pipeline.save_15min_grid_and_points_netcdf_for_day(
                grid_data=grid_ds,
                df_s5=df_s5,
                meta_xy=meta_xy_grid,
                out_dir=str(self.output_dir),
                day=latest_dt.date().strftime("%Y-%m-%d"),
                base_name=self.config["base_name"],
                version="V1",

                # Keep Rainboo/TAHMO operational product clean:
                # export only R_mm_per_h + support/confidence/quality + link midpoint rainfall.
                include_display_field=False,
                )
            logger.info("Generated NetCDF files: %s", len(generated_files))

            uploaded_files = self._upload_to_s3(generated_files)
            self._persist_artifacts(uploaded_files, latest_dt, recent_files)
            message = (
                      f"Generated {len(generated_files)} operational CML rainfall NetCDF artifact(s) "
                      f"for {latest_dt.date().isoformat()} based on latest timestamp {latest_dt.isoformat()}."
                      )
            return self._result(
                "success",
                message,
                generated_files=generated_files,
                uploaded_files=uploaded_files,
                latest_timestamp=latest_dt.isoformat(),
            )
        except Exception as exc:
            logger.exception("Rainfall map generation failed")
            return self._result("failed", str(exc))

    def _load_pipeline_module(self):
        missing_modules = self._detect_missing_modules(
            ["joblib", "matplotlib", "netCDF4", "pycomlink", "pykrige", "scipy", "sklearn", "xarray"]
        )
        if missing_modules:
            raise RuntimeError(
                "Rainfall pipeline dependencies are missing: "
                + ", ".join(missing_modules)
            )

        return importlib.import_module("tahmocml.utils.rainfall_pipeline")

    def _select_recent_files(self, pipeline):
        if not self.raw_archive_dir.exists():
            return None, []

        all_files = sorted(self.raw_archive_dir.glob("*.txt"))
        file_datetimes = []
        for file_path in all_files:
            dt = pipeline.extract_datetime_from_filename(file_path.name)
            if dt is not None:
                file_datetimes.append((dt, file_path))

        if not file_datetimes:
            return None, []

        latest_dt, _ = max(file_datetimes, key=lambda item: item[0])
        cutoff_time = latest_dt - timedelta(hours=self.config["lookback_hours"])
        recent_files = [
            file_path for dt, file_path in sorted(file_datetimes, key=lambda item: item[0])
            if cutoff_time <= dt <= latest_dt
        ]
        return latest_dt, recent_files

    def _upload_to_s3(self, file_paths: List[str]) -> List[Dict[str, str]]:
        if not file_paths or not self.config.get("upload_to_s3"):
            return []

        missing_config = [
            key for key in ("s3_bucket", "s3_endpoint_url", "ibm_api_key_id", "ibm_service_instance_id")
            if not self.config.get(key)
        ]
        if missing_config:
            raise RuntimeError(
                "IBM COS upload is enabled but these settings are missing: "
                + ", ".join(missing_config)
            )

        missing_modules = self._detect_missing_modules(["ibm_boto3", "ibm_botocore"])
        if missing_modules:
            raise RuntimeError("IBM COS upload requires ibm-boto3 and ibm-cos-sdk-core to be installed.")

        client = create_ibm_cos_client(self.config)

        uploaded_files = []
        prefix = str(self.config.get("s3_prefix", "")).strip("/")
        for file_path in file_paths:
            path = Path(file_path)
            key = f"{prefix}/{path.name}" if prefix else path.name
            extra_args = self._build_s3_extra_args(path)
            client.upload_file(str(path), self.config["s3_bucket"], key, ExtraArgs=extra_args)
            uploaded_files.append({
                "file_name": path.name,
                "storage_key": key,
                "download_url": self._build_object_url(key),
                "content_type": extra_args.get("ContentType", "application/octet-stream"),
            })

        return uploaded_files

    def _build_s3_extra_args(self, file_path: Path) -> Dict[str, str]:
        content_type, _ = mimetypes.guess_type(file_path.name)
        if file_path.suffix == ".nc":
            content_type = "application/x-netcdf"
        return {"ContentType": content_type or "application/octet-stream"}

    def _detect_missing_modules(self, modules: List[str]) -> List[str]:
        missing = []
        for module_name in modules:
            if importlib.util.find_spec(module_name) is None:
                missing.append(module_name)
        return missing

    def _build_object_url(self, key: str) -> str:
        endpoint = str(self.config["s3_endpoint_url"]).rstrip("/")
        bucket = self.config["s3_bucket"]
        return f"{endpoint}/{bucket}/{key}"

    def _persist_artifacts(self, uploaded_files: List[Dict[str, str]], latest_dt, recent_files: List[Path]) -> None:
        if not uploaded_files:
            return

        latest_source_name = recent_files[-1].name if recent_files else ""
        source_file = CMLFile.objects.filter(file_name=latest_source_name).order_by('-processed_at').first()
        rainfall_timestamp = latest_dt.replace(tzinfo=dt_timezone.utc)

        for uploaded in uploaded_files:
            RainfallMapArtifact.objects.update_or_create(
                rainfall_timestamp=rainfall_timestamp,
                file_name=uploaded["file_name"],
                defaults={
                    "source_file": source_file,
                    "source_file_name": latest_source_name,
                    "storage_key": uploaded["storage_key"],
                    "download_url": uploaded["download_url"],
                    "content_type": uploaded.get("content_type"),
                },
            )

    def _result(
        self,
        status: str,
        message: str,
        generated_files: Optional[List[str]] = None,
        uploaded_files: Optional[List[str]] = None,
        latest_timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        return RainfallRunResult(
            status=status,
            message=message,
            generated_files=generated_files or [],
            uploaded_files=uploaded_files or [],
            latest_timestamp=latest_timestamp,
        ).__dict__


def create_ibm_cos_client(config):
    import ibm_boto3
    from ibm_botocore.client import Config

    return ibm_boto3.client(
        "s3",
        ibm_api_key_id=config["ibm_api_key_id"],
        ibm_service_instance_id=config["ibm_service_instance_id"],
        ibm_auth_endpoint=config["ibm_auth_endpoint"],
        config=Config(signature_version="oauth"),
        endpoint_url=config["s3_endpoint_url"],
    )
