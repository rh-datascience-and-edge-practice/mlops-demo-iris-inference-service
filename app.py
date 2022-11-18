"""Main entrypoint for starting container."""

import os

from iris_inference_service.config import app_cfg

if __name__ == "__main__":
    os.system(  # noqa - S605
        "seldon-core-microservice iris_inference_service.iris.Iris "
        f"--service-type {app_cfg.model.service_type} "
        f"--log-level {app_cfg.log.level}"
    )
