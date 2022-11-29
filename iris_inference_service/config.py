"""Configuration file for managing environment variables."""
from pathlib import Path

from dotenv import load_dotenv

import environ


@environ.config(prefix="")
class AppConfig:
    """Application configuration object used for managing environment variables."""

    def __init__():
        """Load environment variables with dotenv."""
        load_dotenv()

    @environ.config
    class Log:
        """App configuration object used for managing logging settings."""

        level = environ.var("INFO", help="The log level of the service.")

        @level.validator
        def _ensure_level_is_valid(self, var, level):
            valid_options = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if level not in valid_options:
                raise ValueError(
                    f"LOG_LEVEL of {level} is invalid.  "
                    f"Must be set to one of the following: {valid_options}"
                )

    log = environ.group(Log)

    @environ.config
    class Model:
        """App configuration object used for managing model settings."""

        name = environ.var("Iris", help="The name of the model.")

        service_type = environ.var("MODEL")

        @service_type.validator
        def _ensure_service_type_is_valid(self, var, service_type):
            valid_options = [
                "MODEL",
                "ROUTER",
                "TRANSFORMER",
                "COMBINER",
                "OUTLIER_DETECTOR",
            ]
            if service_type not in valid_options:
                raise ValueError(
                    f"MODEL_SERVICE_TYPE of {service_type} is invalid.  "
                    f"Must be set to one of the following: {valid_options}"
                )

        file = environ.var(
            "./models/iris-model.pkl",
            converter=Path,
            help="The file location of the model.",
        )

        @file.validator
        def _ensure_model_file_exists(self, var, file):
            if not file.exists():
                raise ValueError("MODEL_FILE {file} not found.")

    model = environ.group(Model)


app_cfg = environ.to_config(AppConfig)

if __name__ == "__main__":
    print(environ.generate_help(AppConfig, display_defaults=True))
