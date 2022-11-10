"""Configuration file for managing environment variables."""
from dotenv import load_dotenv

import environ


@environ.config(prefix="")
class AppConfig:
    """Application configuration object used for managing environment variables."""

    @environ.config
    class Log:
        """App configuration object used for managing logging settings."""

        level = environ.var("DEBUG")

        @level.validator
        def _ensure_level_is_valid(self, var, level):
            valid_options = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if level not in valid_options:
                raise ValueError(
                    f"LOG_LEVEL is invalid.  Must be set to one of the following: {valid_options}"
                )

    log = environ.group(Log)

    @environ.config
    class Model:
        """App configuration object used for managing model settings."""

        name = environ.var("Iris", help="The name of the model.")
        service_type = environ.var("MODEL")
        file = environ.var(
            "../models/iris-model.h5",
            help="The file location of the model.",
        )

    model = environ.group(Model)


# load environment and assign it to the environ config
load_dotenv()
app_cfg = environ.to_config(AppConfig)

if __name__ == "__main__":
    print(environ.generate_help(AppConfig, display_defaults=True))
