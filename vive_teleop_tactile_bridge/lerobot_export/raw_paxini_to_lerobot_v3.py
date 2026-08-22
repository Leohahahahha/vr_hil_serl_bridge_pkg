from __future__ import annotations

from .common import ExportProfile, export_main


PROFILE = ExportProfile(
    name="paxini",
    tactile_msg_package="paxini_tactile",
    tactile_description="Raw flattened Paxini tactile frame exported from zarr without normalization.",
)


def main() -> None:
    export_main(PROFILE, "Export raw Paxini tactile dataset to a LeRobot v3-style dataset.")


if __name__ == "__main__":
    main()

