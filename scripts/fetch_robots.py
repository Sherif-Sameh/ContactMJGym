import itertools
import shutil
from pathlib import Path

from robot_descriptions import fr3_mj_description, ur10e_mj_description

import contact_gym


def fetch_robot(mod) -> None:
    root = Path(contact_gym.__file__).parent / "robots"
    pkg_path = Path(mod.PACKAGE_PATH)
    robot_path = root / pkg_path.stem
    # move full directory
    if not robot_path.exists():
        shutil.copytree(pkg_path, robot_path)
    # move assets
    if not (robot_path / "assets").exists():
        shutil.copytree(pkg_path / "assets", robot_path / "assets")
    # remove unneccesary files
    image_files = itertools.chain.from_iterable(
        robot_path.glob(ext) for ext in ["*.png", "*.jpg", "*.jpeg"]
    )
    for img_file in image_files:
        img_file.unlink()
    (robot_path / "scene.xml").unlink(missing_ok=True)


if __name__ == "__main__":
    # fetch Franka Robotics FR3
    fetch_robot(fr3_mj_description)
    # fetch Universal Robots UR10e
    fetch_robot(ur10e_mj_description)
