from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

@dataclass
class Config:

    root: Path = ROOT

    data_path: Path = Path("data")
    output_path: Path = Path("output")

    def __post_init__(self):
        self.data_path = self.root / self.data_path
        self.output_path = self.root / self.output_path

        self.input_photo = self.data_path / "photoes"
        self.input_video = self.data_path / "videos"
        self.aruco_markers = self.data_path / "aruco_markers"

        self.output_photo = self.output_path / "photoes"
        self.output_video = self.output_path / "videos"

        for path in (self.aruco_markers, self.output_photo, self.output_video):
            path.mkdir(parents=True, exist_ok=True)


settings = Config()