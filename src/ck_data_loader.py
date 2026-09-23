"""Load human ratings from the Cowen-Keltner emotional videos CSV."""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from .ck_config import CKConfig


class CKDataLoader:
    """Load and access CK human ratings and video metadata."""

    def __init__(self, config: Optional[CKConfig] = None):
        self.config = config or CKConfig()
        self._human_ratings: Optional[pd.DataFrame] = None

    def load_human_ratings(self) -> pd.DataFrame:
        """Load human valence/arousal ratings from the CK CSV.

        Returns:
            DataFrame with columns: video_id, valence, arousal
        """
        if self._human_ratings is not None:
            return self._human_ratings

        csv_path = self.config.ck_human_ratings_csv
        if not csv_path.exists():
            raise FileNotFoundError(f"CK human ratings CSV not found: {csv_path}")

        raw = pd.read_csv(csv_path)

        # Extract video_id from Filename column (e.g., "0001.mp4" -> "0001")
        df = pd.DataFrame({
            "video_id": raw["Filename"].str.replace(".mp4", "", regex=False),
            "valence": raw["valence"].astype(float),
            "arousal": raw["arousal"].astype(float),
        })

        self._human_ratings = df
        return df

    def load_human_ratings_full(self) -> pd.DataFrame:
        """Load all 34 categories + 14 dimensions from the CK CSV.

        Returns:
            DataFrame with columns: video_id, + 34 category cols + 14 dimension cols.
            Category values are 0-1 proportions (as in the original CSV).
            Dimension values are 1-9 Likert scale.
        """
        from .ck_dimensions import CK_CATEGORIES, CK_DIMENSIONS

        csv_path = self.config.ck_human_ratings_csv
        if not csv_path.exists():
            raise FileNotFoundError(f"CK human ratings CSV not found: {csv_path}")

        raw = pd.read_csv(csv_path)

        df = pd.DataFrame()
        df["video_id"] = raw["Filename"].str.replace(".mp4", "", regex=False)

        # 34 emotion categories (0-1 proportions)
        for cat in CK_CATEGORIES:
            if cat in raw.columns:
                df[cat] = raw[cat].astype(float)

        # 14 affective dimensions (1-9 Likert)
        for dim in CK_DIMENSIONS:
            if dim in raw.columns:
                df[dim] = raw[dim].astype(float)

        return df

    def get_video_ids(self) -> List[str]:
        """Return list of all video IDs from the CSV."""
        df = self.load_human_ratings()
        return df["video_id"].tolist()

    def get_video_path(self, video_id: str) -> Path:
        """Get path to a CK video file."""
        return self.config.get_video_path(video_id)
