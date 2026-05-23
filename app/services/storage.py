"""
JSON 文件存储引擎
"""

import json
import csv
from pathlib import Path
from datetime import datetime, timezone

try:
    from filelock import FileLock
except ImportError:
    FileLock = None

from app.models.user import User, UserProfile
from app.config import USERS_FILE, PROFILES_DIR, TRADES_DIR, QUESTIONNAIRES_DIR


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_lock(file_path: Path):
    """返回 filelock 对象（如果可用），否则返回空上下文管理器"""
    if FileLock:
        return FileLock(str(file_path) + ".lock", timeout=5)
    return _NullLock()


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class StorageService:

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or Path(__file__).parent.parent / "data"
        self.users_file = self.data_dir / "users.json"
        self.profiles_dir = self.data_dir / "profiles"
        self.trades_dir = self.data_dir / "trades"
        self.questionnaires_dir = self.data_dir / "questionnaires"
        self._ensure_dirs()
        self._ensure_users_file()

    def _ensure_dirs(self):
        for d in [self.profiles_dir, self.trades_dir, self.questionnaires_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _ensure_users_file(self):
        if not self.users_file.exists():
            self._write_json(self.users_file, {"users": []})

    # ===== JSON 读写 =====

    def _read_json(self, path: Path) -> dict:
        with _safe_lock(path):
            if not path.exists():
                return {}
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    def _write_json(self, path: Path, data: dict):
        with _safe_lock(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    # ===== 用户 CRUD =====

    def save_user(self, user: User) -> bool:
        data = self._read_json(self.users_file)
        users = data.get("users", [])
        # 更新或追加
        for i, u in enumerate(users):
            if u["user_id"] == user.user_id:
                users[i] = user.to_dict()
                self._write_json(self.users_file, {"users": users})
                return True
        users.append(user.to_dict())
        self._write_json(self.users_file, {"users": users})
        return True

    def get_user(self, username: str) -> User | None:
        data = self._read_json(self.users_file)
        for u in data.get("users", []):
            if u["username"] == username:
                return User.from_dict(u)
        return None

    def get_user_by_id(self, user_id: str) -> User | None:
        data = self._read_json(self.users_file)
        for u in data.get("users", []):
            if u["user_id"] == user_id:
                return User.from_dict(u)
        return None

    def user_exists(self, username: str) -> bool:
        return self.get_user(username) is not None

    def list_users(self) -> list[User]:
        data = self._read_json(self.users_file)
        return [User.from_dict(u) for u in data.get("users", [])]

    def generate_user_id(self) -> str:
        users = self.list_users()
        max_id = 0
        for u in users:
            if u.user_id.startswith("U_"):
                try:
                    num = int(u.user_id[2:])
                    max_id = max(max_id, num)
                except ValueError:
                    pass
        return f"U_{max_id + 1:04d}"

    # ===== 画像 CRUD =====

    def save_profile(self, profile: UserProfile) -> bool:
        path = self.profiles_dir / f"{profile.user_id}.json"
        self._write_json(path, profile.to_dict())
        return True

    def get_profile(self, user_id: str) -> UserProfile | None:
        path = self.profiles_dir / f"{user_id}.json"
        if not path.exists():
            return None
        data = self._read_json(path)
        return UserProfile.from_dict(data)

    def delete_profile(self, user_id: str) -> bool:
        path = self.profiles_dir / f"{user_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    # ===== 交易数据 =====

    def save_trades(self, user_id: str, trades_df) -> bool:
        """保存交易数据到 CSV（追加模式，每次上传一个文件）"""
        upload_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.trades_dir / f"{user_id}_{upload_id}.csv"
        trades_df.to_csv(path, index=False)
        return True

    def load_trades(self, user_id: str):
        """加载用户所有上传的交易数据（合并为一个 DataFrame）"""
        import pandas as pd
        files = sorted(self.trades_dir.glob(f"{user_id}_*.csv"))
        if not files:
            return None
        dfs = []
        for f in files:
            df = pd.read_csv(f)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else None

    def list_trade_uploads(self, user_id: str) -> list[dict]:
        """列出上传历史"""
        files = sorted(self.trades_dir.glob(f"{user_id}_*.csv"))
        uploads = []
        for f in files:
            import pandas as pd
            df = pd.read_csv(f)
            uploads.append({
                "filename": f.name,
                "upload_date": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "trade_count": len(df),
            })
        return uploads

    # ===== 问卷存档 =====

    def save_questionnaire_results(self, user_id: str, level: str, answers: dict) -> bool:
        path = self.questionnaires_dir / f"{user_id}_{level}.json"
        data = {
            "user_id": user_id,
            "level": level,
            "answers": answers,
            "completed_at": _now(),
        }
        self._write_json(path, data)
        return True

    def load_questionnaire_results(self, user_id: str, level: str) -> dict | None:
        path = self.questionnaires_dir / f"{user_id}_{level}.json"
        if not path.exists():
            return None
        return self._read_json(path)

    def list_completed_levels(self, user_id: str) -> list[str]:
        """列出该用户已完成的问卷级别"""
        completed = []
        for level in ["L1", "L2", "L3"]:
            if self.load_questionnaire_results(user_id, level) is not None:
                completed.append(level)
        return completed
