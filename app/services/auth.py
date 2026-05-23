"""
用户认证服务：注册、登录
"""

from app.services.storage import StorageService
from app.models.user import User


class AuthService:

    def __init__(self, storage: StorageService):
        self.storage = storage

    def register(self, username: str, password: str) -> tuple[bool, str, User | None]:
        """注册新用户"""
        if not username or not password:
            return False, "用户名和密码不能为空", None
        if len(username) < 2:
            return False, "用户名至少 2 个字符", None
        if len(password) < 4:
            return False, "密码至少 4 个字符", None
        if self.storage.user_exists(username):
            return False, "用户名已存在", None

        user_id = self.storage.generate_user_id()
        user = User.create(user_id, username, password)
        user.onboarding_status = "new"
        self.storage.save_user(user)
        return True, "注册成功", user

    def login(self, username: str, password: str) -> tuple[bool, str, User | None]:
        """用户登录"""
        user = self.storage.get_user(username)
        if user is None:
            return False, "用户不存在", None
        if not user.verify_password(password):
            return False, "密码错误", None

        user.touch_login()
        self.storage.save_user(user)
        return True, "登录成功", user

    def update_user(self, user: User) -> bool:
        """更新用户信息"""
        return self.storage.save_user(user)
