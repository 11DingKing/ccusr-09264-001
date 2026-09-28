"""回归：幂等结果必须按用户隔离，且跨重启保持。

业务对象：修复幂等结果跨用户回放。

修复前（可观察的错误响应）：
- idempotency 表以 idempotency_key 为全局主键；
- 用户 B 携带与用户 A 相同的 Idempotency-Key 请求时，直接回放 A 的成功响应：
  HTTP 200/201，body 为 A 创建的资源 id，且 replayed=true —— 跨用户泄露；
- 由于结果已被回放，B 永远无法用该 key 得到自己的独立结果。

修复后（本测试锁定的可观察响应）：
- 同一用户用同一 key 重试：回放首次结果（相同资源 id，replayed=true）；
- 不同用户用同一 key：得到各自独立的结果（新资源 id，replayed=false），
  访问对方资源被授权层拒绝（403 permission_denied）；
- 服务重启（重新打开同一数据库）后上述行为不变。
"""
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

from service_09252_006.api.http_api import HttpApiServer
from service_09252_006.application.container import ApplicationContext
from tests.support import Harness
from tests.test_http_api import ApiClient

BOOTSTRAP = "boot-secret"
SHARED_KEY = "shared-key-2026-09"


class IdempotencyIsolationHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.h = Harness()
        self._start_server()

    def _start_server(self) -> None:
        self.server = HttpApiServer(
            self.h.ctx, host="127.0.0.1", port=0, bootstrap_token=BOOTSTRAP
        )
        self.server.start()
        host, port = self.server.address
        self.base = f"http://{host}:{port}"
        self.boot = ApiClient(self.base, bootstrap=BOOTSTRAP)

    def _restart_server(self) -> None:
        """模拟进程重启：停服、关闭上下文，用同一数据库文件重新打开。"""
        self.server.stop()
        self.h.ctx.close()
        ctx = ApplicationContext(self.h.db_path)
        self.h.ctx = ctx
        self.h.repo = ctx.repo
        self._start_server()

    def tearDown(self) -> None:
        self.server.stop()
        self.h.close()

    def _create_user(self, user_id, institution_id, token):
        status, _ = self.boot.request(
            "POST", "/v1/admin/users",
            {"user_id": user_id, "roles": ["institution_admin"],
             "institution_id": institution_id},
        )
        self.assertEqual(status, 201)
        status, _ = self.boot.request(
            "POST", "/v1/admin/tokens",
            {"user_id": user_id, "token": token},
        )
        self.assertEqual(status, 201)
        return ApiClient(self.base, token=token)

    def _create_material(self, client, title, key):
        return client.request(
            "POST", "/v1/materials",
            {"kind": "syllabus", "title": title},
            idempotency_key=key,
        )

    def test_idempotency_scoped_per_user_and_survives_restart(self) -> None:
        alice = self._create_user("alice", "inst-a", "tok-alice")
        bob = self._create_user("bob", "inst-b", "tok-bob")

        # --- Alice 首次成功 ---
        status, mat_alice = self._create_material(alice, "Alice 的材料", SHARED_KEY)
        self.assertEqual(status, 201, mat_alice)
        self.assertFalse(mat_alice["replayed"])
        self.assertEqual(mat_alice["institution_id"], "inst-a")
        self.assertEqual(mat_alice["title"], "Alice 的材料")
        alice_id = mat_alice["material_id"]

        # --- Alice 同 key 重试：回放自己的首次结果（同一用户重试仍幂等）---
        status, replay = self._create_material(alice, "Alice 的材料", SHARED_KEY)
        self.assertEqual(status, 201)
        self.assertEqual(replay["material_id"], alice_id)
        self.assertTrue(replay["replayed"])

        # --- Bob 用同一 key：必须得到独立结果，而不是回放 Alice 的成功响应 ---
        # 修复前这里的可观察响应是：
        #   201 {"material_id": <alice_id>, "replayed": true,
        #        "institution_id": "inst-a", "title": "Alice 的材料"}
        # 即跨用户回放；修复后是 Bob 自己的新建结果。
        status, mat_bob = self._create_material(bob, "Bob 的材料", SHARED_KEY)
        self.assertEqual(status, 201, mat_bob)
        self.assertFalse(mat_bob["replayed"])
        self.assertEqual(mat_bob["institution_id"], "inst-b")
        self.assertEqual(mat_bob["title"], "Bob 的材料")
        self.assertNotEqual(mat_bob["material_id"], alice_id)

        # Bob 再试同 key：回放的是 Bob 自己的结果，与 Alice 无关
        status, bob_replay = self._create_material(bob, "Bob 的材料", SHARED_KEY)
        self.assertEqual(status, 201)
        self.assertEqual(bob_replay["material_id"], mat_bob["material_id"])
        self.assertTrue(bob_replay["replayed"])

        # Bob 无法访问 Alice 的资源：授权层拒绝（不同用户得到拒绝）
        status, body = bob.request("GET", f"/v1/materials/{alice_id}")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"]["code"], "permission_denied")

        # --- 重启进程（同一数据库文件）---
        self._restart_server()
        alice = ApiClient(self.base, token="tok-alice")
        bob = ApiClient(self.base, token="tok-bob")

        # Alice 重启后重试：仍然回放她自己的首次结果（幂等记录持久化）
        status, replay_after = self._create_material(
            alice, "Alice 的材料", SHARED_KEY
        )
        self.assertEqual(status, 201)
        self.assertEqual(replay_after["material_id"], alice_id)
        self.assertTrue(replay_after["replayed"])

        # Bob 重启后用同一 key：回放的仍是 Bob 自己的结果，看不到 Alice 的
        status, bob_after = self._create_material(
            bob, "Bob 的材料", SHARED_KEY
        )
        self.assertEqual(status, 201)
        self.assertEqual(bob_after["material_id"], mat_bob["material_id"])
        self.assertTrue(bob_after["replayed"])
        self.assertNotEqual(bob_after["material_id"], alice_id)

        # 再来第三个用户使用同一 key：依旧独立，不回放任何既有结果
        carol = self._create_user("carol", "inst-c", "tok-carol")
        status, mat_carol = self._create_material(carol, "Carol 的材料", SHARED_KEY)
        self.assertEqual(status, 201, mat_carol)
        self.assertFalse(mat_carol["replayed"])
        self.assertEqual(mat_carol["institution_id"], "inst-c")
        self.assertNotIn(
            mat_carol["material_id"], (alice_id, mat_bob["material_id"])
        )


class IdempotencyProcessRestartTests(unittest.TestCase):
    """真正的进程重启：CLI 子进程 -> 停止 -> 以同一数据库文件再次启动。

    覆盖 WAL 跨进程恢复：首次响应已落盘，重启后同用户仍回放、跨用户仍隔离。
    """

    BOOTSTRAP = "boot-secret"

    def _serve(self, db_path: str) -> subprocess.Popen:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "service_09252_006.cli", "serve",
                "--db", db_path,
                "--host", "127.0.0.1", "--port", "0",
                "--bootstrap-token", self.BOOTSTRAP,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # port=0 时实际端口打印在首行：... http://127.0.0.1:<port>
        assert proc.stdout is not None
        line = proc.stdout.readline()
        self.assertIn("http://", line, f"服务未正常启动: {line}")
        self._base = line.split("http://", 1)[1].strip()
        for _ in range(50):
            try:
                urllib.request.urlopen(
                    f"http://{self._base}/healthz", timeout=1
                ).read()
                break
            except OSError:
                time.sleep(0.1)
        else:
            self.fail("健康检查持续失败")
        return proc

    def _stop(self, proc: subprocess.Popen) -> None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        if proc.stdout is not None:
            proc.stdout.close()

    def test_isolation_holds_across_real_process_restart(self) -> None:
        fd, db_path = tempfile.mkstemp(prefix="qe-restart-", suffix=".db")
        os.close(fd)
        os.unlink(db_path)
        try:
            proc = self._serve(db_path)
            try:
                boot = ApiClient("http://" + self._base, bootstrap=self.BOOTSTRAP)
                for uid, inst, tok in (
                    ("alice", "inst-a", "tok-alice"),
                    ("bob", "inst-b", "tok-bob"),
                ):
                    status, _ = boot.request(
                        "POST", "/v1/admin/users",
                        {"user_id": uid, "roles": ["institution_admin"],
                         "institution_id": inst},
                    )
                    self.assertEqual(status, 201)
                    status, _ = boot.request(
                        "POST", "/v1/admin/tokens",
                        {"user_id": uid, "token": tok},
                    )
                    self.assertEqual(status, 201)

                alice = ApiClient("http://" + self._base, token="tok-alice")
                bob = ApiClient("http://" + self._base, token="tok-bob")

                def create(client, title):
                    return client.request(
                        "POST", "/v1/materials",
                        {"kind": "syllabus", "title": title},
                        idempotency_key=SHARED_KEY,
                    )

                status, first = create(alice, "Alice 的材料")
                self.assertEqual(status, 201, first)
                alice_id = first["material_id"]
                status, other = create(bob, "Bob 的材料")
                self.assertEqual(status, 201, other)
                self.assertNotEqual(other["material_id"], alice_id)
                self.assertFalse(other["replayed"])
            finally:
                self._stop(proc)

            # ---- 用同一数据库文件启动全新进程 ----
            proc = self._serve(db_path)
            try:
                alice = ApiClient("http://" + self._base, token="tok-alice")
                bob = ApiClient("http://" + self._base, token="tok-bob")

                status, replay = alice.request(
                    "POST", "/v1/materials",
                    {"kind": "syllabus", "title": "Alice 的材料"},
                    idempotency_key=SHARED_KEY,
                )
                self.assertEqual(status, 201, replay)
                self.assertEqual(replay["material_id"], alice_id)
                self.assertTrue(replay["replayed"])

                status, bob_replay = bob.request(
                    "POST", "/v1/materials",
                    {"kind": "syllabus", "title": "Bob 的材料"},
                    idempotency_key=SHARED_KEY,
                )
                self.assertEqual(status, 201, bob_replay)
                self.assertEqual(bob_replay["material_id"], other["material_id"])
                self.assertTrue(bob_replay["replayed"])
            finally:
                self._stop(proc)
        finally:
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.unlink(db_path + suffix)
                except FileNotFoundError:
                    pass


class IdempotencyV1MigrationTests(unittest.TestCase):
    """v1 数据库（全局幂等键）升级到 v2 后，旧记录不再被任何用户回放。"""


    def test_v1_records_become_ownerless_and_never_replay(self) -> None:
        fd, db_path = tempfile.mkstemp(prefix="qe-v1-", suffix=".db")
        os.close(fd)
        os.unlink(db_path)
        try:
            conn = sqlite3.connect(db_path)
            # 复刻 v1 的 idempotency 表结构与 user_version
            conn.execute(
                "CREATE TABLE idempotency ("
                "idempotency_key TEXT PRIMARY KEY,"
                " result_json TEXT NOT NULL,"
                " created_at TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO idempotency VALUES(?,?,?)",
                (
                    SHARED_KEY,
                    '{"material_id": "mat_alice_legacy", "replayed": false}',
                    "",
                ),
            )
            conn.execute("PRAGMA user_version = 1")
            conn.commit()
            conn.close()

            # 打开即触发 v1 -> v2 迁移
            ctx = ApplicationContext(db_path)
            try:
                # 旧记录仍保留，但归属为 NULL，任何真实用户都查不到
                self.assertIsNone(
                    ctx.repo.get_idempotent_result("alice", SHARED_KEY)
                )
                self.assertIsNone(
                    ctx.repo.get_idempotent_result("bob", SHARED_KEY)
                )
                row = ctx.repo._conn.execute(  # type: ignore[attr-defined]
                    "SELECT user_id, result_json FROM idempotency"
                    " WHERE idempotency_key = ?",
                    (SHARED_KEY,),
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertIsNone(row["user_id"])
                self.assertIn("mat_alice_legacy", row["result_json"])
                self.assertEqual(
                    ctx.repo._conn.execute(  # type: ignore[attr-defined]
                        "PRAGMA user_version"
                    ).fetchone()[0],
                    2,
                )
            finally:
                ctx.close()
        finally:
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.unlink(db_path + suffix)
                except FileNotFoundError:
                    pass


if __name__ == "__main__":
    unittest.main()
