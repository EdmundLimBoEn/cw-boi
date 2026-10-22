"""Run with python3 -m unittest neural.test_server_limits (no ML dependencies)."""

import contextlib
import importlib.util
from pathlib import Path
import io
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch


# Import the real HTTP implementation while isolating optional ML dependencies.
spec = importlib.util.spec_from_file_location('server_under_test', Path(__file__).with_name('server.py'))
server = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {name: MagicMock() for name in (
    'numpy', 'torch', 'huggingface_hub', 'morseformer', 'morseformer.decoding',
    'morseformer.decoding.streaming', 'streaming', 'morseformer.models',
    'morseformer.models.rnnt', 'scripts.decode_audio',
)}):
    spec.loader.exec_module(server)
server.torch.inference_mode = contextlib.nullcontext


class Connection:
    """Socket-shaped transport; real handler setup, parsing and dispatch still run."""
    def __init__(self, request):
        self.reader = io.BytesIO(request)
        self.output = bytearray()
        self.timeout = None
        self.timeout_at_first_read = None

    def settimeout(self, timeout):
        self.timeout = timeout

    def makefile(self, *_args):
        self.timeout_at_first_read = self.timeout
        return self.reader

    def sendall(self, data):
        self.output.extend(data)


class ServerLimitsTest(unittest.TestCase):
    def setUp(self):
        server.model_lock = threading.Lock()
        server.active_handlers = threading.BoundedSemaphore(8)
        server.sessions.clear()

    def run_request(self, request, connection=None):
        connection = connection or Connection(request)
        server.Handler(connection, ('127.0.0.1', 1234), object())
        head, body = bytes(connection.output).split(b'\r\n\r\n', 1)
        self.assertIn(b'Content-Type: application/json', head)
        self.assertIn(b'Cache-Control: no-store', head)
        return int(head.split()[1]), server.json.loads(body)

    def test_timeout_is_set_before_request_line(self):
        conn = Connection(b'GET / HTTP/1.1\r\nHost: localhost\r\n\r\n')
        self.assertEqual(self.run_request(None, conn)[0], 404)
        self.assertEqual(conn.timeout_at_first_read, 30)

    def test_eight_slow_connections_reject_ninth_before_headers(self):
        entered = threading.Barrier(9)
        release = threading.Event()
        errors = []

        class SlowHeaders(io.BytesIO):
            def readline(self, *args):
                entered.wait(timeout=5)
                release.wait(timeout=5)
                return b''

        def slow_client():
            conn = Connection(b'')
            conn.reader = SlowHeaders()
            try:
                server.Handler(conn, ('127.0.0.1', 1234), object())
            except Exception as error:
                errors.append(error)

        workers = [threading.Thread(target=slow_client) for _ in range(8)]
        for worker in workers:
            worker.start()
        try:
            entered.wait(timeout=5)
            # Empty input would normally close silently; overload must respond
            # before reading even a request line.
            self.assertEqual(self.run_request(b'')[0], 503)
        finally:
            release.set()
            for worker in workers:
                worker.join(timeout=5)
        self.assertEqual(errors, [])
        self.assertEqual(self.run_request(b'GET / HTTP/1.1\r\nHost: localhost\r\n\r\n')[0], 404)

    def test_busy_model_rejects_without_reading_body(self):
        conn = Connection(b'POST /api/decode HTTP/1.1\r\nHost: localhost\r\nContent-Length: 19200000\r\n\r\n')
        conn.reader.read = MagicMock(side_effect=AssertionError('Body read while model busy'))
        with server.model_lock:
            status, body = self.run_request(None, conn)
        self.assertEqual(status, 503)
        self.assertIn('error', body)
        conn.reader.read.assert_not_called()

    def test_decode_response_and_lock_release(self):
        decode = MagicMock(return_value=' CQ ')
        conn = Connection(b'POST /api/decode HTTP/1.1\r\nHost: localhost\r\nContent-Length: 4\r\n\r\n1234')
        original_read = conn.reader.read

        def read_under_lock(length):
            self.assertTrue(server.model_lock.locked())
            return original_read(length)

        conn.reader.read = read_under_lock
        with patch.dict(server.engines, {'rnnt': {
            'network': object(), 'decode': decode, 'device': 'cpu', 'checkpoint': 'stub.pt',
        }}, clear=True), patch.object(server, 'samples_from_bytes', return_value=[0] * 160):
            self.assertEqual(self.run_request(None, conn), (200, {'text': 'CQ', 'model': 'stub.pt', 'engine': 'rnnt'}))
            decode.assert_called_once()
        self.assertTrue(server.model_lock.acquire(timeout=1))
        server.model_lock.release()

    def test_incomplete_body_releases_lock(self):
        request = b'POST /api/decode HTTP/1.1\r\nHost: localhost\r\nContent-Length: 4\r\n\r\nx'
        self.assertEqual(self.run_request(request), (400, {'error': 'Incomplete audio request.'}))
        self.assertTrue(server.model_lock.acquire(timeout=1))
        server.model_lock.release()


if __name__ == '__main__':
    unittest.main()
