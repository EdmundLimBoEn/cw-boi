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
        server.active_uploads = threading.BoundedSemaphore(2)
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

    def test_full_upload_slots_reject_without_reading_body(self):
        conn = Connection(b'POST /api/decode HTTP/1.1\r\nHost: localhost\r\nContent-Length: 19200000\r\n\r\n')
        conn.reader.read1 = MagicMock(side_effect=AssertionError('Upload limit exceeded'))
        with server.active_uploads, server.active_uploads:
            self.assertEqual(self.run_request(None, conn)[0], 503)
        conn.reader.read1.assert_not_called()

    def test_decode_response_and_lock_release(self):
        decode = MagicMock(return_value=' CQ ')
        conn = Connection(b'POST /api/decode HTTP/1.1\r\nHost: localhost\r\nContent-Length: 4\r\n\r\n1234')
        original_read = conn.reader.read1

        def read_without_lock(length):
            self.assertFalse(server.model_lock.locked())
            return original_read(length)

        conn.reader.read = conn.reader.read1 = read_without_lock
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

    def post(self, path, body=b''):
        return (f'POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n'.encode() + body)

    def test_stalled_upload_allows_decode_and_session_controls(self):
        conn = Connection(self.post('/api/decode', b'1234'))
        original_read = conn.reader.read1
        decoder = MagicMock()
        engine = {'network': object(), 'decode': MagicMock(return_value='CQ'),
                  'stream': MagicMock(return_value=decoder), 'device': 'cpu', 'checkpoint': 'stub.pt'}

        def paused_read(length):
            self.assertFalse(server.model_lock.locked())
            self.assertEqual(self.run_request(self.post('/api/decode', b'1234'))[0], 200)
            status, data = self.run_request(self.post('/api/stream/start'))
            self.assertEqual(status, 200)
            self.assertEqual(self.run_request(self.post('/api/stream/cancel?session=' + data['session']))[0], 200)
            return original_read(length)

        conn.reader.read = conn.reader.read1 = paused_read
        with patch.dict(server.engines, {'rnnt': engine}, clear=True), patch.object(server, 'samples_from_bytes', return_value=[0] * 160):
            self.assertEqual(self.run_request(None, conn)[0], 200)

    def test_slow_drip_hits_absolute_deadline_and_releases_upload(self):
        conn = Connection(self.post('/api/decode', b'1234'))
        clock = [0.0]
        timeouts = []

        def drip(_length):
            timeouts.append(conn.timeout)
            clock[0] += 11
            return b'x'

        conn.reader.read = conn.reader.read1 = drip
        with patch.object(server.time, 'monotonic', side_effect=lambda: clock[0]):
            self.assertEqual(self.run_request(None, conn), (400, {'error': 'Audio upload deadline exceeded.'}))
        self.assertEqual(timeouts, [30, 19, 8])
        self.assertFalse(server.model_lock.locked())
        self.assertTrue(server.active_uploads.acquire(blocking=False))
        self.assertTrue(server.active_uploads.acquire(blocking=False))

    def test_control_body_rejected_without_network_read_or_model_lock(self):
        for path in ('start', 'finish', 'cancel'):
            conn = Connection(self.post('/api/stream/' + path, b'x'))
            conn.reader.read1 = MagicMock(side_effect=AssertionError('Control body read'))
            with server.model_lock, server.active_uploads, server.active_uploads:
                self.assertEqual(self.run_request(None, conn)[0], 400)
            conn.reader.read1.assert_not_called()

    def test_invalid_audio_is_validated_before_model_lock(self):
        with server.model_lock, patch.object(server, 'samples_from_bytes', side_effect=ValueError('Invalid audio')):
            self.assertEqual(self.run_request(self.post('/api/decode', b'1234')), (400, {'error': 'Invalid audio'}))

    def test_busy_inference_releases_upload_slot(self):
        with server.model_lock, patch.object(server, 'samples_from_bytes', return_value=[0] * 160):
            self.assertEqual(self.run_request(self.post('/api/decode', b'1234'))[0], 503)
        self.assertTrue(server.active_uploads.acquire(blocking=False))
        self.assertTrue(server.active_uploads.acquire(blocking=False))

    def test_decoder_failures_invalidate_mutated_sessions(self):
        for operation in ('feed', 'finish'):
            for exception in (RuntimeError, ValueError):
                with self.subTest(operation=operation, exception=exception):
                    decoder = MagicMock()
                    mutated = []

                    def fail(*_args):
                        mutated.append(True)
                        raise exception('decoder failed')

                    decoder.feed.side_effect = fail
                    decoder.flush.side_effect = fail
                    session = {'engine': 'rnnt', 'decoder': decoder, 'samples': 160,
                               'touched': server.time.monotonic(), 'text': ''}
                    server.sessions['test'] = session
                    with patch.dict(server.engines, {'rnnt': {}}, clear=True), patch.object(server, 'samples_from_bytes', return_value=[0] * 160):
                        status, _ = self.run_request(self.post('/api/stream/' + operation + '?session=test', b'1234' if operation == 'feed' else b''))
                        self.assertEqual(status, 500 if exception is RuntimeError else 400)
                        self.assertEqual(mutated, [True])
                        self.assertNotIn('test', server.sessions)
                        self.assertEqual(self.run_request(self.post('/api/stream/feed?session=test', b'1234'))[0], 404)
                    self.assertFalse(server.model_lock.locked())


if __name__ == '__main__':
    unittest.main()
