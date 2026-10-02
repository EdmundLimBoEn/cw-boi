"""Offline checks for local capture transport; never opens an audio device."""

from contextlib import redirect_stdout, redirect_stderr
from io import BytesIO, StringIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import listen


def rejected(function, error=ValueError):
    try:
        function()
    except error:
        return
    raise AssertionError('Expected rejection')


def run_capture(reads, fail=None, clock=None, waits=None):
    process = Mock()
    process.stdout.fileno.return_value = 123
    process.poll.return_value = None
    process.wait.return_value = 0
    process.wait.side_effect = waits
    calls = []
    feeds = 0

    def request(url, action, params, body=b''):
        nonlocal feeds
        calls.append((action, params.copy(), body))
        if action == 'start':
            return {'session': 'local-session'}
        assert params['session'] == 'local-session'
        if action == 'feed':
            feeds += 1
            if fail:
                raise fail
            return {'text': 'CQ' if feeds == 1 else 'CQ DE'}
        if action == 'finish':
            process.stdout.close.assert_called_once()
        return {'text': 'CQ DE K' if action == 'finish' else ''}

    args = SimpleNamespace(input_index=1, seconds=2, frequency=711, bandwidth=80, url='http://127.0.0.1:8787')
    output = StringIO()
    errors = StringIO()
    failure = None
    with (patch.object(listen.subprocess, 'Popen', return_value=process) as spawn,
          patch.object(listen.select, 'select', return_value=([process.stdout], [], [])),
          patch.object(listen.os, 'read', side_effect=reads),
          patch.object(listen.time, 'monotonic', side_effect=clock or (lambda: 0)),
          patch.object(listen, 'request', side_effect=request),
          redirect_stdout(output), redirect_stderr(errors)):
        try:
            listen.listen(args)
        except (RuntimeError, KeyboardInterrupt) as error:
            failure = error
    process.stdout.close.assert_called_once()
    command = spawn.call_args.args[0]
    assert command[command.index('-i') + 1] == ':1'
    assert 'pipe:1' in command and 'f32le' in command
    return calls, process, output.getvalue(), failure, errors.getvalue()


if __name__ == '__main__':
    assert listen.local_url('http://localhost:8787/') == 'http://127.0.0.1:8787'
    assert listen.local_url('http://[::1]:8787') == 'http://[::1]:8787'
    for value in ('https://127.0.0.1', 'http://example.com', 'http://user@127.0.0.1',
                  'http://127.0.0.1:99999', 'http://127.0.0.1/api', 'http://localhost/?target=elsewhere'):
        rejected(lambda: listen.local_url(value))
    rejected(lambda: listen.NoRedirect().redirect_request())
    opener = Mock()
    opener.open.return_value = BytesIO(b'{"session":"local-session"}')
    with patch.object(listen, 'build_opener', return_value=opener) as build:
        assert listen.request('http://127.0.0.1:8787', 'start', {'frequency': 711})['session'] == 'local-session'
        handlers = build.call_args.args
        assert handlers[0].proxies == {} and isinstance(handlers[1], listen.NoRedirect)
        sent = opener.open.call_args.args[0]
        assert sent.method == 'POST' and sent.full_url == 'http://127.0.0.1:8787/api/stream/start?frequency=711'

    parts = [b'\0' * n for n in (3, 1, 25000, 6996, 6004, 0)]
    calls, process, text, failure, errors = run_capture(parts)
    assert failure is None and text == 'CQ DE K\n'
    assert 'No audio arrived' not in errors
    assert 'every sample was digital zero' in errors
    assert [action for action, _, _ in calls] == ['start', 'feed', 'feed', 'finish']
    assert b''.join(body for action, _, body in calls if action == 'feed') == b'\0' * 38004
    assert all(len(body) % 4 == 0 for action, _, body in calls if action == 'feed')
    _, _, _, failure, errors = run_capture([b'\0\0\0?' * 8000, b''])
    assert failure is None and 'digital zero' not in errors

    for failure_in in (RuntimeError('Decoder unavailable'), KeyboardInterrupt()):
        calls, process, text, failure, _ = run_capture([b'\0' * 32000], fail=failure_in)
        if isinstance(failure_in, KeyboardInterrupt):
            assert isinstance(failure, KeyboardInterrupt) and text == 'CQ DE K\n'
            assert [action for action, _, _ in calls] == ['start', 'feed', 'finish']
        else:
            assert failure is failure_in and calls[-1][0] == 'cancel'
        process.terminate.assert_called_once()
        process.wait.assert_called_once()

    calls, process, text, failure, _ = run_capture([KeyboardInterrupt()])
    assert isinstance(failure, KeyboardInterrupt) and text == 'CQ DE K\n'
    assert [action for action, _, _ in calls] == ['start', 'finish']
    process.terminate.assert_called_once()

    calls, process, _, failure, errors = run_capture([], clock=[0, 3])
    assert failure is None and [action for action, _, _ in calls] == ['start', 'finish']
    assert 'No audio arrived' in errors and 'Terminal microphone permission' in errors
    process.terminate.assert_called_once()

    calls, _, _, failure, _ = run_capture([b'\0' * 3, b''])
    assert isinstance(failure, RuntimeError) and calls[-1][0] == 'cancel'
    calls, process, _, failure, _ = run_capture([b'\0' * 32000], fail=RuntimeError('Disconnected'),
                                               waits=[listen.subprocess.TimeoutExpired('ffmpeg', 2), 0])
    assert isinstance(failure, RuntimeError) and calls[-1][0] == 'cancel'
    process.kill.assert_called_once()
    with (patch.object(listen, 'listen', side_effect=KeyboardInterrupt),
          patch.object(listen.sys, 'platform', 'darwin'),
          patch.object(listen.shutil, 'which', return_value='/usr/local/bin/ffmpeg'),
          redirect_stderr(StringIO())):
        assert listen.main(['--input-index', '1']) == 130
    print('Local-only transport, partial reads, incremental copy, timeout and interruption cleanup passed.')
