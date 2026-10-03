"""Capture a macOS audio input and print CW copy from the local neural service."""

import argparse
from contextlib import closing
import json
import math
import os
import select
import shutil
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise ValueError('Decoder redirects are forbidden; audio must stay on this computer.')


def local_url(value):
    url = urlsplit(value)
    if (url.scheme != 'http' or url.hostname not in {'localhost', '127.0.0.1', '::1'}
            or url.username is not None or url.password is not None
            or url.path not in {'', '/'} or url.query or url.fragment):
        raise ValueError('Use a local HTTP origin, such as http://127.0.0.1:8787.')
    host = '[::1]' if url.hostname == '::1' else '127.0.0.1'
    return f'http://{host}:{url.port or 80}'


def request(url, action, params, body=b''):
    req = Request(f'{url}/api/stream/{action}?{urlencode(params)}', data=body,
                  headers={'Content-Type': 'application/octet-stream'}, method='POST')
    try:
        with build_opener(ProxyHandler({}), NoRedirect()).open(req, timeout=15) as response:
            data = json.loads(response.read(65536))
        if not isinstance(data, dict):
            raise ValueError('The local decoder returned an invalid response.')
        return data
    except HTTPError as error:
        try:
            detail = json.loads(error.read(4096)).get('error', str(error))
        except (ValueError, AttributeError):
            detail = str(error)
        raise RuntimeError(f'Decoder: {detail}') from None
    except (URLError, TimeoutError) as error:
        raise RuntimeError(f'Cannot reach the local decoder at {url}: {error}') from None


def pcm_chunks(input_index, seconds):
    command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 'avfoundation',
               '-i', f':{input_index}', '-t', str(seconds), '-ac', '1', '-ar', '8000',
               '-acodec', 'pcm_f32le', '-f', 'f32le', 'pipe:1']
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    deadline = time.monotonic() + seconds
    pending = bytearray()
    remaining = seconds * 32000
    try:
        while remaining > 0 and time.monotonic() < deadline:
            ready, _, _ = select.select([process.stdout], [], [], min(0.25, max(0, deadline - time.monotonic())))
            if not ready:
                continue
            block = os.read(process.stdout.fileno(), min(32000 - len(pending), remaining))
            if not block:
                code = process.wait(timeout=2)
                if code:
                    raise RuntimeError(f'ffmpeg exited with status {code}. Check the input index and Terminal microphone permission.')
                if len(pending) % 4:
                    raise RuntimeError('Audio capture ended in the middle of a float32 sample.')
                break
            pending.extend(block)
            remaining -= len(block)
            if len(pending) == 32000:
                yield bytes(pending)
                pending.clear()
        if pending:
            complete = len(pending) // 4 * 4
            if complete:
                yield bytes(pending[:complete])
    finally:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        finally:
            process.stdout.close()


def listen(args):
    params = {'frequency': args.frequency}
    if args.bandwidth is not None:
        params['bandwidth'] = args.bandwidth
    session = request(args.url, 'start', params).get('session')
    if not isinstance(session, str) or not session:
        raise RuntimeError('The local decoder did not open a session.')
    params['session'] = session
    previous = ''

    def show(response):
        nonlocal previous
        text = response.get('text')
        if not isinstance(text, str):
            raise RuntimeError('The local decoder returned invalid copy.')
        if text != previous:
            print(text[len(previous):] if text.startswith(previous) else '\n' + text, end='', flush=True)
            previous = text

    try:
        print(f'Listening to audio input {args.input_index} for up to {args.seconds}s at {args.frequency:g} Hz. Ctrl-C stops.', file=sys.stderr)
        captured = 0
        nonzero_audio = False
        interrupted = False
        try:
            with closing(pcm_chunks(args.input_index, args.seconds)) as chunks:
                for block in chunks:
                    captured += len(block)
                    nonzero_audio = nonzero_audio or any(memoryview(block).cast('f'))
                    show(request(args.url, 'feed', params, block))
        except KeyboardInterrupt:
            interrupted = True
        if not captured and not interrupted:
            print('No audio arrived. Check the input index, audio routing to BlackHole, and Terminal microphone permission.', file=sys.stderr)
        elif captured and not nonzero_audio:
            print('Audio arrived, but every sample was digital zero. Check that the radio audio is playing and routed to the selected BlackHole input.', file=sys.stderr)
        final = request(args.url, 'finish', params)
        session = None
        show(final)
        if interrupted:
            raise KeyboardInterrupt
    finally:
        if session:
            try:
                request(args.url, 'cancel', params)
            except (Exception, KeyboardInterrupt) as error:
                print(f'Could not cancel the decoder session: {error}. It expires after 120s idle.', file=sys.stderr)
        print(flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog='macOS only. Select BlackHole using --list-inputs. Audio is never saved.')
    parser.add_argument('--list-inputs', action='store_true', help='List AVFoundation audio device indices and exit.')
    parser.add_argument('--input-index', type=int, help='Audio input index from --list-inputs (for example, 1 for BlackHole).')
    parser.add_argument('--frequency', type=float, default=711, help='CW carrier in Hz, 250–1400 (default: 711).')
    parser.add_argument('--bandwidth', type=float, help='Filter width in Hz, 40–500 (default: selected engine recommendation).')
    parser.add_argument('--seconds', type=int, default=180, help='Capture limit, 1–600 seconds (default: 180).')
    parser.add_argument('--url', default='http://127.0.0.1:8787', help='Local decoder HTTP origin.')
    args = parser.parse_args(argv)
    if sys.platform != 'darwin':
        parser.error('This capture command supports macOS AVFoundation only.')
    if not shutil.which('ffmpeg'):
        parser.error('ffmpeg is required. Install it with brew install ffmpeg.')
    try:
        args.url = local_url(args.url)
        if args.list_inputs:
            subprocess.run(['ffmpeg', '-hide_banner', '-f', 'avfoundation', '-list_devices', 'true', '-i', ''], timeout=10)
            return 0
        if args.input_index is None or args.input_index < 0:
            parser.error('Choose --input-index using --list-inputs.')
        if not 1 <= args.seconds <= 600:
            parser.error('--seconds must be between 1 and 600.')
        if not math.isfinite(args.frequency) or not 250 <= args.frequency <= 1400:
            parser.error('--frequency must be between 250 and 1400 Hz.')
        if args.bandwidth is not None and (not math.isfinite(args.bandwidth) or not 40 <= args.bandwidth <= 500):
            parser.error('--bandwidth must be between 40 and 500 Hz.')
        listen(args)
        return 0
    except KeyboardInterrupt:
        print('Capture stopped.', file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f'Capture failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
