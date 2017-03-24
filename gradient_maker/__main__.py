"""Runs an instance of gradient_maker."""

import argparse

from aiohttp import web

from gradient_maker.app import app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1', help='The host to bind to (127.0.0.1 '
                        'to limit access to) localhost, 0.0.0.0 to allow from anywhere).')
    parser.add_argument('--port', type=int, default=8000, help='The HTTP port to use.')
    args = parser.parse_args()

    web.run_app(app, host=args.host, port=args.port)
