"""An aiohttp web application."""

import asyncio
import json
from pathlib import Path

import aiohttp
from aiohttp import web
import aiohttp_jinja2
import jinja2
from ucs.constants import floatX

from gradient_maker.gradient import Gradient
from gradient_maker.parser import Parser, ParseException


Gradient.compile()

PACKAGE = Path(__file__).resolve().parent

loop = asyncio.get_event_loop()

app = web.Application()
aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(str(PACKAGE/'templates')))
app.tasks = []


@aiohttp_jinja2.template('index.html')
def root_handler(request):
    return {}


async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            msg = json.loads(msg.data)
            if msg['_'] == 'gradRequest':
                def send(msg):
                    loop.call_soon_threadsafe(ws.send_json, msg)
                task = loop.run_in_executor(None, grad_request, msg['text'], send)
                app.tasks.append(task)
            else:
                continue

        else:
            break

    return ws


def grad_request(text, send):
    parser = Parser()
    try:
        parser.parse(text)
    except ParseException as err:
        send({'_': 'error', 'text': str(err)})
        return

    if len(parser.grad_points) < 2:
        error_text = 'At least two colors are required.'
        send({'_': 'error', 'text': error_text})
        return
    else:
        x = [point[0] for point in parser.grad_points]
        y = [point[1] for point in parser.grad_points]
        g = Gradient(x, floatX(y) / 255)
        x_out, y_out, s = g.make_gradient(steps=30,
                                          callback=lambda x: send({'_': 'progress', 'text': x}))
        send({'_': 'progress', 'text': s})
        send({'_': 'result', 'text': g.to_html(x_out, y_out)})

app.router.add_get('/', root_handler)
app.router.add_static('/static', PACKAGE/'static')
app.router.add_get('/websocket', websocket_handler)
