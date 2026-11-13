"""A TCP forwarder on 127.0.0.1 that delays every chunk one way: a tunnel to a far machine, in a test."""

import asyncio
import socket
import threading


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tunnel(target_port: int, delay_s: float) -> int:
    """Forward a new 127.0.0.1 port to ``target_port``, ``delay_s`` each way; returns the new port."""
    port, ready = free_port(), threading.Event()

    async def pipe(r, w):
        try:
            while data := await r.read(65536):
                await asyncio.sleep(delay_s)
                w.write(data)
                await w.drain()
        except ConnectionError:
            pass
        finally:
            w.close()

    async def handle(cr, cw):
        sr, sw = await asyncio.open_connection("127.0.0.1", target_port)
        await asyncio.gather(pipe(cr, sw), pipe(sr, cw))

    async def main():
        server = await asyncio.start_server(handle, "127.0.0.1", port)
        ready.set()
        async with server:
            await server.serve_forever()

    threading.Thread(target=lambda: asyncio.run(main()), daemon=True).start()
    ready.wait(5)
    return port
