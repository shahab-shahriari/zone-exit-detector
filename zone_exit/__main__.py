import argparse

from .pipeline import load_config, run
from .service import serve
from .zone_tool import draw_zone


def main() -> None:
    parser = argparse.ArgumentParser(prog="zone_exit", description="Zone-exit vehicle detector")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="process a video file, camera index or RTSP/HTTP stream")
    p_run.add_argument("--source", required=True)
    p_run.add_argument("--config", default="config.yaml")
    p_run.add_argument("--show", action="store_true", help="show a preview window (Esc to stop)")
    p_run.add_argument("--save-video", help="write an annotated video to this path")

    p_serve = sub.add_parser("serve", help="start the HTTP service (sessions per call and camera)")
    p_serve.add_argument("--config", default="config.yaml")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--no-auth", action="store_true", help="disable the bearer token (local use only)")

    p_zone = sub.add_parser("zone", help="draw the zone polygon on the first frame")
    p_zone.add_argument("--source", required=True)
    p_zone.add_argument("--out", default="zone.json")

    args = parser.parse_args()
    if args.cmd == "run":
        run(load_config(args.config), args.source, show=args.show, save_video=args.save_video)
    elif args.cmd == "serve":
        serve(load_config(args.config), args.host, args.port, args.no_auth)
    else:
        draw_zone(args.source, args.out)


if __name__ == "__main__":
    main()
