"""The ``facevision`` command: the same as FaceRecognizer, from the terminal.

    facevision capture --name armando --shots 20   registra con la cámara
    facevision add --name ana foto1.jpg foto2.jpg  registra desde fotos
    facevision people                              quién está registrado
    facevision remove --name ana                   borra a alguien
    facevision run                                 reconocimiento en vivo
    facevision recognize foto.jpg --out res.jpg    reconoce las caras de una foto
    facevision train                               entrena ya (si no, se hace solo)
    facevision evaluate data/orl_faces             compara los métodos en una base
    facevision show-eigenfaces --out ef.png        visualiza las eigenfaces

Todo se guarda en la carpeta --folder (por defecto la actual): las fotos en
faces/<nombre>/ y el modelo en model.npz. El modelo se reentrena solo cuando
cambian las fotos registradas.
"""

from __future__ import annotations

import argparse

import numpy as np

from .models import METHODS


def _rec(args):
    from .recognizer import FaceRecognizer

    return FaceRecognizer(args.folder, method=args.method, threshold=getattr(args, "threshold", None))


def cmd_capture(args) -> None:
    rec = _rec(args)
    print(f"Capturando {args.shots} fotos de '{args.name}'. Mueve un poco la cabeza. q para terminar.")
    saved = rec.capture(args.name, args.shots, args.camera)
    print(f"Se guardaron {saved} fotos. {rec}")


def cmd_add(args) -> None:
    rec = _rec(args)
    print(f"Se guardaron {rec.add_person(args.name, args.images)} fotos de {args.name}. {rec}")


def cmd_people(args) -> None:
    people = _rec(args).people
    if not people:
        print("No hay nadie registrado.")
    for name, n in people.items():
        print(f"  {name:<20} {n} fotos")


def cmd_remove(args) -> None:
    rec = _rec(args)
    rec.remove_person(args.name)
    print(f"Se borró a {args.name}. {rec}")


def _summary(info: dict) -> str:
    rule = {"eer": "calibrado con las personas registradas",
            "sessions": "calibrado con tus sesiones de captura",
            "default": "valor por defecto"}.get(info.get("rule"), "")
    msg = (f"{info['photos']} fotos de {len(info['people'])} personas. "
           f"Umbral {info['threshold']:.3f} ({rule})")
    if info.get("accepts_registered") is not None:
        where = "en una sesión que el modelo no vio" if info.get("sessions") else "en validación cruzada"
        msg += f"; {where} acepta {info['accepts_registered']:.0%} de tus fotos"
    return msg + "."


def cmd_train(args) -> None:
    info = _rec(args).train()
    print(_summary(info))
    if info["note"]:
        print(f"nota: {info['note']}")


def cmd_run(args) -> None:
    rec = _rec(args)
    print(f"{rec}. Umbral {rec.threshold:.3f}. Pulsa q para salir.")
    rec.live(args.camera)


def cmd_recognize(args) -> None:
    rec = _rec(args)
    faces = rec.recognize(args.image)
    print(f"{len(faces)} cara(s), umbral {rec.threshold:.3f}")
    for f in faces:
        print(f"  {f.name or 'desconocido':<15} distancia {f.distance:.3f}  recuadro {f.box}")
    if args.out:
        import cv2

        cv2.imwrite(args.out, rec.draw(args.image, faces))
        print(f"Imagen guardada en {args.out}")


def cmd_evaluate(args) -> None:
    from .evaluation import evaluate

    methods = [args.only] if args.only else METHODS
    evaluate(args.root, methods, seeds=args.seeds)


def cmd_show_eigenfaces(args) -> None:
    import cv2

    from .models import pca
    from .features import compute_cached
    from .preprocessing import FACE_SIZE, load_generic

    crops, _ = load_generic()
    _, C, _ = pca(compute_cached("pixels", crops), 16)
    w, h = FACE_SIZE
    tiles = []
    for i in range(C.shape[1]):
        c = C[:, i].reshape(h, w)
        tiles.append(((c - c.min()) / (np.ptp(c) + 1e-12) * 255).astype(np.uint8))
    grid = np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(args.out, grid)
    print(f"Las primeras {len(tiles)} eigenfaces (de las caras genéricas) en {args.out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="facevision", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", default=".", help="carpeta de datos (default: la actual)")
    p.add_argument("--method", choices=METHODS, default="combined",
                   help="método de reconocimiento (default: combined)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="registra a una persona con la cámara")
    c.add_argument("--name", required=True)
    c.add_argument("--shots", type=int, default=20)
    c.add_argument("--camera", type=int, default=0)
    c.set_defaults(fn=cmd_capture)

    a = sub.add_parser("add", help="registra a una persona desde fotos")
    a.add_argument("--name", required=True)
    a.add_argument("images", nargs="+")
    a.set_defaults(fn=cmd_add)

    sub.add_parser("people", help="lista a las personas registradas").set_defaults(fn=cmd_people)

    r = sub.add_parser("remove", help="borra a una persona y sus fotos")
    r.add_argument("--name", required=True)
    r.set_defaults(fn=cmd_remove)

    sub.add_parser("train", help="entrena ahora (si no, se hace solo al reconocer)").set_defaults(fn=cmd_train)

    v = sub.add_parser("run", help="reconocimiento en vivo con la cámara")
    v.add_argument("--camera", type=int, default=0)
    v.add_argument("--threshold", type=float, default=None)
    v.set_defaults(fn=cmd_run)

    g = sub.add_parser("recognize", help="reconoce las caras de una foto")
    g.add_argument("image")
    g.add_argument("--out", default=None, help="guardar la foto con las caras marcadas")
    g.add_argument("--threshold", type=float, default=None)
    g.set_defaults(fn=cmd_recognize)

    e = sub.add_parser("evaluate", help="compara los métodos en una base root/<persona>/<fotos>")
    e.add_argument("root", nargs="?", default="data/orl_faces/train")
    e.add_argument("--seeds", type=int, default=5)
    e.add_argument("--only", choices=METHODS, default=None, help="evaluar solo este método")
    e.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("show-eigenfaces", help="guarda una imagen con las eigenfaces")
    s.add_argument("--out", default="eigenfaces.png")
    s.set_defaults(fn=cmd_show_eigenfaces)

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except (RuntimeError, ValueError, FileNotFoundError, KeyError) as e:
        raise SystemExit(f"error: {e}")
