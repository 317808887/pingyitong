"""Create reproducible multilingual desktop inputs; never calls a model or controls UI.

References are authored equivalents, not exact-string scoring rules. A reviewer
must judge meaning, including the attached invariants, before marking a result.
Randomization covers language direction, content, quantities, layout and style.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

LANGUAGES = ("en", "zh", "ja", "de", "fr", "es")
PHRASES = {
    "before": {
        "en": "Save the file before closing the window.",
        "zh": "关闭窗口前，请保存文件。",
        "ja": "ウィンドウを閉じる前に、ファイルを保存してください。",
        "de": "Speichern Sie die Datei, bevor Sie das Fenster schließen.",
        "fr": "Enregistrez le fichier avant de fermer la fenêtre.",
        "es": "Guarde el archivo antes de cerrar la ventana.",
    },
    "until": {
        "en": "Do not disconnect the cable until the transfer is complete.",
        "zh": "传输完成前，请勿断开电缆。",
        "ja": "転送が完了するまで、ケーブルを抜かないでください。",
        "de": "Trennen Sie das Kabel nicht, bevor die Übertragung abgeschlossen ist.",
        "fr": "Ne débranchez pas le câble avant la fin du transfert.",
        "es": "No desconecte el cable hasta que termine la transferencia.",
    },
    "quantity": {
        "en": "The report contains {n} pages and {m} tables.",
        "zh": "报告包含{n}页和{m}张表格。",
        "ja": "レポートには{n}ページと{m}個の表が含まれています。",
        "de": "Der Bericht enthält {n} Seiten und {m} Tabellen.",
        "fr": "Le rapport contient {n} pages et {m} tableaux.",
        "es": "El informe contiene {n} páginas y {m} tablas.",
    },
    "authorization": {
        "en": "Only administrators can delete these files.",
        "zh": "只有管理员可以删除这些文件。",
        "ja": "これらのファイルを削除できるのは管理者だけです。",
        "de": "Nur Administratoren können diese Dateien löschen.",
        "fr": "Seuls les administrateurs peuvent supprimer ces fichiers.",
        "es": "Solo los administradores pueden eliminar estos archivos.",
    },
    "relation": {
        "en": "The blue button is above the red button.",
        "zh": "蓝色按钮位于红色按钮上方。",
        "ja": "青いボタンは赤いボタンの上にあります。",
        "de": "Die blaue Schaltfläche befindet sich über der roten Schaltfläche.",
        "fr": "Le bouton bleu se trouve au-dessus du bouton rouge.",
        "es": "El botón azul está encima del botón rojo.",
    },
    "condition": {
        "en": "If the test fails, do not install the update.",
        "zh": "如果测试失败，请勿安装更新。",
        "ja": "テストが失敗した場合は、更新をインストールしないでください。",
        "de": "Wenn der Test fehlschlägt, installieren Sie das Update nicht.",
        "fr": "Si le test échoue, n’installez pas la mise à jour.",
        "es": "Si la prueba falla, no instale la actualización.",
    },
}
INVARIANTS = {
    "before": "保存文件先于关闭窗口；不能把关闭译为保存的对象。",
    "until": "传输完成之前禁止断开；否定针对断开，不针对传输。",
    "quantity": "页数为{n}，表格数为{m}；实体及数量不能互换。",
    "authorization": "删除权限仅属于管理员；不是所有人都能删除。",
    "relation": "蓝在红上方；颜色和上下关系均须保留。",
    "condition": "测试失败是条件；禁止的动作是安装更新。",
}


def generate(seed: int, count: int = 9, run_id: str = "preview") -> dict:
    if count < 3:
        raise ValueError("At least three scenes are needed to cover scene classes")
    rng = random.Random(seed)
    languages = list(LANGUAGES)
    rng.shuffle(languages)
    scenes = []
    for index in range(count):
        source = languages[index % len(languages)]
        target = rng.choice([language for language in LANGUAGES if language != source])
        category = ("subtitle", "dense_page", "technical_document")[index % 3]
        categories = list(PHRASES)
        rng.shuffle(categories)
        if category == "subtitle":
            categories = categories[:2]
        rows = []
        for row_index, semantic in enumerate(categories):
            quantities = {"n": rng.randint(12, 87), "m": rng.randint(2, 9)}
            rows.append({"id": f"s{index + 1:02d}-r{row_index + 1:02d}", "semantic": semantic,
                         "source": PHRASES[semantic][source].format(**quantities),
                         "reference": PHRASES[semantic][target].format(**quantities),
                         "invariants": INVARIANTS[semantic].format(**quantities),
                         "review": "pending"})
        font_size = rng.choice((13, 15, 18)) if category == "dense_page" else rng.choice((18, 21, 24))
        color = rng.choice(("#ffffff", "#c7c7c7", "#8b8b8b"))
        # One source alphabet per scene: changing the direction is a real UI step.
        geometry = [120, 380, 1120, 330]
        step = 47 if len(rows) > 2 else 100
        controls = {"text": "", "run_id": run_id, "scene_id": f"scene-{index + 1:02d}",
                    "font_size": font_size, "foreground": color, "background": "#17191d",
                    "geometry": geometry,
                    "blocks": [{"text": row["source"], "geometry": [25, 12 + n * step, 1070, 90 if category == "subtitle" else 44]}
                               for n, row in enumerate(rows)]}
        scenes.append({"scene_id": controls["scene_id"], "category": category,
                       "source_language": source, "target_language": target, "rows": rows,
                       "control": controls, "latency_target_ms": 600 if category != "technical_document" else None})
    identity_scenes = [{**scene, "control": {key: value for key, value in scene["control"].items() if key != "run_id"}}
                       for scene in scenes]
    identity = json.dumps(identity_scenes, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {"schema": "pingyitong-random-scenes-v1", "seed": seed, "run_id": run_id,
            "corpus_sha256": hashlib.sha256(identity).hexdigest(), "scenes": scenes,
            "scope": "Seeded combinations of six authored semantic families in six languages; not unrestricted language coverage.",
            "instructions": "Use actual installed app with the scene's explicit source/target languages. Record full and compact outputs and human meaning review. Keep known bug regressions separately."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=9)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite a scene corpus; use another output path")
    corpus = generate(args.seed, args.count, args.run_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seed": args.seed, "scenes": len(corpus["scenes"]),
                      "corpus_sha256": corpus["corpus_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
