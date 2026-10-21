"""Read only newly generated, known synthetic desktop test translations."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    database = Path(os.environ["LOCALAPPDATA"]) / "PingYiTong/data/history.db"
    allowed = ["The door is open.", "Please save the file.", "请保存文件。", "然后关闭窗口。",
               "Click Settings.", "Then close the window.", "安装完成。", "Close the window."]
    with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id,created_at,source_lang,target_lang,source_text,target_text,model_id,from_cache "
            "FROM translation_history WHERE created_at >= ? ORDER BY id", (args.since,),
        ).fetchall()
    # Only the exact synthetic lines belong to this test. A substring match could
    # accidentally include a user's document after the fixture window closes.
    records = [dict(row) for row in rows if row["source_text"].strip() in allowed]
    if args.output:
        args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
