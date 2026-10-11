import ast
import pathlib
import sqlite3
import sys

session, module_path, class_name = sys.argv[1:4]
tree = ast.parse(pathlib.Path(module_path).read_text())
node = next(
    n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name
)
connection = sqlite3.connect(session)
outside = [
    job_id
    for job_id, row in connection.execute(
        "select job_id, start_pos_row from mutation_specs"
    )
    if not node.lineno <= row <= node.end_lineno
]
for job_id in outside:
    connection.execute("delete from mutation_specs where job_id = ?", (job_id,))
    connection.execute("delete from work_items where job_id = ?", (job_id,))
connection.commit()
remaining = connection.execute("select count(*) from work_items").fetchone()[0]
print(f"{class_name}: {remaining} mutants kept, {len(outside)} outside lines {node.lineno}-{node.end_lineno} dropped")
